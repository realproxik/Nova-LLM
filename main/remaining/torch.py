# nova_torch.py
# NOVA AI - PyTorch Integration Layer
# Copyright (c) NOVA AI. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from typing import Optional, List, Dict, Any, Tuple, Union
import os
import json
import time
import math
from dataclasses import dataclass, field

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class NovaTorchConfig:
    vocab_size: int = 50272
    hidden_size: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    max_seq_len: int = 8192
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    bias: bool = False
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    moe_num_experts: int = 8
    moe_top_k: int = 2
    use_flash_attn: bool = True
    use_paged_attn: bool = False
    use_speculative: bool = False
    dtype: torch.dtype = torch.bfloat16
    device: str = "cuda"

# ============================================================================
# RMS NORM
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight

# ============================================================================
# ROTARY EMBEDDING
# ============================================================================

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos', emb.cos())
        self.register_buffer('sin', emb.sin())

    def forward(self, x: torch.Tensor, pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[1]
        cos = self.cos[pos:pos+seq_len]
        sin = self.sin[pos:pos+seq_len]
        return cos, sin

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1).unsqueeze(3)
    sin = sin.unsqueeze(1).unsqueeze(3)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# ============================================================================
# ATTENTION
# ============================================================================

class GroupedQueryAttention(nn.Module):
    def __init__(self, config: NovaTorchConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.scale = self.head_dim ** -0.5
        self.group_size = self.num_heads // self.num_kv_heads

        self.wq = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.bias)
        self.wk = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.bias)
        self.wv = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.bias)
        self.wo = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.bias)

        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len, config.rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self.wq(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(x, 0)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        if self.group_size > 1:
            k = k.repeat_interleave(self.group_size, dim=1)
            v = v.repeat_interleave(self.group_size, dim=1)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask

        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=0.0, training=self.training)

        output = attn @ v
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.wo(output)

# ============================================================================
# FLASH ATTENTION
# ============================================================================

class FlashAttention(nn.Module):
    def __init__(self, config: NovaTorchConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.scale = self.head_dim ** -0.5

        self.wq = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.bias)
        self.wk = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.bias)
        self.wv = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.bias)
        self.wo = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self.wq(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        if self.num_heads != self.num_kv_heads:
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

        # Flash Attention implementation
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)

        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.wo(output)

# ============================================================================
# MLP
# ============================================================================

class MLP(nn.Module):
    def __init__(self, config: NovaTorchConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = self._get_intermediate_size(config)

        self.gate = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.bias)
        self.up = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.bias)
        self.down = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.bias)
        self.act = nn.SiLU()

    def _get_intermediate_size(self, config: NovaTorchConfig) -> int:
        hidden = int(2 * 4 * config.hidden_size / 3)
        if config.ffn_dim_multiplier:
            hidden = int(config.ffn_dim_multiplier * hidden)
        return config.multiple_of * ((hidden + config.multiple_of - 1) // config.multiple_of)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.act(self.gate(x)) * self.up(x))

# ============================================================================
# MOE (Mixture of Experts)
# ============================================================================

class MoE(nn.Module):
    def __init__(self, config: NovaTorchConfig):
        super().__init__()
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_top_k
        self.hidden_size = config.hidden_size
        self.intermediate_size = int(2 * 4 * config.hidden_size / 3)

        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([
            MLP(config) for _ in range(self.num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden = x.shape
        x_flat = x.view(-1, hidden)

        logits = self.gate(x_flat)
        weights, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)

        output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (indices == i).any(dim=-1)
            if mask.any():
                expert_input = x_flat[mask]
                expert_output = expert(expert_input)
                expert_weight = weights[mask][indices[mask] == i].unsqueeze(-1)
                output[mask] += expert_output * expert_weight

        return output.view(batch, seq_len, hidden)

# ============================================================================
# TRANSFORMER BLOCK
# ============================================================================

class TransformerBlock(nn.Module):
    def __init__(self, config: NovaTorchConfig):
        super().__init__()
        if config.use_flash_attn:
            self.attention = FlashAttention(config)
        else:
            self.attention = GroupedQueryAttention(config)

        self.mlp = MoE(config) if config.moe_num_experts > 1 else MLP(config)

        self.attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), mask)
        out = h + self.mlp(self.ffn_norm(h))
        return out

# ============================================================================
# NOVA LLM MODEL
# ============================================================================

class NovaLLM(nn.Module):
    def __init__(self, config: NovaTorchConfig):
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        x = self.tok_embeddings(input_ids)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        logits = self.output(x)
        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self(input_ids)
                logits = logits[:, -1, :] / temperature

                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(F.softmax(logits, dim=-1), descending=True)
                    cumsum = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumsum > top_p
                    sorted_probs[mask] = 0
                    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                    logits = torch.zeros_like(logits).scatter_(-1, sorted_indices, sorted_probs)

                if top_k > 0:
                    top_k_vals, top_k_indices = torch.topk(logits, top_k, dim=-1)
                    logits = torch.zeros_like(logits).scatter_(-1, top_k_indices, top_k_vals)

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

# ============================================================================
# DISTRIBUTED TRAINING
# ============================================================================

class NovaTrainer:
    def __init__(
        self,
        model: NovaLLM,
        config: NovaTorchConfig,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        warmup_steps: int = 2000,
        total_steps: int = 100000,
        grad_accum: int = 8,
        max_grad_norm: float = 1.0,
        use_ddp: bool = False,
    ):
        self.model = model
        self.config = config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.grad_accum = grad_accum
        self.max_grad_norm = max_grad_norm
        self.use_ddp = use_ddp

        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        self.scaler = GradScaler()
        self.step = 0

        if use_ddp:
            self.model = DDP(model)

    def train_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        with autocast(dtype=self.config.dtype):
            logits = self.model(inputs)
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1))

        self.scaler.scale(loss).backward()

        if (self.step + 1) % self.grad_accum == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        self.step += 1

        return {"loss": loss.item(), "step": self.step}

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            metrics = self.train_step(inputs, targets)
            total_loss += metrics["loss"]

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Step {metrics['step']}, Loss: {metrics['loss']:.4f}")

        return {"loss": total_loss / len(dataloader)}

# ============================================================================
# DATASET
# ============================================================================

class NovaDataset(Dataset):
    def __init__(self, data: List[int], seq_len: int):
        self.data = data
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.data) // self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len
        x = torch.tensor(self.data[start:end], dtype=torch.long)
        y = torch.tensor(self.data[start+1:end+1], dtype=torch.long)
        return x, y

# ============================================================================
# LOAD/SAVE
# ============================================================================

def save_model(model: NovaLLM, path: str):
    torch.save(model.state_dict(), path)

def load_model(model: NovaLLM, path: str):
    model.load_state_dict(torch.load(path))

# ============================================================================
# CREATE MODEL
# ============================================================================

def create_model(config: NovaTorchConfig) -> NovaLLM:
    return NovaLLM(config)

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    config = NovaTorchConfig()
    model = create_model(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    input_ids = torch.randint(0, config.vocab_size, (2, 512))
    logits = model(input_ids)
    print(f"Logits shape: {logits.shape}")