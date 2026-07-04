# example_torch.py
from nova_torch import NovaTorchConfig, NovaLLM, NovaTrainer, NovaDataset, create_model
import torch
import numpy as np

# Create config
config = NovaTorchConfig()
config.vocab_size = 32000
config.hidden_size = 2048
config.num_layers = 16
config.num_heads = 16
config.max_seq_len = 2048

# Create model
model = create_model(config)
print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

# Create dummy data
data = np.random.randint(0, 32000, 1000000).tolist()
dataset = NovaDataset(data, seq_len=512)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

# Create trainer
trainer = NovaTrainer(model, config, learning_rate=3e-4)

# Train
for epoch in range(3):
    metrics = trainer.train_epoch(dataloader, epoch)
    print(f"Epoch {epoch}: Loss={metrics['loss']:.4f}")

# Save model
save_model(model, "nova_model.pt")
print("Model saved")

# Generate
input_ids = torch.randint(0, 32000, (1, 10))
output = model.generate(input_ids, max_new_tokens=20)
print(f"Generated: {output.shape}")