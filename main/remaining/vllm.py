# example_vllm.py
from nova_vllm import NovaVLLM, ContinuousBatcher, PrefixCache
from nova_torch import NovaTorchConfig, create_model
import torch

# Create model
config = NovaTorchConfig()
model = create_model(config)

# Create tokenizer (simulated)
class SimpleTokenizer:
    def encode(self, text):
        return [ord(c) % 32000 for c in text]
    def decode(self, tokens):
        return ''.join([chr(t) for t in tokens if t < 256])

tokenizer = SimpleTokenizer()

# Create vLLM engine
engine = NovaVLLM(model, tokenizer, max_batch_size=16, max_seq_len=2048)

# Start engine
engine.start()

# Add requests
prompts = ["Hello world", "How are you", "What is AI"]
for prompt in prompts:
    engine.add_request(prompt, max_tokens=50)

# Wait for results
time.sleep(1)

# Stop engine
engine.stop()
print("Done")