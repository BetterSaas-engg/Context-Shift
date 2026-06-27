"""
Slice 2 – Prove we can read the model's internal hidden state.
We run one sentence through GPT-2, then inspect the hidden state
at a middle layer to understand its shape.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Config ───────────────────────────────────────────────────────────
MODEL_NAME = "gpt2"

# ── Load model & tokenizer ──────────────────────────────────────────
# output_hidden_states=True tells the model to return the internal
# activations at every layer, not just the final prediction.
print(f"Loading model: {MODEL_NAME} (with hidden states enabled) ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, output_hidden_states=True
).to("cpu")
print("Model loaded.\n")

# ── Define a sentence ───────────────────────────────────────────────
sentence = "Please find attached the requested documentation."

# ── Run a forward pass ──────────────────────────────────────────────
# We only need to read the hidden states, not generate new text,
# so a single forward pass is enough.
# torch.no_grad() tells PyTorch we won't be training — saves memory
# and makes it faster.
input_ids = tokenizer.encode(sentence, return_tensors="pt")

with torch.no_grad():
    outputs = model(input_ids)

# ── Inspect hidden states ───────────────────────────────────────────
# outputs.hidden_states is a tuple with one tensor per layer.
# Layer 0 is the raw token embeddings (before any transformer block).
# Layers 1–12 are the outputs of each of the 12 transformer blocks.
# So there are 13 entries total: embedding layer + 12 blocks.
hidden_states = outputs.hidden_states
print(f"Number of hidden-state layers returned: {len(hidden_states)}")
print(f"  (That's 1 embedding layer + {len(hidden_states) - 1} transformer blocks)\n")

# ── Look at the middle layer (layer 6) ─────────────────────────────
# Layer 6 is the output of the 6th transformer block — roughly the
# middle of the network.
layer_6 = hidden_states[6]
print(f"Shape of layer-6 hidden state: {tuple(layer_6.shape)}")
print()
print("What each number means:")
print(f"  {layer_6.shape[0]}   = batch size (we fed in 1 sentence)")
print(f"  {layer_6.shape[1]}   = number of tokens in that sentence")
print(f"  {layer_6.shape[2]} = features per token (GPT-2 small uses 768-dimensional vectors)")
