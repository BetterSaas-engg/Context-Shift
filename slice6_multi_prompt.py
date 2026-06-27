"""
Slice 6 – Multi-prompt test at a fixed coefficient.
Slice 5 showed the sweet spot on one prompt. This slice tests whether
the formality shift generalises across different prompts and topics,
or whether it only works on that one sentence.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Config ───────────────────────────────────────────────────────────
MODEL_NAME = "gpt2-medium"
COEFFICIENT = 0.2  # In the sweet spot found in slice 5.

# ── Load model & tokenizer ──────────────────────────────────────────
print(f"Loading model: {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cpu")
tokenizer.pad_token = tokenizer.eos_token
print("Model loaded.\n")

# ── Load the steering vector ────────────────────────────────────────
steering_vector = torch.load("formal_vector.pt", weights_only=True)

# ── Hook setup ──────────────────────────────────────────────────────
steer_enabled = False


def steering_hook(module, input, output):
    """Add COEFFICIENT * steering_vector to the hidden state in-place."""
    if not steer_enabled:
        return
    output[0].add_(COEFFICIENT * steering_vector)


# Attach to transformer block 17 (its output = hidden_states[18]).
hook_handle = model.transformer.h[17].register_forward_hook(steering_hook)

# ── Prompts ─────────────────────────────────────────────────────────
# Varied topics and styles to see if the formality vector generalises.
prompts = [
    "My thoughts on the new policy are",
    "The best way to fix this bug is",
    "Hey everyone, I just wanted to say",
    "After reviewing the data, I believe",
    "So basically what happened was",
    "In conclusion, the results suggest that",
    "I'm not sure about this, but I think",
    "Dear team, I wanted to update you on",
]

# ── Generate and compare ────────────────────────────────────────────
# For each prompt we generate twice (unsteered then steered) so we can
# read the difference side by side. Both use greedy decoding so the
# only variable is the steering vector.

print(f"Coefficient: {COEFFICIENT}")
print("=" * 70)

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt")

    # Unsteered
    steer_enabled = False
    unsteered_ids = model.generate(**inputs, max_new_tokens=40, do_sample=False)
    unsteered_text = tokenizer.decode(unsteered_ids[0], skip_special_tokens=True)

    # Steered
    steer_enabled = True
    steered_ids = model.generate(**inputs, max_new_tokens=40, do_sample=False)
    steered_text = tokenizer.decode(steered_ids[0], skip_special_tokens=True)

    print(f"  PROMPT:     {prompt}")
    print(f"  UNSTEERED:  {unsteered_text}")
    print(f"  STEERED:    {steered_text}")
    print("-" * 70)

# ── Clean up ─────────────────────────────────────────────────────────
hook_handle.remove()
