"""
Slice 5 – Sweep the steering coefficient across a ladder of values.
We print the same prompt at each level so we can read the tone shift
and spot the sweet spot (where formality improves) vs the cliff
(where the output collapses into nonsense).
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Config ───────────────────────────────────────────────────────────
MODEL_NAME = "gpt2-medium"

# ── Load model & tokenizer ──────────────────────────────────────────
print(f"Loading model: {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cpu")
tokenizer.pad_token = tokenizer.eos_token
print("Model loaded.\n")

# ── Load the steering vector ────────────────────────────────────────
steering_vector = torch.load("formal_vector.pt", weights_only=True)

# ── Hook setup ──────────────────────────────────────────────────────
# We store the coefficient in a mutable container (a list) so the hook
# can read a new value each time without needing to be re-registered.
current_coeff = [0.0]


def steering_hook(module, input, output):
    """Add coefficient * steering_vector to the hidden state in-place."""
    if current_coeff[0] == 0.0:
        return  # No modification needed for the baseline.
    output[0].add_(current_coeff[0] * steering_vector)


# Attach to transformer block 17 (its output = hidden_states[18]).
hook_handle = model.transformer.h[17].register_forward_hook(steering_hook)

# ── Prompt & coefficient ladder ─────────────────────────────────────
prompt = "My thoughts on the new policy are"
inputs = tokenizer(prompt, return_tensors="pt")

# Coefficient 0 is the unsteered baseline — the model's natural output.
# As the coefficient climbs we expect the tone to shift toward formal.
# Past a certain point the signal overwhelms the model and output
# quality collapses — that's the cliff.
coefficients = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8]

print(f"Prompt: \"{prompt}\"\n")
print("=" * 64)

for coeff in coefficients:
    current_coeff[0] = coeff
    output_ids = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=False,
    )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    label = f"coefficient = {coeff}"
    if coeff == 0:
        label += "  (unsteered baseline)"
    print(f"  {label}")
    print(f"  {text}")
    print("-" * 64)

# ── Clean up ─────────────────────────────────────────────────────────
hook_handle.remove()
