"""
Slice 4 – Inject the formality steering vector during generation
and compare steered vs unsteered output on the same prompt.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Config ───────────────────────────────────────────────────────────
MODEL_NAME = "gpt2"
COEFFICIENT = 4.0  # How strongly to push toward formality.

# ── Load model & tokenizer ──────────────────────────────────────────
print(f"Loading model: {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cpu")

# GPT-2 doesn't have a pad token by default. We set it to the
# end-of-sequence token so the attention mask works correctly.
tokenizer.pad_token = tokenizer.eos_token
print("Model loaded.\n")

# ── Load the steering vector ────────────────────────────────────────
# This is the 768-length formality direction we built in slice 3.
steering_vector = torch.load("formal_vector.pt", weights_only=True)
print(f"Steering vector loaded. Shape: {tuple(steering_vector.shape)}\n")

# ── Set up the forward hook ─────────────────────────────────────────
# A forward hook is a function that PyTorch calls automatically every
# time a layer finishes its forward pass. It lets us intercept the
# model mid-computation and modify the values flowing through.
#
# We attach this hook to model.transformer.h[5] — the 6th transformer
# block (index 5). Its output becomes hidden_states[6], which is
# exactly where we extracted the vector in slice 3. Injecting at the
# same place the vector was measured keeps things consistent.
#
# The flag lets us turn steering on/off without removing the hook.
steer_enabled = False


def steering_hook(module, input, output):
    """
    Called after transformer block h[5] finishes.
    output is a tuple; the first element is the hidden-state tensor
    with shape [batch, num_tokens, 768].
    We add the steering vector (scaled by COEFFICIENT) to every token
    position, nudging the model's internal state toward "formal".
    """
    if not steer_enabled:
        return output

    hidden_state = output[0]  # shape: [1, num_tokens, 768]

    # The steering vector is (768,). Broadcasting adds it to every
    # token position in the sequence automatically.
    # We modify the tensor in-place with add_() so we don't need to
    # reconstruct the block's output container (which can be complex).
    hidden_state.add_(COEFFICIENT * steering_vector)


# Register the hook on transformer block 5.
hook_handle = model.transformer.h[5].register_forward_hook(steering_hook)

# ── Define the prompt ───────────────────────────────────────────────
prompt = "My thoughts on the new policy are"

# Tokenize with an explicit attention mask (avoids the warning about
# pad_token_id not being set).
inputs = tokenizer(prompt, return_tensors="pt")

# ── Generate: UNSTEERED ─────────────────────────────────────────────
# Greedy decoding (do_sample=False) means the model always picks the
# single most likely next token. This removes randomness so the ONLY
# difference between the two outputs is the steering vector.
steer_enabled = False
print("Generating UNSTEERED output (greedy, no randomness) ...")
unsteered_ids = model.generate(
    **inputs,
    max_new_tokens=40,
    do_sample=False,
)
unsteered_text = tokenizer.decode(unsteered_ids[0], skip_special_tokens=True)

# ── Generate: STEERED ───────────────────────────────────────────────
steer_enabled = True
print("Generating STEERED output (greedy + formality vector) ...\n")
steered_ids = model.generate(
    **inputs,
    max_new_tokens=40,
    do_sample=False,
)
steered_text = tokenizer.decode(steered_ids[0], skip_special_tokens=True)

# ── Clean up the hook ───────────────────────────────────────────────
hook_handle.remove()

# ── Print comparison ────────────────────────────────────────────────
print("=" * 60)
print(f"PROMPT:     {prompt}")
print("=" * 60)
print(f"UNSTEERED:  {unsteered_text}")
print("-" * 60)
print(f"STEERED:    {steered_text}")
print("=" * 60)
