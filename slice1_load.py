"""
Slice 1 – Prove we can load a small model and generate text.
Nothing fancy: load GPT-2, feed it a prompt, print the continuation.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Config ───────────────────────────────────────────────────────────
# Change this one variable to swap to a different model later.
MODEL_NAME = "gpt2"

# ── Load model & tokenizer ──────────────────────────────────────────
# The tokenizer turns text into token IDs the model understands.
# The model is the actual neural network that predicts the next token.
# We force everything onto CPU since this machine has no GPU.
print(f"Loading model: {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cpu")
print("Model loaded.\n")

# ── Define a prompt ─────────────────────────────────────────────────
# This is the text we'll ask the model to continue.
prompt = "I went to the store and"

# ── Generate a continuation ─────────────────────────────────────────
# Tokenize the prompt into IDs, then ask the model to keep writing.
# max_new_tokens=40 keeps it short so it runs fast on CPU.
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output_ids = model.generate(input_ids, max_new_tokens=40, do_sample=True)

# Decode the full sequence (prompt + generated tokens) back to text.
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# ── Print results ───────────────────────────────────────────────────
print(f"Prompt:    {prompt}")
print(f"Generated: {generated_text}")
