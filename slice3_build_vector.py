"""
Slice 3 – Build a "formality" steering vector and save it to disk.
We contrast formal vs casual sentences to find the direction in the
model's internal space that represents formality.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Config ───────────────────────────────────────────────────────────
MODEL_NAME = "gpt2"

# ── Load model & tokenizer ──────────────────────────────────────────
print(f"Loading model: {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, output_hidden_states=True
).to("cpu")
print("Model loaded.\n")

# ── Define contrastive sentence lists ───────────────────────────────
# The sentences cover varied topics so that topic-specific patterns
# cancel out when we average. The ONLY consistent difference between
# the two lists is formality — that's what survives the averaging.

formal_sentences = [
    "I am writing to formally request your assistance with this matter.",
    "Please find the enclosed report for your review and consideration.",
    "We would like to schedule a meeting at your earliest convenience.",
    "I respectfully submit the following proposal for your evaluation.",
    "Kindly confirm receipt of the attached documentation at your leisure.",
    "The committee has reviewed the application and reached a decision.",
    "We appreciate your prompt attention to this important issue.",
    "I would be grateful if you could provide clarification on this point.",
    "It is my pleasure to inform you that your request has been approved.",
    "We wish to express our sincere gratitude for your continued support.",
]

casual_sentences = [
    "hey can you help me out with this thing?",
    "yo check out this report lol",
    "wanna grab a meeting sometime this week?",
    "so i got this idea, tell me what you think",
    "got the docs, thanks a bunch!",
    "ok so they looked at the app and made a call",
    "hey just wanted to flag this real quick",
    "can you explain what you meant by that?",
    "nice, looks like they said yes!",
    "thanks so much dude, really appreciate it",
]

# ── Helper: get a single vector for a sentence ─────────────────────
def get_sentence_vector(sentence):
    """
    Run one sentence through the model and return a single 768-length
    vector representing that sentence at layer 6.
    """
    input_ids = tokenizer.encode(sentence, return_tensors="pt")

    # No training here, just reading — torch.no_grad() saves memory.
    with torch.no_grad():
        outputs = model(input_ids)

    # Grab the layer-6 hidden state. Shape: [1, num_tokens, 768]
    layer_6 = outputs.hidden_states[6]

    # Mean-pool across the token dimension (dim=1).
    # WHY: each sentence has a different number of tokens, so the
    # hidden state tensors have different lengths. By averaging across
    # all token positions we collapse each sentence down to one fixed
    # 768-length vector, making sentences with different token counts
    # directly comparable.
    sentence_vector = layer_6.mean(dim=1).squeeze()  # shape: (768,)

    return sentence_vector


# ── Compute average vectors for each style ──────────────────────────
# WHY average many sentences? If we only used one formal and one casual
# sentence, the difference would be contaminated by topic, word choice,
# sentence length, etc. By averaging ~10 sentences on varied topics,
# all those random differences cancel out. The only thing that's
# consistently different between the two groups — formality — survives.

print("Computing formal sentence vectors ...")
formal_vectors = [get_sentence_vector(s) for s in formal_sentences]
formal_avg = torch.stack(formal_vectors).mean(dim=0)  # shape: (768,)

print("Computing casual sentence vectors ...")
casual_vectors = [get_sentence_vector(s) for s in casual_sentences]
casual_avg = torch.stack(casual_vectors).mean(dim=0)  # shape: (768,)

# ── Build the steering vector ───────────────────────────────────────
# The difference points from "casual" toward "formal" in the model's
# internal representation space. This is our steering vector.
steering_vector = formal_avg - casual_avg

# ── Save to disk ────────────────────────────────────────────────────
torch.save(steering_vector, "formal_vector.pt")
print("\nSteering vector saved to formal_vector.pt")

# ── Print summary ───────────────────────────────────────────────────
print(f"\nSteering vector shape: {tuple(steering_vector.shape)}")
print(f"Steering vector magnitude (L2 norm): {torch.norm(steering_vector).item():.4f}")
print("  (A non-zero magnitude confirms this is a real direction, not noise.)")
