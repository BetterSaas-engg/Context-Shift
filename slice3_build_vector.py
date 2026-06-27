"""
Slice 3 – Build a "formality" steering vector and save it to disk.
We contrast formal vs casual sentences to find the direction in the
model's internal space that represents formality.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Config ───────────────────────────────────────────────────────────
MODEL_NAME = "gpt2-medium"

# ── Load model & tokenizer ──────────────────────────────────────────
print(f"Loading model: {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, output_hidden_states=True
).to("cpu")
print("Model loaded.\n")

# ── Define matched contrast pairs ───────────────────────────────────
# Each formal_sentences[i] and casual_sentences[i] express the SAME
# meaning and differ ONLY in formality — same topic, same intent,
# opposite register.
#
# WHY matched pairs sharpen the vector: when we average the formal
# vectors and the casual vectors separately, any signal related to
# topic, sentence length, or specific content appears in BOTH averages
# equally (because each pair shares the same meaning). So when we
# subtract casual_avg from formal_avg, those shared signals cancel out
# perfectly — leaving a cleaner formality direction than loosely-matched
# lists where topic/content differences would leak into the vector.

formal_sentences = [
    "I would be grateful if you could assist me with this matter.",
    "Please find the requested report attached for your review.",
    "I am writing to formally request a meeting at your earliest convenience.",
    "Thank you for your prompt response to my inquiry.",
    "I regret to inform you that we are unable to proceed.",
    "Kindly confirm whether the proposed terms are acceptable.",
    "It was a pleasure to make your acquaintance yesterday.",
    "I would appreciate your feedback at your earliest convenience.",
    "We apologize for any inconvenience this may have caused.",
    "Please do not hesitate to contact me should you require assistance.",
    "I look forward to our continued collaboration.",
    "Could you please clarify the requirements for this task?",
]

casual_sentences = [
    "can u help me out with this?",
    "here's that report u wanted lol",
    "wanna grab a meeting sometime soon?",
    "thanks for getting back to me so fast!",
    "yeah sorry we can't do it",
    "lemme know if those terms work for u",
    "was great meeting u yesterday!",
    "hit me back with thoughts whenever",
    "sorry for the hassle",
    "just ping me if u need anything",
    "excited to keep working together!",
    "what exactly do u need for this?",
]

# ── Helper: get a single vector for a sentence ─────────────────────
def get_sentence_vector(sentence):
    """
    Run one sentence through the model and return a single 1024-length
    vector representing that sentence at layer 12.
    """
    input_ids = tokenizer.encode(sentence, return_tensors="pt")

    # No training here, just reading — torch.no_grad() saves memory.
    with torch.no_grad():
        outputs = model(input_ids)

    # Grab the layer-18 hidden state. Shape: [1, num_tokens, 1024]
    # gpt2-medium has 24 transformer blocks; layer 18 is in the upper third.
    layer_6 = outputs.hidden_states[18]

    # Take the last token's hidden state instead of mean-pooling.
    # WHY: mean-pooling averages in topic-specific words from across
    # the sentence, leaking content into the vector. The last-token
    # state tends to summarize the sentence's overall style with less
    # topic contamination, which should reduce content drift when
    # steering. It also gives us a fixed 1024-length vector regardless
    # of sentence length.
    sentence_vector = layer_6[0, -1, :]  # shape: (1024,)

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
