# ContextShift POC — "The Formality Dial"

## 1. Goal (one sentence)

Prove that we can take one small open-weight model, build a single "formal" steering
vector from a handful of example sentences, and turn a dial that visibly moves a
generated sentence from casual to formal — without retraining anything.

That's it. If we can turn that one dial, the core idea is real and everything else
is engineering on top.

## 2. What we are proving (and what we are NOT)

**In scope — the only thing this POC must show:**
- Read the model's internal state while it processes text.
- Build a "formal" direction by averaging + subtracting (the steps AK already understands).
- Inject that direction during generation.
- Sweep a strength dial (0 → up) and watch the same answer get more formal.

**Explicitly NOT in scope (do not build these now):**
- No factual-accuracy / "stick to the facts" steering. Out of scope, known weak axis.
- No web UI, no API, no packaging, no auto-tuning. CLI script only.
- No big model, no GPU, no cloud. Laptop CPU only.
- No multiple concepts. One concept: formality.
- No "make it production reliable." This is a tracer bullet, not a product.

If a step tempts us to add something not on the in-scope list, we stop and note it
instead of building it.

## 3. Environment (AK's actual setup)

- Windows 11, PowerShell, VS Code, Claude Code in terminal.
- Python (latest), `uv` for package management.
- CPU only. No GPU. This is fine for the model we chose.

## 4. Model choice (a decision, with the reasoning)

**Decision: GPT-2 small (124M parameters).**

Why this one for the POC:
- It is open-weight, so we can reach inside its layers (the whole point).
- It is tiny (~500 MB). Loads and runs on a normal CPU.
- It is the canonical model used in the original activation-steering work, so there
  is lots of reference code if we get stuck.
- Generating a short sentence takes a few seconds on CPU — slow enough to notice,
  fast enough to iterate.

**Reversibility note:** nothing in the code should hard-assume GPT-2. The model name
lives in one variable at the top. Swapping to a small Qwen/Mistral later = change one
line. We are not marrying GPT-2; we are dating it for the POC.

**On AK's speed worry:** a few seconds per generation is the expected cost. If it ever
feels too slow, the lever is "generate fewer words" (a max-length setting), not a
bigger machine. We will keep generations short on purpose.

## 5. The build, as a tracer bullet (5 thin slices)

Each slice ends with something we can *run and see*. We do not move to the next slice
until the current one prints something real. This is the "get it working end-to-end
early" principle — we are not building all the parts and assembling at the end.

### Slice 1 — Make the pipe work
- Load GPT-2 small.
- Give it a prompt, print a normal generated continuation.
- **Done when:** we see the model say *anything* back. No steering yet.

### Slice 2 — Read the internals
- Run one sentence through the model.
- Grab the internal state (the "sliders") at one middle layer.
- Print its shape so we can see it's a real list of numbers.
- **Done when:** we can capture and print the hidden state for any sentence.

### Slice 3 — Build the steering vector
- Write ~8–10 short formal sentences and ~8–10 casual ones (in the script).
- Capture each one's middle-layer state.
- Average the formal states, average the casual states, subtract.
- Save the result (the "formal" steering vector) to a file.
- **Done when:** a `formal_vector` file exists on disk.

### Slice 4 — Inject the vector
- During generation, add the saved vector to that same middle layer at every step.
- Use a fixed strength for now (e.g. start with a small number).
- Generate the same prompt as Slice 1.
- **Done when:** the steered output reads differently (hopefully more formal) than Slice 1.

### Slice 5 — The dial
- Loop over strengths: 0, 2, 4, 6, 8.
- Print the same prompt's output at each strength, stacked, so we can read the
  progression top to bottom.
- **Done when:** we can see casual → formal as the number climbs — and we can see
  where it falls off the cliff into nonsense.

## 6. The one thing that will go "wrong" (expected, not a bug)

At some strength the output will break — repeating words or going garbled. **That is
the cliff from step 6 of the explanation, and seeing it is a success, not a failure.**
It proves the dial is real and shows us where the usable range ends. We are looking
*for* the cliff, not avoiding it.

If the formal effect is weak before the cliff, the knobs to try (in order) are:
1. A different middle layer.
2. More / cleaner example sentences.
3. A different prompt to steer.

## 7. Definition of done for the whole POC

A single PowerShell command runs the script and prints the same sentence at five
strengths, and AK can read down the list and watch it get more formal until it breaks.

That is the proof. Nothing more is required from this POC.

## 8. Decisions log (so we don't re-litigate)

- **D1:** Model = GPT-2 small. Reason: open-weight, tiny, CPU-friendly, canonical.
- **D2:** One concept only = formality. Reason: strongest, simplest axis to demo.
- **D3:** CLI script, no UI. Reason: tracer bullet; UI is not what we're proving.
- **D4:** Model name in one variable. Reason: reversibility — easy model swap later.
- **D5:** Short generations on purpose. Reason: CPU speed + faster iteration.
- **D6:** Hitting the "cliff" is a goal, not a bug. Reason: it proves the dial works.

## 9. What comes after (parked — not now)

Only once the dial works, the natural next questions are: does it hold on a better
small model, can we make the strength auto-pick its sweet spot, can we add a second
concept. All parked until Slice 5 prints something real.
