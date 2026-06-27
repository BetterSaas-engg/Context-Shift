# ContextShift POC — Results

**Date:** 2026-06-27
**Model:** GPT-2 medium (355M), CPU only
**Status:** POC landed — a single tunable dial produces a legible casual→formal
shift while staying coherent.

This file records what the POC actually proved, the working recipe, and the
honest limits — so nobody (including us) overclaims later.

---

## TL;DR

We can steer a real model's register with one dial, no retraining, on a laptop.
The decisive factor turned out to be **which layer** we steer at: shallow/middle
layers gave mushy results; the **upper-third layer (18 of 24)** gave a clean,
readable formality shift with a wide, stable usable band.

The casual→formal change is visible to the naked eye at coefficient 0.1–0.3.

---

## What we built (5 slices, each runs and prints something real)

| Slice | File | Proves |
|---|---|---|
| 1 | `slice1_load.py` | A small open-weight model loads and generates on CPU. |
| 2 | `slice2_read_state.py` | We can read the model's internal hidden state. |
| 3 | `slice3_build_vector.py` | We can build a steering vector from contrasting examples and save it. |
| 4 | `slice4_steer.py` | Injecting the vector during generation changes the output (clean, deterministic). |
| 5 | `slice5_dial.py` | Strength is a tunable dial with a findable sweet spot and a findable cliff. |

---

## The working recipe (what actually produces the formal shift)

- **Model:** `gpt2-medium` (355M), CPU.
- **Layer:** build the vector from `hidden_states[18]`; inject at `model.transformer.h[17]`
  (these are the same point — output of block 17 — and they MUST match).
- **Contrast set:** 12 **matched pairs** — each formal sentence paired with a
  casual sentence of the same meaning, differing only in register.
- **Vector:** mean-pool each sentence's layer-18 state to one 1024-length vector,
  average the formal set, average the casual set, subtract. Magnitude ≈ **208**.
- **Coefficient sweet spot:** **0.1 – 0.3** (clean and readable). Stable up to ~0.6.
- **Cliff:** begins around **0.8**, full collapse by 1.0+.

### The headline example (greedy decoding, same prompt)

Prompt: *"My thoughts on the new policy are"*

- **Coefficient 0 (baseline):** "...mixed. I think it's a good idea, but I don't
  think it's the right way to go about it." — casual, hedgy, blog-ish.
- **Coefficient 0.1:** "...as follows: 1. The new policy is a good first step. It
  is a step in the right direction, but it is not enough." — restructures into a
  formal, numbered, memo-style register and stays coherent.

That "as follows: / numbered points" shift is a legible formality fingerprint.

---

## The path that got us here (what we ruled in/out, with evidence)

This is the useful part — we found the working recipe by elimination, not luck.

1. **GPT-2 small (124M), middle layer 6** -> mechanism worked, but effect was just
   "shorter/stiffer," and the cliff came almost immediately (broke below
   coefficient 1.0). Too small to steer gracefully.
2. **GPT-2 medium, middle layer 12, loose contrast lists** -> stable across the
   sweep (no early cliff), but effect read as "measured/analytical," not formal.
3. **GPT-2 medium, middle layer 12, matched contrast pairs** -> vector got
   stronger (130 -> 203), but pushing harder caused **topic drift** (wandered to
   random news/political content), never formal register. So contrast quality and
   raw strength were NOT the bottleneck.
4. **GPT-2 medium, layer 18, matched pairs** -> clean, legible formality with a
   wide sweet spot. **Layer depth was the deciding lever.**

Conclusion: for register/style on this model, *where* you steer matters more than
model size, contrast-set polish, or coefficient strength. Style lives in the
upper layers.

---

## What this PROVES

1. The ContextShift mechanism works end to end on a real model, no GPU, no
   retraining, no weight changes — toggling the hook on/off is the whole control surface.
2. With the right layer, the steered output shows a **legible** formality shift,
   not just "different" output.
3. The dial is predictable: findable sweet spot, findable cliff, wide stable band.

## What this does NOT prove (do not overclaim)

1. **Reliability.** This is one prompt. We have not shown the same clean shift
   across many varied prompts. "It can happen cleanly" != "it reliably happens."
2. **Which formality.** The effect is "structured/officious" register (numbered,
   memo-like), not the "polite-correspondence" flavor ("I would be grateful /
   kindly") that our contrast pairs emphasized. It found *a* formal register, not
   precisely the one in our examples.
3. **Generalization to other models or concepts.** Untested here.
4. **Production-grade anything** — no auto-tuning, no multi-concept composition,
   no eval harness. All still out of scope.

---

## Honest headline

> "With the right layer, the steering produces a clean, readable casual->formal
> shift on a real model on a laptop. Proving it does so *reliably across prompts*
> is the next thing to earn."

---

## Highest-value next steps (parked, in order)

1. **Reliability test.** Run the layer-18 / coeff-0.2 recipe across 8–10 different
   prompts. Does the formal shift hold every time? This is the single most
   important next experiment — it's the line between "demo" and "works."
2. **Auto-pick the coefficient.** Right now we eyeball the sweet spot. A product
   needs to find it automatically per model/concept.
3. **Second concept.** Try an easier axis (e.g. positive/negative sentiment) to
   confirm the layer-depth lesson generalizes beyond formality.

Everything else stays parked until step 1 (reliability) is answered.
