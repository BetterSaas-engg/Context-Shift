# ContextShift POC — Results

**Date:** 2026-06-27
**Model:** GPT-2 small (124M), CPU only
**What this documents:** what the 5-slice tracer bullet actually proved, and —
just as importantly — what it did NOT prove. Written to keep us honest before
anyone builds anything else on top of it.

---

## TL;DR

The activation-steering **mechanism works end to end** on a real model, on a
laptop, with no GPU and no retraining. We can read a model's internal state,
build a direction from contrasting examples, inject it during generation, and
tune its strength with a dial — including watching the output break exactly when
theory says it should.

**We have NOT yet demonstrated clean "formality" control specifically.** On a
model this small, the steering produced shorter/stiffer/more-repetitive text
rather than text that clearly reads as *more formal*. That distinction matters
and is the next thing to earn.

---

## What we built (5 slices, each runs and prints something real)

| Slice | File | Proved |
|---|---|---|
| 1 | `slice1_load.py` | A small open-weight model loads and generates text on CPU. |
| 2 | `slice2_read_state.py` | We can reach into the model and read its internal hidden state. Shape `(1, 7, 768)` → 768 "features" (sliders) per token. |
| 3 | `slice3_build_vector.py` | We can build a steering vector by averaging formal vs casual sentence states and subtracting. Saved to `formal_vector.pt`. |
| 4 | `slice4_steer.py` | Injecting that vector during generation changes the output. Same prompt + same greedy decoding → different text, purely from steering. |
| 5 | `slice5_dial.py` | The strength is a real, tunable dial. Swept coefficients and found both a usable band and the breakdown point. |

---

## Key measured findings

- **The direction is real and strong.** The steering vector's magnitude (L2
  norm) came out at **~94.93**. A near-zero number would have meant formal and
  casual text look the same inside the model. It doesn't — they sit in clearly
  different places. (This proves a *difference exists*, not that the vector is
  clean — see limitations.)

- **Injection point matters and was matched exactly.** The vector was built from
  `hidden_states[6]` (the output of transformer block index 5), and injected at
  the same place via a forward hook on `model.transformer.h[5]`. Build-location
  and inject-location must agree.

- **The dial has a narrow usable band on this model:**
  - **0** → baseline (normal GPT-2).
  - **0.05** → no visible change yet.
  - **0.1** → first visible nudge ("The new policy" → "The policy"), still coherent.
  - **0.15–0.3** → text tightens and restructures, still grammatical.
  - **0.5** → **cliff**: collapses into repetition ("policy to policy to the policy...").
  - **1.0 and above** → fully broken ("in, in, in, in...").

- **The cliff is real and showed up on cue.** This is the inverted-U from the
  literature: quality rises to a sweet spot, then falls off sharply as strength
  increases. Seeing it is a success — it confirms the dial genuinely controls
  the model.

- **Coefficient scale was the surprise.** Because the vector magnitude is ~95,
  even a coefficient of 1.0 was already far past the cliff. The entire usable
  range lives in fractions below 1.0. Finding the right scale was the real
  fiddly work — and is exactly the part that makes this a product rather than a
  one-liner.

---

## What this PROVES

1. The core ContextShift mechanism is sound and reproducible on real hardware.
2. It needs no retraining, no GPU, no weight changes — toggling the hook on/off
   is the entire control surface.
3. The dial behaves predictably, including a findable sweet spot and a findable
   cliff.

## What this does NOT prove (do not overclaim)

1. **That we steered "formality."** The steered text got more clipped and
   repetitive, not obviously more *formal* in register. On GPT-2 small the
   vector likely captured "shorter/stiffer" more than true formality. The
   machinery is proven; crisp *concept* control is not yet.
2. **That the effect is clean or reliable.** One prompt, one concept, one tiny
   model. No measurement of how often it works or how it generalizes.
3. **Anything about bigger or better models.** Untested.
4. **Anything about production reliability, composition of multiple concepts, or
   auto-tuning.** All explicitly out of scope for this POC.

---

## Honest headline

> "The steering machinery works end to end. Demonstrating crisp, legible concept
> control (e.g. formality you can actually read) is the next thing to earn."

Not: "We built a formality dial." We built the dial; we haven't yet shown it
turns *formality* cleanly.

---

## Highest-value next step (parked, one decision)

Swap `MODEL_NAME` to a slightly stronger small model and re-run the same five
scripts unchanged. If the steered output reads as *crisply more formal* (not
just shorter), that confirms the concept-cleanliness gap was a model-size limit,
and gives a demo that actually reads as formal. One-line change; we designed for
it (decision D4 in the spec).

Everything else (auto-tuning the sweet spot, a second concept, an eval harness,
any packaging) stays parked until that re-run produces legible formality.
