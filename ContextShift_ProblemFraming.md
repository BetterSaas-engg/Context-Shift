# ContextShift — The Two Real Problems (and who already solved them)

A strategic note. The single most useful reframe from the build sessions: what we
hit are NOT novel "AI steering" problems. They are two old, well-studied problems
in disguise — and both have mature solutions in other fields we can borrow. Naming
them properly is the unlock, and it sharpens what the actual product is.

---

## What we observed (the raw symptoms)

Steering a small model toward "formal" gave us two recurring failures:

1. **Drift** — the steer changed the *topic*, not just the *style* (e.g. a casual
   sentence got pulled toward random political/news content).
2. **The seesaw** — turn the dial low and the effect is clean but too weak; turn
   it up and the effect is strong but drift/breakdown returns. No single setting
   was simultaneously strong, clean, and reliable across prompts.

These are not bugs in our code. They are the genuine shape of the problem.

---

## Problem 1: DRIFT is a "confound / source-separation" problem

Our steering vector is meant to carry ONE thing (style) but is tangled with
another (topic). Isolating one factor from a mixture is a classic problem that
shows up across many fields:

- **Debiasing word embeddings** — the closest cousin, same math as ours.
  Researchers found the "gender direction" in word vectors and *projected it out*
  of words that shouldn't carry gender. Identical operation to "find the content
  direction and subtract it out of the style vector."
- **Genetics (GWAS)** — ancestry confounds gene-trait studies; the standard fix is
  to compute the ancestry direction and regress it out before testing.
- **Signal processing (the cocktail-party problem)** — separate two mixed voices
  from one recording. An entire field: "source separation."

**Shared solution: PROJECTION.** Explicitly identify the unwanted direction and
remove its component, rather than hoping it averages away. (This is the untried
"path B" lever for our drift. Three unrelated fields converging on it is why it's
worth a real attempt.)

---

## Problem 2: THE SEESAW is a "gain vs. saturation" problem

Turn the knob up, the wanted effect grows — but so does a side-effect, and past a
point the system breaks (our cliff). The field that owns this is **control theory**.

- **Control theory's answer:** stop using a fixed knob (open-loop). Instead,
  *measure the output and adjust the push on the fly* — closed-loop feedback (PID).
- **Audio engineering:** compressors/limiters get loudness without clipping.
- **RF amplifiers:** predistortion to get power without distortion.

**Shared solution: CLOSE THE LOOP.** Replace the fixed coefficient with feedback
that measures effect and self-adjusts.

---

## The analogy that contains BOTH problems: pharmacology

A drug's benefit rises with dose, but so does toxicity, and past a point toxicity
wins. The gap where it helps without harming is the **therapeutic window** —
literally our sweet spot. Medicine handles it with exactly our two levers:

1. **Selectivity** — design a molecule that hits the target receptor and not the
   off-targets. = a PURER vector (our last-token-pooling instinct, taken further
   via projection). Attacks Problem 1.
2. **Titration** — start low, measure response, adjust dose to hit effect without
   crossing into toxicity. = FEEDBACK CONTROL. Attacks Problem 2.

A whole rigorous discipline already faces our precise dilemma and resolved it with
the same two moves.

---

## Why this matters: the frontier is already importing these

The newer steering research (PID controllers for steering; conditional/gated
steering) is literally control theory and feedback bolted onto this problem.
People are independently arriving at "don't use a fixed coefficient, close the
loop." The cross-domain hunch isn't just valid — it's where the field is heading.

---

## What this does to the product thesis

The "hard part worth paying for" now has a concrete name. It is NOT "add a vector"
(a free `pip install` does that). It is **importing the mature solutions**:

- **Projection-based purification** of the steering direction (selectivity) — so
  the dial changes style without dragging content.
- **Closed-loop auto-titration** of strength (feedback) — so the dial finds and
  holds its own sweet spot per model/concept/prompt, instead of a human
  hand-tuning a fragile fixed knob.

The raw open-source library does NONE of this. That gap is the company:
a clean, self-regulating tone dial vs. a fragile manual one.

---

## Honest caveat

These solutions are proven *in their home fields*. Whether they fully tame the
seesaw on a tiny model (GPT-2-medium) is genuinely uncertain — small models may
lack the headroom regardless of how cleverly we control the gain. The right test
bed is a modern small instruction-tuned model (1–3B), where importing feedback
control is (my estimate) the single most promising direction.

---

## Concrete next experiments (in priority order)

1. **Projection** — build a "content/topic" direction, remove its component from
   the formality vector, re-run the 8-prompt test. Directly targets drift.
2. **Move to a modern small model** (1–3B instruct) — re-run the whole recipe;
   likely widens the therapeutic window on its own.
3. **Closed-loop strength** — prototype a feedback rule that measures effect and
   auto-adjusts the coefficient, instead of a fixed number.
