# ContextShift — Plain-English Summary

A one-page record of what we set out to do, what we built, where it landed, and who might actually pay for it. Written to be readable by anyone, not just engineers.

---

## The idea in one breath

An AI model holds a list of numbers inside it while it reads and writes — its "state of mind." Directions in those numbers line up with concepts (formal, casual, cheerful, cautious). If you find the direction for a concept, you can gently push the model along it and steer how it writes — **without retraining it and without changing the model itself.** It's a dial you bolt on, and you can turn it off instantly.

---

## Where we started

A product idea ("make any model behave like it was built for your world") and a fair question: is the core trick real, and can a non-expert actually make it work on a normal laptop? We did not assume — we set out to prove it.

## What we did

We built it in five small steps, each one tested before moving on:

1. Loaded a small open model on a plain laptop (no special hardware).  
2. Read the model's internal "state" — confirmed we could reach inside it.  
3. Built a "formal" direction by comparing formal vs casual example sentences.  
4. Injected that direction while the model wrote — and the output changed.  
5. Added a strength dial and tuned it to find the usable range.

Along the way we hit the expected wall (push too hard and the writing breaks), and we learned by elimination what actually controls the effect.

## Where we ended

It works. With the right settings, turning the dial moves a real model's writing from casual to formal in a way you can read with your own eyes — same model, same prompt, one number changed. The big lesson: **where** you apply the steer (which internal layer) mattered more than model size or anything else.

**Honest limit:** we proved it works on one example. We have NOT yet proven it works *reliably* across many different prompts. That reliability test is the next step — and notably, it's also exactly the hard part a real product would sell.

---

## What this is good for (real applications)

The sweet spot is **controlling how an AI writes** — tone, style, persona — for teams running their own open models. Strongest uses:

1. **"Sound less like AI."** Steer away from the obvious AI writing tics. Content and marketing teams actively hate AI-sounding copy right now.  
2. **Brand voice that holds.** Keep tone and formality consistent across a whole conversation, instead of the model drifting off-brand after a few messages.  
3. **One model, many client personas.** A SaaS company serving 50 clients can make the model feel "theirs" per client, without building 50 custom models.  
4. **Reading-level dial.** Same answer, dialled from "explain to a 10-year-old" to "explain to an expert" — without rewriting prompts.  
5. **Length / cost control.** Steer toward concise output, which also trims the compute bill.  
6. **In-character consistency.** Keep a game character, tutor, or companion in persona over a long session.

We deliberately stayed away from high-stakes uses (medical, self-driving) where a wrong nudge could hurt someone — that's also where the technique is weakest.

---

## Who would be a good customer

The technique needs "open" models that a team runs themselves, so the buyer is someone already self-hosting AI and fighting to control how it sounds:

- **SaaS / AI product companies** that embed an open model and resell it to many customers who each want a distinct voice. (Best fit — pain is sharp, budget exists.)  
- **Marketing / content platforms** that generate copy at scale and need it on-brand and not obviously AI-written.  
- **Customer-support / chatbot teams** that need a consistent tone and register across every reply.  
- **Localization teams** juggling formal vs casual register across languages.  
- **Game / education / companion-app studios** that need characters or tutors to stay in voice.

The common thread: they run their own model, they care a lot about *how it sounds*, and today their only tools are fragile prompts or expensive retraining. The pitch is "a reliable tone dial that needs neither."

---

## The honest one-liner

"We proved the dial is real and a non-expert can build it. The money is in making it work *reliably*, automatically, for people who don't want to think about any of this — that reliability is the product, not the trick."  
