# Human evaluation — rating interface and rater guide

This directory holds the interface used for the paper's human study: a local
Node server that walks a rater through one checklist question at a time, showing
the generated video, the task instruction and optional source context. 

```
node build_data.mjs     # specs/ + checklists/ + outputs/ -> data.js
node server.mjs --open  # serve data/ and open the survey in a browser
```

or double-click `start.command` (macOS) / `start.bat` (Windows).

---

# LabInstruct rater guide

## 0. The one rule

Judge only by **what actually happens or appears on screen**. The wording of the
question, the task description shown on the page, "this is how the action should
be done", "the final result looks plausible" — none of these is evidence. Do not
answer *yes* on the strength of any of them.

## 1. Choosing between the three options

| Option | Meaning | When to pick it |
|---|---|---|
| **Yes** | The video shows clear enough evidence that it **was done** | You can see the action / state / requirement hold |
| **No** | The video shows clear enough evidence that it **was not done, or done wrong** | You can see it missing, violated, or contradicted |
| **Unjudgeable** | The decisive moment is occluded, badly blurred, the object leaves the frame or disappears, or there are obvious artifacts — **you cannot judge reliably** | If you cannot see it, pick this. **Do not guess.** It is not "probably a pass" and not "probably a fail" |

## 2. Order and condition questions (the important ones)

These ask about the relationship between two actions — for example, *"Does the
scalpel being passed from the right hand to the left hand happen before the left
hand starts scraping the root?"* They implicitly assume both actions occur.
Decide in three steps:

1. **You saw both actions** → judge *yes* or *no* by the order on screen. (The
   scraping happens first, then the pass → **No**.)
2. **One of the actions clearly did not happen** (e.g. the left hand never
   scrapes the root) → answer **No**. The premise fails, so the item counts as
   not passed.
3. **You cannot tell whether an action happened** (blocked by a hand, off
   screen) → answer **Unjudgeable**. **Do not treat "I could not see it" as
   "it did not happen."**
