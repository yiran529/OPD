# AGENTS.md

## Project Overview
This project studies how to improve the self-correction ability of linear attention models under rollout errors.

The current research idea is described in `AI/ideas/1.md`.

The implementation is expected to:
- use a flash-linear-attention implementation,
- load pretrained weights from `m-a-p`,
- and finetune on FineWeb-Edu 100B.

---

## Source of Truth
Read files in this order:

1. `AGENTS.md`
2. `AI/ideas/*.md`
3. `AI/code_map.md`
4. `AI/memory.md`

### In details

`AI/` contains lightweight project context for the agent.

- `AI/ideas/`: what we are trying to build.
- `AI/code_map.md`: where things are in the codebase. Update it when the structure changes or when it is no longer accurate.
- `AI/memory.md`: durable decisions and constraints.

Read them when the task needs method context, codebase navigation, or prior implementation decisions.
Update `AI/code_map.md` after structural changes.
Write to `AI/memory.md` only when a decision or lesson is likely to matter later.
Do not use `AI/memory.md` as a scratchpad or raw experiment log.
NOTE THAT NEW CONTENTS MUST WRITE AT THE VERY END IN `AI/memory.md`

---

## Engineering Style
Prefer code that is:
- simple,
- small,
- explicit,
- correct.

Avoid unnecessary abstraction.
Avoid framework-like overdesign.
Do not build for hypothetical future use cases.

This codebase is for a specific research goal, not for a general-purpose training framework.

### Code Readability (Complex Logic)

If a piece of code contains complex logic:

- Break it into clearly separated logical blocks.
- Add short comments for each block explaining its purpose.
- Use comments to visually separate blocks (e.g., divider-style comments).

Prefer:

```python
# ---- compute corrupted prefix ----
...

# ---- rollout with model ----
...

# ---- compute alignment loss ----
```

---

## Core Preferences
- Keep the training loop explicit.
- Prefer direct PyTorch-style code.
- Reuse external libraries where they already solve the problem well.
- Keep experimental logic local and easy to change.
- Prefer config-driven switches for actual experiments.

---

## Fail Fast
Fail early when assumptions are violated.

Do not add silent fallbacks, defensive auto-recovery, or broad compatibility layers unless explicitly needed.

Prefer:
- clear assertions,
- only check real assumptions,
- prefer `assert` for simple invariants,
- avoid verbose `raise`,
- explicit shape checks when necessary,
- explicit error messages when necessary,
- narrow supported paths.

If something is unsupported, it should fail clearly rather than degrade gracefully.

---

## Scope Control
Do not add support for cases we do not intend to support.

Do not add generic compatibility code just in case it may be useful later.

Do not preserve old code paths unless they are still part of the intended workflow.

When choosing between:
- a simple implementation for the intended use case,
- or a more general implementation with extra complexity,

prefer the simple implementation.

---

## Implementation Guidance
The main code should make it easy to inspect:
- clean-prefix behavior,
- corrupted-prefix rollout,
- continuation rollout,
- main training loss,
- state alignment loss.

Keep these parts readable and easy to debug.

Prefer small local modules over deep abstraction.

Do not modify third-party library source directly unless absolutely necessary.
If integration is needed, prefer a thin local wrapper.

---

## Validation
At minimum, changes should preserve a small end-to-end runnable path.

Before considering an implementation done, check:
- pretrained weights load correctly,
- rollout runs correctly,
- losses are finite,
- backward works,
- logging is sufficient to debug failures.

---

## Working Rules
When implementing a change:

1. read `AI/ideas/x.md`, where x is the largest number in AI/ideas
2. inspect the relevant code,
3. make the smallest correct change,m inimize the scope of changes.
    - make the smallest change that correctly implements the idea,
    - avoid modifying unrelated code,
    - do not refactor or introduce abstractions unless necessary.
    - If a change is large, invasive, or structurally significant: propose a plan first, get user confirmation before implementing.
4. keep the implementation explicit,
5. record durable decisions in `AI/memory.md` and `AI/implementation.md` if needed.
