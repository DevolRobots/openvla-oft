# 01 — Overview

<!-- Human-oriented top-level overview of openvla-oft. Prose, not a file-table
     (the file index lives in 00_README.md). Not redundant with the root CLAUDE.md. -->

## 1. What this project is

A fork of [`moojink/openvla-oft`](https://github.com/moojink/openvla-oft) ("Optimized
Fine-Tuning" for OpenVLA — [arXiv:2502.19645](https://arxiv.org/abs/2502.19645)), being adapted
to Devol's Flexiv dual-arm rollout. It exists to answer one question: **can action chunking make
the policy fast enough to be usable on the real robot?**

The sibling project `~/dev/openvla` fine-tuned vanilla OpenVLA-7B on the Flexiv dual-arm
"Insert the Ethernet connector" task and got it moving the arms (after fixing two serving bugs,
see `~/dev/openvla/docs/09s_serving_inference.md`§9–§10), but control frequency is capped around
3.6 Hz — one full 7B forward pass buys exactly one timestep, because vanilla OpenVLA has no
action chunking at all. That's too slow for smooth manipulation.

OpenVLA-OFT replaces OpenVLA's autoregressive per-timestep discrete decoding with **parallel
decoding + action chunking** (one forward pass predicts `NUM_ACTIONS_CHUNK` consecutive actions)
plus a **continuous L1-regression action head** (no more 256-bin discretization). The paper
reports 25–50x faster inference. This project's job is to fine-tune that recipe on the same
Flexiv dual-arm dataset and re-serve it, to see if the speedup is real for this task.

## 2. Scope & non-goals

**In scope:**
- Porting the Flexiv dual-arm dataset registration into this fork (done — see `05a_codemap.md`).
- A new LoRA fine-tune from `openvla/openvla-7b`, using OFT's chunking + L1-regression recipe.
- New serving code that speaks the same `DevolInference` wire contract the sibling project's
  `deployment/flexiv_dualarm/serve.py` does, so the robot-side client needs no changes.
- Re-running the same rollout to see if the higher effective control rate fixes the "too slow"
  problem.

**Out of scope / explicitly not this project's job:**
- Re-deriving the dataset itself (conversion, no-op filtering, stride choice) — that's done in
  `~/dev/openvla` and this project reuses its output (`/cpfs01/wutingsh/rlds224`) as-is.
- FiLM ("OFT+") — the task has exactly one fixed instruction string
  ("Insert the Ethernet connector"), so there's nothing for stronger language grounding to buy;
  default to `--use_film False` unless a reason emerges to revisit.
- Anything about the box-stacking dataset — no OpenVLA checkpoint (vanilla or OFT) has been
  trained on it yet; out of scope until the dual-arm Ethernet task is settled.
- This is a **scope/priority experiment, not a committed direction.** Per the kickoff
  (`h0_inputs.md`), if OFT-style chunking turns out not to be adoptable or not worth the cost,
  the plan is to drop it — the paper already has other models covered.
