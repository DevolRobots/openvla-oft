# 04 — Plan

<!-- High-level plan, synthesised from 02 (papers) + 03 (basecode). More detailed than 01.
     Exotic/peripheral ideas go in 08_side, not here. -->

## 1. Goal & approach

Fine-tune OpenVLA-OFT's action-chunking recipe on the same Flexiv dual-arm "Insert the Ethernet
connector" dataset the sibling `openvla` project already used, then serve it through the same
`DevolInference` wire contract, to see whether chunked/parallel decoding raises the achievable
control frequency enough to make the rollout usable. Full reasoning:
`~/dev/openvla/docs/04s_openvla_oft_feasibility.md`.

## 2. Steps / milestones

1. ~~**Fork + port the dataset registration.**~~ **Done (2026-09-14).** Forked
   `moojink/openvla-oft` to `DevolRobots/openvla-oft`; ported the `devol_flexiv_dualarm` /
   `devol_flexiv_dualarm_stackboxes` dataset configs, transform, and action-mask logic from the
   sibling `openvla` fork; added a `FLEXIV_CONSTANTS` platform profile. Detail:
   [`05a_codemap.md`](05a_codemap.md).
2. **Build a dedicated venv.** Needs `transformers @ git+https://github.com/moojink/transformers-openvla-oft.git`
   (not stock `transformers==4.40.1`) — cannot reuse `~/dev/openvla/.venv`. See
   [`05b_remote.md`](05b_remote.md)§3 for the exact gotchas already hit once in the sibling
   project's shared-venv setup (uv's per-machine interpreter cache, root-owned leftover files).
3. **Submit a LoRA fine-tune from `openvla/openvla-7b`.** Cannot continue the sibling project's
   existing checkpoint — OFT's continuous action head + chunked attention pattern isn't present
   in a vanilla-OpenVLA checkpoint. Recipe shape: same `torchrun ... finetune.py` invocation as
   the sibling project, plus `--use_l1_regression True --use_diffusion False --use_film False
   --num_images_in_input 1 --lora_rank 32`. **Open decision, not yet made:** `--use_proprio`
   (this project's schema has never fed proprio to the model; enabling it here would be a new
   capability, not parity) — see `hA_for_agent.md`/`Ah_for_human.md` if this needs a call before
   submitting. Go through `gbatch`/the GPU queue exactly as the sibling project does (same
   cluster, same policy: save at most every 10000 steps, re-confirm GPU allocation before every
   run). Order-of-magnitude cost: expect similar to the sibling project's ~24h wall-clock run —
   still a 7B LoRA fine-tune at the same batch/model scale.
4. **Write a new serving path.** OFT's own `vla-scripts/deploy.py` speaks a different wire
   protocol (FastAPI/`json-numpy`) than what the `DevolInference` robot client expects. Needs a
   `serve.py` analog that: loads an OFT checkpoint (continuous L1-regression head + chunked
   decode), runs one forward pass to get a chunk of `NUM_ACTIONS_CHUNK` actions, and publishes
   `declared.action_chunk_size` accordingly through the same schema/self-check machinery the
   sibling project's `deployment/flexiv_dualarm/schema.py` already implements (reuse, don't
   reinvent — port and adapt it). The `DevolInference` client has already been verified to accept
   `action_chunk_size > 1` with no client-side change (see the sibling project's
   `Ah_for_human.md`§2 Q3) — this step is server-side only.
5. **Re-run the pre-flight ladder and the rollout.** Same ladder the sibling project's
   `09s_serving_inference.md`§6 documents (schema self-check → client's own offline test →
   `--fake` → `probe.py` → real motion) — don't skip straight to the robot just because this is a
   different model. Measure real achieved control frequency this time (client-side timestamps),
   which neither of the sibling project's two rollout attempts captured
   (`~/dev/openvla/docs/10_results.md` R6/R7).

## 3. Decision point

This is an experiment with an explicit exit condition (`01_overview.md`§2): if step 3 or 4 turns
out substantially harder than expected, or the resulting control frequency isn't actually usable
once measured, the plan is to drop this and rely on the paper's other models — not to keep
pushing indefinitely. Log that decision in `06_current.md`/`07_HISTORY.md` if it's made.
