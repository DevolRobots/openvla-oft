# 04 — Plan

<!-- High-level plan, synthesised from 02 (papers) + 03 (basecode). More detailed than 01.
     Exotic/peripheral ideas go in 08_side, not here. -->

## 1. Goal & approach

Fine-tune OpenVLA-OFT's action-chunking recipe on the Flexiv dual-arm task family (originally
"Insert the Ethernet connector"; widened 2026-09-14/15 to the three tasks in `h0_inputs.md`§2 —
see `06_current.md`§1) the sibling `openvla` project already has data for, then serve it through
the same `DevolInference` wire contract, to see whether chunked/parallel decoding raises the
achievable control frequency enough to make the rollout usable. Full reasoning:
`~/dev/openvla/docs/04s_openvla_oft_feasibility.md`.

## 2. Steps / milestones

1. ~~**Fork + port the dataset registration.**~~ **Done (2026-09-14).** Forked
   `moojink/openvla-oft` to `DevolRobots/openvla-oft`; ported the `devol_flexiv_dualarm` /
   `devol_flexiv_dualarm_stackboxes` dataset configs, transform, and action-mask logic from the
   sibling `openvla` fork; added a `FLEXIV_CONSTANTS` platform profile. Detail:
   [`05a_codemap.md`](05a_codemap.md).
2. ~~**Build a dedicated venv.**~~ **Done (2026-09-15).** Needed
   `transformers @ git+https://github.com/moojink/transformers-openvla-oft.git` (not stock
   `transformers==4.40.1`) — cannot reuse `~/dev/openvla/.venv`. Built and verified (real import)
   on both `devolremote` and `gpu245`. See [`05b_remote.md`](05b_remote.md)§3 for the gotchas hit
   once in the sibling project's shared-venv setup, and `08r_gpt_review.md`#2.1 for a second
   import failure (protobuf/`tensorflow-metadata`) found and fixed after the venv first built
   clean but before a real RLDS import was tried.
3. **Submit a LoRA fine-tune from `openvla/openvla-7b`, per task.** **Not yet — see
   [`08r_gpt_review.md`](08r_gpt_review.md), verdict "do not submit yet".** Cannot continue the
   sibling project's existing checkpoint — OFT's continuous action head + chunked attention
   pattern isn't present in a vanilla-OpenVLA checkpoint. Recipe: `scripts/submit_finetune.py` +
   `openvla_train.sh` (ported 2026-09-15, ready but not yet used for a real submission) run
   `--use_l1_regression True --use_diffusion False --use_film False --num_images_in_input 1
   --lora_rank 32 --use_proprio False` (decided, `Ah_for_human.md`§1 Q1) through `gbatch`, save at
   most every 10000 steps with `merge_lora_during_training=False` by default
   (`08r_gpt_review.md`#3.1 — merging every save can consume 300GB+ of shared `/cpfs01`).
   **All decisions made 2026-09-15 (`Ah_for_human.md`§2 Q3–Q5):** all three tasks in
   `h0_inputs.md`§2, as three separate fine-tunes submitted **serially, one GPU each** (not
   concurrently); cadence is **native 30 Hz** (`NUM_ACTIONS_CHUNK=30` in `constants.py`, not the
   inherited stride-5-at-6Hz); dataset names are `openvla_oft_flexiv_dualarm_stackboxes` /
   `openvla_oft_flexiv_leftarm_stackboxes` / `openvla_oft_flexiv_dualarm_dinrail`.
   **Still blocked on:** re-converting each task's RLDS build at native rate with the inherited
   no-op filter disabled/redesigned so interior deletions don't break fixed-length action chunks
   (`08r_gpt_review.md`§2.3) — sibling-repo (`~/dev/openvla`) data-conversion work, not this
   repo's code. Order-of-magnitude cost once unblocked, per task: expect similar to the sibling
   project's \~24h wall-clock run — still a 7B LoRA fine-tune at the same batch/model scale, times
   three since they run serially.
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
