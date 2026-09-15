# 02 — Papers summary

<!-- One section per paper. Each: a short summary subsection first, then detailed notes
     including how the paper applies TO THIS PROJECT. Raw PDFs go in 02_papers/. -->

## 1. Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success (Kim et al., 2025)

PDF: not yet saved locally — [arXiv:2502.19645](https://arxiv.org/abs/2502.19645). Project page:
[openvla-oft.github.io](https://openvla-oft.github.io/). This is the paper this entire project
implements; read it before touching the fine-tuning recipe.

### 1.1. Summary

Introduces "Optimized Fine-Tuning" (OFT) for OpenVLA, four changes bundled together:

1. **Parallel decoding + action chunking.** Conditions on `NUM_ACTIONS_CHUNK` empty action
   embeddings with bidirectional attention and predicts that many consecutive actions in **one**
   forward pass, instead of an autoregressive loop that produces one action at a time.
2. **Continuous action representation + L1 regression.** Replaces OpenVLA's 256-bin-per-dimension
   discretization with a separate action head (`L1RegressionActionHead`, or a diffusion
   alternative) trained with plain L1 loss on continuous normalized actions.
3. **Optional FiLM ("OFT+")** for stronger language grounding when a policy must distinguish
   between several instructions.
4. Still LoRA-fine-tunable (`use_lora=True`, `lora_rank=32` defaults — same as this project's
   sibling `openvla` repo's recipe).

Headline numbers: parallel decoding + chunking alone gives 26x faster action generation / 3x
lower latency vs. base OpenVLA; the full recipe reports 25–50x faster inference and a 20+ point
LIBERO success-rate gain. Real-robot results include high-frequency bimanual ALOHA control with
FiLM ("OFT+").

### 1.2. How this applies here

- **`ACTION_DIM=14`** is already a first-class profile in the codebase (`ALOHA_CONSTANTS`) —
  exactly this project's dual-arm dimensionality. See `03_basecode_summary.md`§2 and
  `05a_codemap.md` for what was ported/added to reuse this.
- **Normalization mode matters and is easy to get wrong.** ALOHA's own action space is absolute
  joint angles, so its profile uses raw `BOUNDS` (unclipped). This project's actions are **delta
  end-effector pose** (cartesian + rotation), the same style as the paper's LIBERO experiments,
  which use `BOUNDS_Q99` — that's what `FLEXIV_CONSTANTS` uses (`05a_codemap.md`), not ALOHA's
  default. Don't "helpfully" match ALOHA's setting here; it's for a different action
  representation.
- **The paper trains from the base VLM, not on top of an existing OpenVLA fine-tune.** This
  project's LoRA adapter/merged checkpoints from vanilla OpenVLA (in `~/dev/openvla`) cannot be
  continued into OFT — a new fine-tune from `openvla/openvla-7b` is required (`04_plan.md`).
- **"~1 second-long chunks are a good default"** (the paper's own ALOHA guidance). This project's
  RLDS conversion targets native 30 Hz (decided 2026-09-15, `Ah_for_human.md`§2 Q4 — not the
  stride-5-subsampled ~6 Hz rate originally inherited from vanilla OpenVLA), so
  `NUM_ACTIONS_CHUNK=30` (`05a_codemap.md`) targets ~1 s per query — treat as adjustable, not
  fixed, once real inference latency is measured.

### 1.3. Also read

The sibling project's own paper summary,
[`~/dev/openvla/docs/02_papers_summary.md`](../../openvla/docs/02_papers_summary.md)§1, covers
the base OpenVLA paper this one builds on (architecture, discretization, the fused
DINOv2/SigLIP projector). Read that first if unfamiliar with vanilla OpenVLA.
