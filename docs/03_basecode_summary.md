# 03 — Base code summary

<!-- The pre-existing / upstream code this project evolves from. Read before 05a_codemap.md,
     which covers only what THIS project adds. -->

## 1. Upstream / starting point

`moojink/openvla-oft` on GitHub — itself a fork of `openvla/openvla` (which the sibling
`~/dev/openvla` project also forks). Remotes here: `origin` = `DevolRobots/openvla-oft` (this
fork), `upstream` = `moojink/openvla-oft` (the public original), `main` a clean mirror, this
project's changes on `ting/dev` — same convention as `openvla` and `openpi`.

Same base VLM (`openvla/openvla-7b`), same RLDS/TFDS data pipeline, same `prismatic/` package
layout as vanilla OpenVLA. What's different: the action decoder (parallel + chunked instead of
autoregressive + discrete), the training/serving scripts, and per-robot-platform constants.

## 2. Key components to know

- **`prismatic/vla/constants.py`** — auto-detects a "robot platform" from `sys.argv` (substring
  match on "libero"/"aloha"/"bridge"/now "flexiv") and sets `NUM_ACTIONS_CHUNK`, `ACTION_DIM`,
  `PROPRIO_DIM`, `ACTION_PROPRIO_NORMALIZATION_TYPE` globally from that. This is the single most
  important file to understand before running anything — get the wrong platform detected and
  every downstream shape assumption is wrong. See `05a_codemap.md` for the `FLEXIV_CONSTANTS`
  block added here.
- **`vla-scripts/finetune.py`** — LoRA fine-tuning entry point. Same shape as the sibling
  project's `finetune.py` (`torchrun ... --vla_path openvla/openvla-7b --dataset_name ... --lora_rank 32`),
  plus OFT-specific flags: `--use_l1_regression`, `--use_diffusion`, `--use_film`,
  `--num_images_in_input`, `--use_proprio`. Also handles `NUM_ACTIONS_CHUNK`-windowed action
  chunks and the action head's own optimizer/checkpointing.
- **`prismatic/models/action_heads.py`** — `L1RegressionActionHead` and `DiffusionActionHead`.
  This project uses L1 regression (`--use_l1_regression True --use_diffusion False`), matching
  the paper's primary recipe and its faster/simpler behavior vs. diffusion.
- **`vla-scripts/deploy.py`** — the upstream serving script. **Not directly reusable here**: it
  speaks a FastAPI/`json-numpy` wire protocol, not the msgpack/websocket schema-driven contract
  the `DevolInference` robot client expects (that contract is implemented in the sibling
  project's `deployment/flexiv_dualarm/serve.py`+`schema.py`). A new serve.py analog needs to
  load an OFT checkpoint (continuous head + chunked decode) the way `deploy.py` does, but publish
  through the existing schema/self-check machinery instead. See `04_plan.md` step 4.
- **`vla-scripts/merge_lora_weights_and_save.py`** — merges a LoRA adapter into the base model
  offline, analogous to the sibling project's `scripts/merge_adapter.py`. Also handles the
  device-mismatch caveat: merge on the same device type used for training, or re-merge on the
  target device before testing (see the ALOHA.md note this was ported from).
- **`ALOHA.md`** (upstream doc, still in this repo) — the closest existing precedent to this
  project's own bimanual setup. Read it before writing the fine-tune command: it documents the
  exact registration steps (`configs.py`/`transforms.py`/`mixtures.py`), the chunk-size
  reasoning, and the full training/serving command shapes this project's own commands are
  modeled on.
- **`prismatic/vla/datasets/rlds/oxe/{configs,transforms,materialize}.py`** — the OXE dataset
  registry. Same file structure and same registration pattern as vanilla `openvla`'s fork. See
  `05a_codemap.md` for exactly what was ported into this fork's copies of these files.

## 3. Dependency note (do not skip)

`transformers` here is **not** the stock PyPI package — `pyproject.toml` pins
`transformers @ git+https://github.com/moojink/transformers-openvla-oft.git`, a patched fork
needed for the bidirectional attention parallel decoding uses. `torch==2.2.0`,
`torchvision==0.17.0`, `peft==0.11.1`, `timm==0.9.10` match the sibling `openvla` project's pins
exactly, but **this needs its own venv** — it cannot share `~/dev/openvla/.venv`. See
`05b_remote.md`§3.
