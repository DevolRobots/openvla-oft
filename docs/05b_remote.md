# 05b — Remote infrastructure

<!-- Remote infra map: hosts, paths, data locations, scheduler. Split from 05a because infra
     churns independently of code. -->

## 1. Hosts & access

Same cluster as the sibling `openvla`/`openpi` projects — **`/cpfs01` is one shared network
filesystem, identical on both physical boxes.** See `~/.claude/CLAUDE.md`'s "Remote server stuff"
section (loaded globally in every session on this machine) for the generic `gbatch`/`gqueue`
mechanics, root/`runuser` file-ownership gotcha, and the two-machine (`devolremote` /
`gpu245`=`devolaction002`) setup — not repeated here.

Project-specific:
- **This repo has no remote checkout yet.** The sibling `openvla` project lives at
  `/cpfs01/wutingsh/openvla` on both machines; this one will need its own clone, e.g.
  `/cpfs01/wutingsh/openvla-oft`, before anything can run on `gbatch`.
- **Checkpoint output convention** (not yet created): mirror the sibling project's
  `/cpfs01/wutingsh/openvla_runs/` pattern, e.g. `/cpfs01/wutingsh/openvla_oft_runs/` — keep it
  separate from the vanilla-OpenVLA runs directory so the two don't collide or get confused.
- **Checkpoint policy**: same as every project on this cluster — save at most every 10000 steps
  (disk conservation). `/cpfs01` has repeatedly run into the 95–99% full range during the
  sibling project's work — check `df -h /cpfs01` before submitting anything that writes a lot.

## 2. Key paths

| Path | Contents |
|---|---|
| `/cpfs01/wutingsh/rlds224/devol_flexiv_dualarm` | The dataset to fine-tune on — **already built** by the sibling project, reused as-is. 224x224 images, stride-5 subsampled (~6 Hz effective step rate). Do not rebuild; this is the same data the sibling project's vanilla-OpenVLA checkpoint trained on. |
| `/cpfs01/data/devol/lerobot_vla_jepa/batch_20260902_145405_flexiv_action_superset_3cam` | Source LeRobot batch (for `meta/modality.json`, `meta/stats_gr00t.json` — needed by the schema/serving side, not by training itself). |
| `/cpfs01/wutingsh/openvla/` | The sibling vanilla-OpenVLA repo checkout — read its `deployment/flexiv_dualarm/` for the schema/serving contract to port (`04_plan.md` step 4). Its `.venv` **cannot** be reused here (§3). |
| `/cpfs01/wutingsh/openvla_runs/` | Sibling project's checkpoints (`step_50000` merged model etc.) — for reference/comparison only, not inputs to this project's training. |

## 3. Building the venv (needs its own — cannot share `~/dev/openvla/.venv`)

`pyproject.toml` here pins `transformers @ git+https://github.com/moojink/transformers-openvla-oft.git`
(a patched fork for bidirectional attention / parallel decoding), not stock
`transformers==4.40.1` the sibling project uses. `torch==2.2.0`/`torchvision==0.17.0`/
`peft==0.11.1`/`timm==0.9.10` match exactly, but the `transformers` difference alone means a
fresh venv, e.g.:

```bash
export UV_CACHE_DIR=/cpfs01/wutingsh/.cache/uv   # /cpfs01/uv is root-owned and NOT writable
cd /cpfs01/wutingsh/openvla-oft   # after cloning there
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e .
# flash_attn is commented out in pyproject.toml on purpose -- install AFTER the editable install
# per SETUP.md upstream, if using it.
```

> [!WARNING] Shared-venv gotchas already hit once in the sibling project — read before rebuilding
> anything
> 1. **`uv`'s own downloaded Python interpreters are per-machine**, even though the venv
>    directory itself is shared via `/cpfs01`. The *first* `uv run`/`uv sync` from whichever
>    machine didn't build the venv will notice the Python symlink doesn't resolve locally and try
>    to auto-repair — this is expected, not corruption, but it can look alarming.
> 2. **Never `rm -rf` a shared venv to "see what error comes up."** That happened once in the
>    sibling project and deleted a real, in-use environment. If a rebuild is ever needed, `mv
>    .venv .venv.stale_$(date +%s)` (rename, not delete) — a live process holding the old files
>    open won't break.
> 3. **Verify a rebuilt venv with a real import** (`uv run python -c "import torch;
>    print(torch.cuda.is_available())"`), not just a clean `uv sync` exit code — a partial install
>    can still exit 0.
> 4. **Watch for root-owned leftover files** inside an otherwise `wutingsh`-owned tree —
>    `find <path> -not -user wutingsh` finds them; they can silently block a rebuild with
>    "Permission denied" even though the rest of the tree looks fine.

## 4. W&B credentials

The sibling project's `submit_finetune.py` convention: a gitignored repo-root `.env` with
`WANDB_API_KEY_VALUE`, promoted to `WANDB_API_KEY` for the job, never embedded in the
scheduler-visible command (goes into a `chmod 600` env file the job sources and deletes via an
`EXIT` trap). This repo has no submitter script yet (`04_plan.md` step 3) — write one following
the sibling project's `scripts/submit_finetune.py` as the reference (same file-ownership,
environment, and secrets concerns apply identically), and set up this repo's own `.env` (copy the
key from the sibling project's if reusing the same W&B entity/project).
