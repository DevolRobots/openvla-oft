# 05a — Code map

<!-- Directory + code structure of THIS project's new code. May include base code if highly
     relevant. Remote infra lives in 05b_remote.md. -->

## 1. Directory layout

Devol-added / Devol-modified paths on top of upstream `moojink/openvla-oft` (all on branch
`ting/dev`; `main` is an untouched mirror of upstream):

```
openvla-oft/
└── prismatic/vla/
    ├── constants.py                      # + FLEXIV_CONSTANTS platform profile
    └── datasets/rlds/oxe/
        ├── configs.py                    # + devol_flexiv_dualarm(_stackboxes) dataset configs
        ├── transforms.py                 # + devol_flexiv_dualarm_dataset_transform
        └── materialize.py                # + EEF_POS_BIMANUAL action-mask branch
```

Everything else is unmodified upstream `moojink/openvla-oft` — see
[`03_basecode_summary.md`](03_basecode_summary.md) for what upstream provides.

## 2. What was ported, and from where

All four changes were **ported verbatim** (same values, same reasoning) from the sibling
`openvla` project's fork of vanilla `openvla/openvla`, where the equivalent registration already
exists and is battle-tested (it trained the sibling project's real checkpoint). Diffed against
`~/dev/openvla`'s `git diff upstream/main -- prismatic/vla/datasets/rlds/oxe/{configs,transforms,materialize}.py`
to confirm exactly what to port; nothing here was written from scratch.

- **`configs.py`**: added `StateEncoding.POS_QUAT_BIMANUAL = 5` and
  `ActionEncoding.EEF_POS_BIMANUAL = 5` (this fork's enums topped out at 4 in both, same as
  upstream `openvla`, so no collision). Added `devol_flexiv_dualarm` and
  `devol_flexiv_dualarm_stackboxes` dataset configs: single camera (`primary` only — OpenVLA(-OFT)
  with `num_images_in_input=1` never reads `secondary`/`wrist`), 16-D state
  (`EEF_state_left`/`EEF_state_right`/`gripper_state`), `POS_QUAT_BIMANUAL` state encoding,
  `EEF_POS_BIMANUAL` action encoding.
- **`transforms.py`**: added `devol_flexiv_dualarm_dataset_transform` — action passes through
  untouched (the converter in the sibling repo already emits the final 14-D layout); state is
  split from a flat 16-D vector `[grip_L, grip_R, xyz_L(3), xyz_R(3), quat_L(4), quat_R(4)]` into
  the three keys `configs.py` declares. Registered for both dataset names.
- **`materialize.py`**: added `ActionEncoding.EEF_POS_BIMANUAL` to the allowed-encodings check,
  and an `elif` branch setting `absolute_action_mask`/`action_normalization_mask` =
  `([False]*6 + [True]) * 2` — two independent per-arm blocks, each ending in its own absolute
  gripper dimension. Verified this file resolves the mask by `traj["action"].shape[-1]`
  (14, per-timestep), **not** by the chunked/flattened action length — so no `NUM_ACTIONS_CHUNK`
  multiplication needed here; chunking windows the trajectory elsewhere.
- **`constants.py`**: added `FLEXIV_CONSTANTS` (`ACTION_DIM=14`, `PROPRIO_DIM=16`,
  `NUM_ACTIONS_CHUNK` originally `8` as a stride-5-at-6Hz placeholder, now **`30`** — decided
  2026-09-15 for native-30Hz cadence, `Ah_for_human.md`§2 Q4 — `ACTION_PROPRIO_NORMALIZATION_TYPE=
  BOUNDS_Q99` — **not** ALOHA's raw `BOUNDS`, since this project's actions are delta
  end-effector pose like LIBERO's, not ALOHA's absolute joint angles) and a `"flexiv" in cmd_args`
  branch in `detect_robot_platform()`.
  Verified in isolation (loading the file directly, bypassing the full `prismatic` package import
  chain which needs the venv from step 2 of `04_plan.md`): passing
  `--dataset_name devol_flexiv_dualarm` on the command line correctly selects
  `ROBOT_PLATFORM == "FLEXIV"` and all four constants.

**Not ported:** `mixtures.py` (`OXE_NAMED_MIXTURES`) — the sibling `openvla` fork never added an
entry there either; it's only needed for weighted multi-dataset mixtures, not a single
`--dataset_name` fine-tune.

**2026-09-15 addition:** two more dataset config + transform registrations, `configs.py`/
`transforms.py` only — same schema and `devol_flexiv_dualarm_dataset_transform`, no new code,
just distinct dataset names (`08r_gpt_review.md`#2.2): `openvla_oft_flexiv_leftarm_stackboxes`
(left-arm-only box-stacking) and `openvla_oft_flexiv_dualarm_dinrail` (DIN-rail wire-connector).
No `materialize.py` change needed — its action-mask branch keys on
`ActionEncoding.EEF_POS_BIMANUAL`, not on dataset name.

**Naming decided (`Ah_for_human.md`§2 Q5):** all three stack/DIN-rail registrations use the
`openvla_oft_flexiv_<armconfig>_<task>` scheme (also renamed the existing
`devol_flexiv_dualarm_stackboxes` → `openvla_oft_flexiv_dualarm_stackboxes`, since no RLDS build
existed for it either), matching the sibling `openpi` project's naming for these same three tasks
as closely as sensible — see `configs.py`'s comment for the full rationale.

**Done (2026-09-15):** the sibling `~/dev/openvla` repo's builder for box-stacking was renamed to
match (`DevolFlexivDualarmStackboxes` → `OpenvlaOftFlexivDualarmStackboxes`), and two new builder
packages were created there from scratch for the other two tasks (same pattern, ~15-line
subclasses). **All three RLDS builds now exist** on `/cpfs01/wutingsh/rlds224` at native 30 Hz
with no-op filtering off (`07_HISTORY.md`). These sibling-repo changes are **uncommitted** in both
the local `~/dev/openvla` checkout and the remote `/cpfs01/wutingsh/openvla` one — see
`Ah_for_human.md`§3 Q7.

Also 2026-09-15: `scripts/submit_finetune.py` + `openvla_train.sh` (repo root, not under
`prismatic/`) — the queue-safe `gbatch` launcher, ported from the sibling project
(`08r_gpt_review.md`#2.5). See `05b_remote.md`§4 and `09_commands.md`§3.

## 3. What's still to write (not yet done)

- A `serve.py` analog for this fork (`04_plan.md` step 4) — nothing exists here yet. Port
  `deployment/flexiv_dualarm/schema.py` from the sibling repo largely as-is (it's model-agnostic:
  builds the wire-contract schema from dataset stats + declared fields), and write a new
  `serve.py` that loads an OFT checkpoint (base model + LoRA adapter + separate action-head
  state dict — see `merge_lora_weights_and_save.py` for the loading pattern) instead of a plain
  `AutoModelForVision2Seq` discrete-token model.
