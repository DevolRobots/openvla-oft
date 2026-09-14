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
  `NUM_ACTIONS_CHUNK=8`, `ACTION_PROPRIO_NORMALIZATION_TYPE=BOUNDS_Q99` — **not** ALOHA's raw
  `BOUNDS`, since this project's actions are delta end-effector pose like LIBERO's, not ALOHA's
  absolute joint angles) and a `"flexiv" in cmd_args` branch in `detect_robot_platform()`.
  Verified in isolation (loading the file directly, bypassing the full `prismatic` package import
  chain which needs the venv from step 2 of `04_plan.md`): passing
  `--dataset_name devol_flexiv_dualarm` on the command line correctly selects
  `ROBOT_PLATFORM == "FLEXIV"` and all four constants.

**Not ported:** `mixtures.py` (`OXE_NAMED_MIXTURES`) — the sibling `openvla` fork never added an
entry there either; it's only needed for weighted multi-dataset mixtures, not a single
`--dataset_name` fine-tune.

## 3. What's still to write (not yet done)

- A `serve.py` analog for this fork (`04_plan.md` step 4) — nothing exists here yet. Port
  `deployment/flexiv_dualarm/schema.py` from the sibling repo largely as-is (it's model-agnostic:
  builds the wire-contract schema from dataset stats + declared fields), and write a new
  `serve.py` that loads an OFT checkpoint (base model + LoRA adapter + separate action-head
  state dict — see `merge_lora_weights_and_save.py` for the loading pattern) instead of a plain
  `AutoModelForVision2Seq` discrete-token model.
