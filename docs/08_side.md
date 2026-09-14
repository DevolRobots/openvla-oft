# 08 — Side notes

<!-- Odds-and-ends: non-essential but worth keeping. Gotchas, abandoned approaches, lore,
     surprising findings with their explanations. Often blank at project start. -->

## 1. Platform auto-detection is a substring match on `sys.argv` — watch the whole command line

`prismatic/vla/constants.py`'s `detect_robot_platform()` just checks whether `"flexiv"` (or
`"libero"`/`"aloha"`/`"bridge"`) appears anywhere in the joined command-line string. This is
normally fine since `--dataset_name devol_flexiv_dualarm` will always contain "flexiv", but it
means **any other argument, path, or `--run_id_note` string containing one of those other
platform names first (alphabetically earlier in `sys.argv`, or just present at all) doesn't
override it** — there's no precedence, it's whichever `elif` branch matches first, and only one
can match a single command line since they're checked in a fixed if/elif chain in that order
(`libero` → `aloha` → `bridge` → `flexiv` → default `LIBERO`). Don't put "aloha" or "libero"
anywhere in a Flexiv run's command line (e.g. a `--run_id_note` that says something like
"vs_aloha_baseline") or the wrong constants profile will silently get selected. Worth a quick
`grep` of the full launch command before submitting if ever in doubt, or just print
`ROBOT_PLATFORM` (already logged at import time) and check the log before trusting a run.
