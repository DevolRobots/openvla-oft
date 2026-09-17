#!/usr/bin/env python3
"""
submit_finetune.py

Submits the OpenVLA-OFT LoRA fine-tune to the local `gbatch` queue (devolremote or gpu245). GPU
jobs must go through the queue; `openvla_train.sh` on its own would take GPUs outside it.

Ported from the sibling `openvla` project's scripts/submit_finetune.py (docs/08r_gpt_review.md
#2.5 -- no queue-safe submission path existed yet in this repo). Three things it exists to get
right, unchanged from that reference:

  1. **File ownership.** The gflow daemon runs as root, so a job inherits root's UID and every
     checkpoint and log it writes ends up `root:root`. Root `cd`s into the repo, then
     `runuser -u wutingsh` takes over for everything that creates files.
  2. **Environment.** A queued job starts from root's environment, so anything the run needs
     (WANDB_*, HF_HOME, TMPDIR, ...) is silently absent unless re-exported inside the job.
  3. **Secrets.** `gjob show <id>` prints the stored command, so a WANDB key embedded in it
     would be readable by anyone on the box. The key goes into a chmod-600 env file that the
     job sources and then deletes via an EXIT trap; the command only ever names the file.

Deliberately stdlib-only and Python 3.6-compatible, so `python3 scripts/submit_finetune.py`
works with the remote's system interpreter. A submit tool that only runs if you remember to
reach into a venv is a footgun.

Usage:
    python3 scripts/submit_finetune.py --gpus 2 --dataset-name openvla_oft_flexiv_dualarm_stackboxes
    python3 scripts/submit_finetune.py --gpus 2 --dry-run       # print the command, submit nothing

NOTE (2026-09-15): do not submit for real yet -- docs/Ah_for_human.md has open questions (task
scope, action-execution cadence) and docs/08r_gpt_review.md has blocking findings not yet
resolved (no-op-filtered RLDS build unsafe for action chunks; only one of the intended datasets
even has an RLDS build). This script exists so the queue-safe path is ready once those clear.
"""

import argparse
import datetime
import os
import pathlib
import re
import shlex
import subprocess
import sys
import time
import uuid

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
LOG_DIR = pathlib.Path("/cpfs01/wutingsh/openvla_oft_runs/logs")
VENV_PYTHON = pathlib.Path("/cpfs01/wutingsh/openvla-oft/.venv/bin/python")

# Re-exported inside the job. A queued job inherits root's env, so anything omitted here is lost.
QUEUE_EXPORT_VARS = (
    "WANDB_MODE",
    "WANDB_PROJECT",
    "WANDB_ENTITY",
    "HF_HOME",
    "HF_HUB_CACHE",
    "TMPDIR",
    "TEMP",
    "TMP",
    # openvla_train.sh reads all of these; passing them through lets a submission override the
    # recipe without editing the script.
    "GPUS",
    "VLA_PATH",
    "DATA_ROOT_DIR",
    "DATASET_NAME",
    "RUN_ROOT_DIR",
    "USE_L1_REGRESSION",
    "USE_DIFFUSION",
    "USE_FILM",
    "NUM_IMAGES_IN_INPUT",
    "USE_PROPRIO",
    "LORA_RANK",
    "LEARNING_RATE",
    "GRAD_ACCUM",
    "IMAGE_AUG",
    "BATCH_SIZE",
    "SAVE_FREQ",
    "MAX_STEPS",
    "MERGE_LORA_DURING_TRAINING",
    "SAVE_LATEST_CHECKPOINT_ONLY",
    "SHUFFLE_BUFFER_SIZE",
    "RUN_ID_NOTE",
)

# Never embedded in the scheduler-visible command; passed via a chmod-600 env file instead.
SENSITIVE_VARS = ("WANDB_API_KEY",)

# House convention (openpi's CLAUDE.md, matching the sibling openvla project): the key lives in a
# gitignored repo-root `.env` as WANDB_API_KEY_VALUE and is promoted to WANDB_API_KEY for the
# job. A shell-exported WANDB_API_KEY wins over the file.
DOTENV_PATH = REPO_ROOT / ".env"


TRAIN_SCRIPT = REPO_ROOT / "openvla_train.sh"


def train_script_default(var):
    """The `${VAR:-default}` fallback from openvla_train.sh, so a submission can display the
    values it does not itself set. A wrong W&B entity fails at wandb.init, so it is worth
    showing at submit time rather than only in the log afterwards."""
    try:
        text = TRAIN_SCRIPT.read_text()
    except OSError:
        return "?"
    m = re.search(r'^%s="\$\{%s:-([^}]*)\}"' % (var, var), text, re.M)
    return m.group(1) if m else "?"


def load_dotenv_key():
    """Return WANDB_API_KEY from the repo-root .env, or None. Never logs the value."""
    if os.environ.get("WANDB_API_KEY"):
        return None  # shell export wins
    if not DOTENV_PATH.exists():
        return None
    for raw in DOTENV_PATH.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        if key.strip() in ("WANDB_API_KEY_VALUE", "WANDB_API_KEY"):
            return val.strip().strip("\"'")
    return None


def build_env(args):
    """The env the job should see: inherited values, plus anything the CLI overrides."""
    env = {}
    for k in QUEUE_EXPORT_VARS + SENSITIVE_VARS:
        v = os.environ.get(k)
        if v:
            env[k] = v
    env["GPUS"] = str(args.gpus)
    env["DATA_ROOT_DIR"] = args.data_root_dir
    env["DATASET_NAME"] = args.dataset_name
    env["RUN_ROOT_DIR"] = args.run_root_dir
    env["SAVE_FREQ"] = str(args.save_freq)
    env["MERGE_LORA_DURING_TRAINING"] = "True" if args.merge_lora_during_training else "False"
    if args.batch_size is not None:
        env["BATCH_SIZE"] = str(args.batch_size)
    if args.max_steps is not None:
        env["MAX_STEPS"] = str(args.max_steps)
    if not env.get("WANDB_API_KEY"):
        key = load_dotenv_key()
        if key:
            env["WANDB_API_KEY"] = key
    env.setdefault("WANDB_MODE", "offline" if not env.get("WANDB_API_KEY") else "online")
    return env


def active_run_names():
    """Run names of jobs still queued or running, from gqueue.

    Returns None if gqueue cannot be consulted -- callers must treat that as "unknown" and
    refuse to delete, rather than assuming nothing is active.
    """
    try:
        out = subprocess.run(
            ["gqueue", "-u", "all", "-s", "Queued,Running"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=30,
        )
    except Exception:
        return None
    if out.returncode != 0:
        return None
    names = set()
    for line in out.stdout.splitlines():
        for token in line.split():
            # gqueue prints job names as gjob-<id>-<name>
            if token.startswith("gjob-"):
                parts = token.split("-", 2)
                if len(parts) == 3:
                    names.add(parts[2])
    return names


def cleanup_secrets():
    """Remove secret env files left behind by jobs that never started.

    A job deletes its own file via an EXIT trap, so a file only survives if the job never ran --
    typically because it was cancelled while queued. Mode 600 means it was never exposed, but it
    should not linger. Consults gqueue and skips anything still queued or running; if gqueue
    cannot be reached it deletes nothing.
    """
    active = active_run_names()
    if active is None:
        return None, []
    stale_after_s = 600
    now = time.time()
    removed, kept = [], []
    for path in sorted(LOG_DIR.glob("*.secrets.env")):
        run_name = path.name[: -len(".secrets.env")]
        if run_name in active:
            kept.append(run_name)
            continue
        if (LOG_DIR / f"{run_name}.log").exists() and (now - path.stat().st_mtime) < stale_after_s:
            continue  # its job ran very recently; the trap handled it, or is about to
        path.unlink()
        removed.append(path.name)
    return removed, kept


def write_secret_env_file(path, env):
    secrets = {k: env[k] for k in SENSITIVE_VARS if env.get(k)}
    if not secrets:
        return None
    text = "".join("export %s=%s\n" % (k, shlex.quote(v)) for k, v in secrets.items())
    # Owner-only from the first byte on disk -- a plain write_text() + chmod() leaves the file at
    # the umask's default mode for the instant between the two calls, on a machine other users
    # share.
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, text.encode())
    finally:
        os.close(fd)
    return path


def build_gbatch_cmd(
    gpus, time, name, log_path, env, secret_env_path, script="./openvla_train.sh", depends_on=None
):
    # Everything below runs as wutingsh, inside runuser.
    #
    # `set -o pipefail` is load-bearing, not hygiene: the job's last step is
    # `./openvla_train.sh 2>&1 | tee <log>`, and without pipefail a pipeline returns *tee's*
    # status -- a training run that died with exit 1 would report to gflow as `Finished (CD)`.
    # Silent false success is the worst failure mode a submitter can have.
    parts = ["set -o pipefail"]
    parts += ["export %s=%s" % (k, shlex.quote(env[k])) for k in QUEUE_EXPORT_VARS if env.get(k)]
    if secret_env_path is not None:
        parts.append("secret_env=%s" % shlex.quote(str(secret_env_path)))
        parts.append("trap 'rm -f \"$secret_env\"' EXIT")
        parts.append('set -a && . "$secret_env" && set +a')
    # PATH so torchrun resolves to the venv's, not root's.
    parts.append("export PATH=%s:$PATH" % shlex.quote(str(VENV_PYTHON.parent)))
    parts.append("%s 2>&1 | tee %s" % (shlex.quote(script), shlex.quote(str(log_path))))
    user_cmd = " && ".join(parts)

    # Root only cd's; everything that writes files has dropped to wutingsh by then.
    inner = "cd %s && runuser -u wutingsh -- bash -c %s" % (
        shlex.quote(str(REPO_ROOT)),
        shlex.quote(user_cmd),
    )
    cmd = ["gbatch", "--gpus", str(gpus), "--time", time, "--name", name]
    if depends_on:
        # auto-cancel-on-failure is gbatch's own default -- if the job this depends on fails,
        # this one is cancelled rather than starting on top of a broken/incomplete predecessor.
        cmd += ["--depends-on", depends_on]
    cmd += ["bash", "-c", inner]
    return cmd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", type=int, required=True, help="GPUs to request -- CONFIRM THE CURRENT ALLOCATION FIRST")
    ap.add_argument("--time", default="24:00:00", help="gbatch wall-clock limit (default 24h)")
    ap.add_argument("--tag", default="", help="suffix for the run name and log filename")
    ap.add_argument("--data-root-dir", default="/cpfs01/wutingsh/rlds224")
    ap.add_argument("--dataset-name", default="openvla_oft_flexiv_dualarm_stackboxes")
    ap.add_argument("--run-root-dir", default="/cpfs01/wutingsh/openvla_oft_runs")
    ap.add_argument("--save-freq", type=int, default=10000, help="policy: at most every 10000")
    ap.add_argument("--batch-size", type=int, default=None, help="PER DEVICE; default 8 in openvla_train.sh")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument(
        "--merge-lora-during-training",
        action="store_true",
        help="merge + save a full ~15GB checkpoint at every save_freq (default off -- "
        "docs/08r_gpt_review.md#3.1: can consume 300GB+ of shared /cpfs01 for one run). "
        "Merge selected checkpoints offline instead unless you have a specific reason not to.",
    )
    ap.add_argument(
        "--script",
        default="./openvla_train.sh",
        help="repo script to run under gbatch. Exists so diagnostics reuse this file's runuser + "
        "pipefail + secret handling instead of hand-rolling the nested quoting, which is where "
        "the bugs live.",
    )
    ap.add_argument(
        "--depends-on",
        default=None,
        help="gbatch job dependency -- a job ID, or shorthand like '@' (the last job submitted "
        "in this shell). This job is queued now but won't START until that one finishes "
        "successfully; gbatch auto-cancels it if the dependency fails, so a broken predecessor "
        "doesn't waste a GPU slot on a doomed run. Use to chain the three Flexiv tasks serially "
        "on one GPU without polling: submit task 1, capture its job ID from this script's "
        "output, pass it as --depends-on for task 2, etc.",
    )
    ap.add_argument("--dry-run", action="store_true", help="print the submission and exit")
    ap.add_argument(
        "--cleanup-secrets",
        action="store_true",
        help="delete secret env files from jobs that never started, then exit. Consults gqueue "
        "and skips anything still queued or running.",
    )
    args = ap.parse_args()

    if args.cleanup_secrets:
        removed, kept = cleanup_secrets()
        if removed is None:
            print("could not consult gqueue; deleting nothing rather than risk a live job's key")
            return 1
        for name in removed:
            print(f"removed {name}")
        for name in kept:
            print(f"kept    {name} (still queued or running)")
        print(f"{len(removed)} stale secret file(s) removed, {len(kept)} live one(s) kept")
        return 0

    if args.script == "./openvla_train.sh" and args.save_freq < 10000:
        ap.error("--save-freq %d violates the every-10000 checkpoint policy" % args.save_freq)
    # Fail fast at the CLI rather than downstream: a bad value otherwise only surfaces as a
    # confusing failure deep inside the queued job.
    if args.gpus <= 0:
        ap.error("--gpus must be positive, got %d" % args.gpus)
    if args.gpus > 16:
        ap.error("--gpus %d exceeds 16" % args.gpus)
    if args.save_freq <= 0:
        ap.error("--save-freq must be positive, got %d" % args.save_freq)
    if args.batch_size is not None and args.batch_size <= 0:
        ap.error("--batch-size must be positive, got %d" % args.batch_size)
    if args.max_steps is not None and args.max_steps <= 0:
        ap.error("--max-steps must be positive, got %d" % args.max_steps)
    if not re.fullmatch(r"\d{1,3}:\d{2}:\d{2}", args.time):
        ap.error("--time must be HH:MM:SS, got %r" % args.time)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # The uuid suffix, not just the timestamp, is what actually guarantees uniqueness: two
    # identical recipes submitted independently on both GPU machines can land in the same second.
    unique = uuid.uuid4().hex[:6]
    name = "openvla_oft_ft_%s_%s%s" % (stamp, unique, ("_" + args.tag) if args.tag else "")
    log_path = LOG_DIR / ("%s.log" % name)
    env = build_env(args)
    env.setdefault("RUN_ID_NOTE", name)

    secret_path = LOG_DIR / ("%s.secrets.env" % name)
    if not args.dry_run:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        secret_env_path = write_secret_env_file(secret_path, env)
    else:
        secret_env_path = secret_path if any(env.get(k) for k in SENSITIVE_VARS) else None

    cmd = build_gbatch_cmd(
        gpus=args.gpus,
        time=args.time,
        name=name,
        log_path=log_path,
        env=env,
        secret_env_path=secret_env_path,
        script=args.script,
        depends_on=args.depends_on,
    )

    print("name    : %s" % name)
    print("gpus    : %d   time: %s" % (args.gpus, args.time))
    print("script  : %s" % args.script)
    print("dataset : %s (%s)" % (args.dataset_name, args.data_root_dir))
    print("log     : %s" % log_path)
    if args.depends_on:
        print("depends : %s (auto-cancelled if that job fails)" % args.depends_on)
    print("merge_lora_during_training: %s" % env.get("MERGE_LORA_DURING_TRAINING", "False"))
    key_src = (
        "shell WANDB_API_KEY"
        if os.environ.get("WANDB_API_KEY")
        else (".env" if env.get("WANDB_API_KEY") else "none -- running offline")
    )
    print(
        "wandb   : %s (project=%s entity=%s, key from %s)"
        % (
            env.get("WANDB_MODE"),
            env.get("WANDB_PROJECT") or train_script_default("WANDB_PROJECT"),
            env.get("WANDB_ENTITY") or train_script_default("WANDB_ENTITY"),
            key_src,
        )
    )
    # The real command carries the secret only by filename, so it is safe to print verbatim.
    print("\ncommand :\n  %s\n" % " ".join(shlex.quote(c) for c in cmd))

    if args.dry_run:
        print("[dry-run] nothing submitted")
        return 0
    if not VENV_PYTHON.exists():
        print("ERROR: %s not found -- see docs/05b_remote.md section 3" % VENV_PYTHON, file=sys.stderr)
        return 1
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    sys.stdout.write(out.stdout)
    m = re.search(r"Submitted batch job (\d+)", out.stdout)
    if m:
        # Machine-parseable line for a caller chaining --depends-on off this submission.
        print("JOB_ID=%s" % m.group(1))
    return out.returncode


if __name__ == "__main__":
    sys.exit(main())
