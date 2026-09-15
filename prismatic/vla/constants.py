"""
Important constants for VLA training and evaluation.

Attempts to automatically identify the correct constants to set based on the Python command used to launch
training or evaluation. If it is unclear, defaults to using the LIBERO simulation benchmark constants.
"""
import sys
from enum import Enum

# Llama 2 token constants
IGNORE_INDEX = -100
ACTION_TOKEN_BEGIN_IDX = 31743
STOP_INDEX = 2  # '</s>'


# Defines supported normalization schemes for action and proprioceptive state.
class NormalizationType(str, Enum):
    # fmt: off
    NORMAL = "normal"               # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"               # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"       # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]
    # fmt: on


# Define constants for each robot platform
LIBERO_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

ALOHA_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 25,
    "ACTION_DIM": 14,
    "PROPRIO_DIM": 14,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
}

BRIDGE_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 5,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 7,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

# Devol Flexiv dual-arm (custom; see docs/04s_openvla_oft_feasibility.md in the sibling openvla
# repo). Action is per-arm [delta_xyz(3), delta_rotvec(3), gripper(1)] x 2 = 14D -- same
# dimensionality as ALOHA_CONSTANTS, but this is a DELTA end-effector-pose action (like LIBERO's
# EEF_POS), not ALOHA's absolute joint angles -- hence BOUNDS_Q99, not ALOHA's raw BOUNDS.
# PROPRIO_DIM is 16, not 14: state is [grip_L, grip_R, xyz_L(3), xyz_R(3), quat_L(4), quat_R(4)]
# (quaternion, not rotvec, per arm). NUM_ACTIONS_CHUNK=30: decided 2026-09-15
# (docs/Ah_for_human.md#2 Q4, docs/08r_gpt_review.md#2.4) to convert at NATIVE 30 Hz rather than
# inheriting vanilla OpenVLA's stride-5 subsampling -- that stride was only chosen to cut
# per-timestep query cost for a single-step policy, which doesn't apply once one query returns a
# whole chunk. 30 also matches the sibling `openpi` project's `action_horizon=30` on this same
# Flexiv data (native rate), so results are comparable across policy families on the same tasks.
# ~1s of open-loop motion per query, same ballpark as the old stride-5 value but at native (not
# 5x-sparser) temporal resolution. Re-derive once real OFT inference latency is measured.
FLEXIV_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 30,
    "ACTION_DIM": 14,
    "PROPRIO_DIM": 16,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}


# Function to detect robot platform from command line arguments
def detect_robot_platform():
    cmd_args = " ".join(sys.argv).lower()

    if "libero" in cmd_args:
        return "LIBERO"
    elif "aloha" in cmd_args:
        return "ALOHA"
    elif "bridge" in cmd_args:
        return "BRIDGE"
    elif "flexiv" in cmd_args:
        return "FLEXIV"
    else:
        # Default to LIBERO if unclear
        return "LIBERO"


# Determine which robot platform to use
ROBOT_PLATFORM = detect_robot_platform()

# Set the appropriate constants based on the detected platform
if ROBOT_PLATFORM == "LIBERO":
    constants = LIBERO_CONSTANTS
elif ROBOT_PLATFORM == "ALOHA":
    constants = ALOHA_CONSTANTS
elif ROBOT_PLATFORM == "BRIDGE":
    constants = BRIDGE_CONSTANTS
elif ROBOT_PLATFORM == "FLEXIV":
    constants = FLEXIV_CONSTANTS

# Assign constants to global variables
NUM_ACTIONS_CHUNK = constants["NUM_ACTIONS_CHUNK"]
ACTION_DIM = constants["ACTION_DIM"]
PROPRIO_DIM = constants["PROPRIO_DIM"]
ACTION_PROPRIO_NORMALIZATION_TYPE = constants["ACTION_PROPRIO_NORMALIZATION_TYPE"]

# Print which robot platform constants are being used (for debugging)
print(f"Using {ROBOT_PLATFORM} constants:")
print(f"  NUM_ACTIONS_CHUNK = {NUM_ACTIONS_CHUNK}")
print(f"  ACTION_DIM = {ACTION_DIM}")
print(f"  PROPRIO_DIM = {PROPRIO_DIM}")
print(f"  ACTION_PROPRIO_NORMALIZATION_TYPE = {ACTION_PROPRIO_NORMALIZATION_TYPE}")
print("If needed, manually set the correct constants in `prismatic/vla/constants.py`!")
