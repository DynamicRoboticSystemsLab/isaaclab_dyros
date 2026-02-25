#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLAY_SCRIPT="${SCRIPT_DIR}/scripts/rsl_rl/play.py"
DEFAULT_TASK="Tocabi-Flat-Play"
DEFAULT_NUM_ENVS="${DYROS_NUM_ENVS:-1}"

find_isaaclab_launcher() {
    if [[ -n "${ISAACLAB_SH:-}" && -f "${ISAACLAB_SH}" ]]; then
        echo "${ISAACLAB_SH}"
        return
    fi
    if [[ -n "${ISAACLAB_PATH:-}" && -f "${ISAACLAB_PATH}/isaaclab.sh" ]]; then
        echo "${ISAACLAB_PATH}/isaaclab.sh"
        return
    fi
    if [[ -n "${ISAACLAB_ROOT:-}" && -f "${ISAACLAB_ROOT}/isaaclab.sh" ]]; then
        echo "${ISAACLAB_ROOT}/isaaclab.sh"
        return
    fi
    if [[ -f "${SCRIPT_DIR}/isaaclab.sh" ]]; then
        echo "${SCRIPT_DIR}/isaaclab.sh"
        return
    fi
}

TASK="${DYROS_TASK:-$DEFAULT_TASK}"
if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
    TASK="$1"
    shift
fi

HAS_TASK_ARG=0
HAS_NUM_ENVS_ARG=0
for arg in "$@"; do
    if [[ "$arg" == "--task" || "$arg" == --task=* ]]; then
        HAS_TASK_ARG=1
    fi
    if [[ "$arg" == "--num_envs" || "$arg" == --num_envs=* ]]; then
        HAS_NUM_ENVS_ARG=1
    fi
done

ARGS=("$@")
if [[ ${HAS_TASK_ARG} -eq 0 ]]; then
    ARGS=(--task "$TASK" "${ARGS[@]}")
fi
if [[ ${HAS_NUM_ENVS_ARG} -eq 0 ]]; then
    ARGS=(--num_envs "$DEFAULT_NUM_ENVS" "${ARGS[@]}")
fi

ISAACLAB_LAUNCHER="$(find_isaaclab_launcher || true)"
if [[ -n "${ISAACLAB_LAUNCHER}" ]]; then
    exec bash "${ISAACLAB_LAUNCHER}" -p "${PLAY_SCRIPT}" "${ARGS[@]}"
fi

if command -v python3 >/dev/null 2>&1; then
    exec python3 "${PLAY_SCRIPT}" "${ARGS[@]}"
fi

exec python "${PLAY_SCRIPT}" "${ARGS[@]}"
