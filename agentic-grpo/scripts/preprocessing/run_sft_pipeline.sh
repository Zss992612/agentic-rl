#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 CONFIG [--dry-run]" >&2
  exit 2
fi

if [[ -n "${AGENTIC_RL_PYTHON:-}" ]]; then
  PYTHON_BIN="${AGENTIC_RL_PYTHON}"
elif [[ -x "${TRAINING_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${TRAINING_DIR}/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${TRAINING_DIR}/src:${PYTHONPATH}"
else
  export PYTHONPATH="${TRAINING_DIR}/src"
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_sft_pipeline.py" "$@"
