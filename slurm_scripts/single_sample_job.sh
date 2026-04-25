#!/bin/bash
set -euo pipefail

sample=$1
split_id=$2
sample_group=$3
model=$4
label_key=$5

export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1

ENV_NAME=cryovit_env
LOCK_DIR=/lscratch/sanketg/tmp
LOCK_FILE="$LOCK_DIR/cryovit_env_setup.lock"
LOCK_TIMEOUT=1800   # 30 min ceiling; env setup realistically takes a few min

mkdir -p "$LOCK_DIR"

# Make `mamba activate` available in this non-interactive shell.
eval "$(mamba shell hook --shell bash)"

# ---- Critical section: only one job per node does env setup at a time ----
#
# The subshell holds an exclusive lock on FD 200 (bound to $LOCK_FILE).
# The lock auto-releases when the subshell exits — normally, on error
# (set -e), or if the process is killed. No stale locks, no cleanup needed.
#
# flock -w gives up after LOCK_TIMEOUT seconds rather than waiting forever,
# so a wedged peer can never hang this job indefinitely.
(
    if ! flock -x -w "$LOCK_TIMEOUT" 200; then
        echo "ERROR: timed out after ${LOCK_TIMEOUT}s waiting for $LOCK_FILE" >&2
        exit 1
    fi

    echo "[$(date)] $(hostname): acquired env setup lock"

    if ! mamba env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        echo "Creating environment: $ENV_NAME"
        mamba create -n "$ENV_NAME" python=3.12 -y
    else
        echo "Environment $ENV_NAME already present, skipping create"
    fi

    # Activate inside the subshell so pip targets the right interpreter.
    # This activation does NOT leak to the parent shell.
    mamba activate "$ENV_NAME"

    # Idempotent; fast when already installed. Kept inside the lock so
    # two peers never run pip against the same env simultaneously.
    pip install -e "$HOME/projects/CryoVIT/[dev]"

    echo "[$(date)] $(hostname): releasing env setup lock"
) 200>"$LOCK_FILE"
# ---- End critical section ----

# Re-activate in the main shell (subshell activation didn't propagate).
# Safe to do without the lock: setup is guaranteed complete by this point.
mamba activate "$ENV_NAME"

echo "Using python: $(which python)"

python -m cryovit.training.train_model \
    model=$model \
    name="single_${sample_group}_${model}_${label_key}" \
    label_key=$label_key \
    datamodule=single \
    datamodule.sample=$sample \
    datamodule.split_id=$split_id

python -m cryovit.training.eval_model \
    model=$model \
    name="single_${sample_group}_${model}_${label_key}" \
    label_key=$label_key \
    datamodule=single \
    datamodule.sample=$sample \
    datamodule.split_id=$split_id