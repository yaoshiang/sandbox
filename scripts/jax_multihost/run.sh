#!/usr/bin/env bash
# Run with: ./run.sh path/to/script.py (after chmod +x run.sh) or `bash run.sh path/to/script.py`. Do NOT source it.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 path/to/script.py" >&2
    exit 1
fi

set -x

PROJECT="tpu-pytorch"
ZONE="us-west1-c"
TPU_NAME="yho-v5e-16"
LOCAL_SCRIPT="$1"
REMOTE_DIR="/home/yho_google_com/"
REMOTE_SCRIPT="${REMOTE_DIR}$(basename "$LOCAL_SCRIPT")"
PYTHON_BIN="${PYTHON_BIN:-/home/yho_google_com/venv312/bin/python}"

# Step 1: broadcast the script to every worker (destination is positional: TPU_NAME:DEST_DIR).
gcloud alpha compute tpus tpu-vm scp "$LOCAL_SCRIPT" "${TPU_NAME}:${REMOTE_DIR}" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --worker=all

# Step 2: run it on every worker; gcloud injects TPU_WORKER_ID/HOSTNAMES automatically.
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --worker=all \
    --command="$PYTHON_BIN $REMOTE_SCRIPT"
