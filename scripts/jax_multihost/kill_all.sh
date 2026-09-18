#!/usr/bin/env bash
set -euo pipefail
PROJECT="tpu-pytorch"
ZONE="us-west1-c"
TPU_NAME="yho-v5e-16"
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --worker=all \
    --command='
      pkill -9 -u "$(id -un)" python || true
      rm -f /tmp/libtpu_lockfile
      echo "Cleaned $(hostname)"
    '