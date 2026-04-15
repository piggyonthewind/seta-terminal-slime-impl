#!/bin/bash
# Full pipeline: convert weights → launch training
# Run this after slimerl/slime:latest is pulled and data/seta-env/ is downloaded.
#
# Usage (from project root, outside Docker):
#   nohup bash scripts/setup_and_train.sh > /tmp/setup_and_train.log 2>&1 &

set -euo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

DOCKER="sg docker -c docker"

echo "=== [1/3] Waiting for slimerl/slime:latest ==="
until sg docker -c "docker image inspect slimerl/slime:latest" &>/dev/null; do
    echo "  Docker image not ready yet, waiting 30s..."
    sleep 30
done
echo "  Image ready."

echo "=== [2/3] Converting Qwen3.5-9B weights to Megatron format (TP=2) ==="
if [ -d "$PROJECT_DIR/models/Qwen3.5-9B-megatron" ]; then
    echo "  Already converted, skipping."
else
    sg docker -c "docker run --rm \
        --gpus all \
        --shm-size 16g \
        --network host \
        -v '$PROJECT_DIR':/workspace/project \
        -w /workspace/project \
        slimerl/slime:latest \
        bash -c \"
            pip install --upgrade transformers --quiet --root-user-action=ignore 2>&1 | tail -1 &&
            PYTHONPATH=/root/slime:/workspace/project:/root/Megatron-LM \
            CUDA_DEVICE_MAX_CONNECTIONS=1 \
            torchrun --nproc_per_node=2 /workspace/project/slime/tools/convert_hf_to_torch_dist.py \
                --spec slime_plugins.models.qwen3_5 get_qwen3_5_spec \
                --disable-bias-linear \
                --qk-layernorm \
                --group-query-attention \
                --num-attention-heads 16 \
                --num-query-groups 4 \
                --kv-channels 256 \
                --num-layers 32 \
                --hidden-size 4096 \
                --ffn-hidden-size 12288 \
                --use-gated-attention \
                --normalization RMSNorm \
                --apply-layernorm-1p \
                --position-embedding-type rope \
                --norm-epsilon 1e-6 \
                --rotary-percent 0.25 \
                --rotary-base 10000000 \
                --swiglu \
                --untie-embeddings-and-output-weights \
                --vocab-size 248320 \
                --attention-output-gate \
                --hf-checkpoint /workspace/project/models/Qwen3.5-9B \
                --save /workspace/project/models/Qwen3.5-9B-megatron \
                --tensor-model-parallel-size 2 \
                --pipeline-model-parallel-size 1 \
                --context-parallel-size 1
        \""
    echo "  Conversion complete."
fi

echo "=== [3/3] Launching training ==="
sg docker -c "docker run --rm \
    --gpus all \
    --shm-size 16g \
    --memory=160g \
    --network host \
    -e WANDB_API_KEY \
    -v '$PROJECT_DIR':/workspace/project \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -w /workspace/project \
    slimerl/slime:latest \
    bash scripts/train_5060.sh"

echo "=== Training complete ==="
