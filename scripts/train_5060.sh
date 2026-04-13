#!/bin/bash
# Train Qwen3-1.7B on SETA tasks using GRPO (single RTX 5060, 8GB)

pkill -9 sglang 2>/dev/null
sleep 2
ray stop --force 2>/dev/null
pkill -9 ray 2>/dev/null
pkill -9 python 2>/dev/null
sleep 2

set -ex

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

source <(sed 's/\r//' "$PROJECT_DIR/slime/scripts/models/qwen3-0.6B.sh")

CKPT_ARGS=(
    --hf-checkpoint "$PROJECT_DIR/models/Qwen3-0.6B"
    --ref-load "$PROJECT_DIR/models/Qwen3-0.6B-megatron"
    --save "$PROJECT_DIR/checkpoints/qwen3-0.6b-seta"
    --save-interval 20
    --rotary-base 1000000
    --tensorboard-dir "$PROJECT_DIR/checkpoints/qwen3-0.6b-seta/tb"
    --tensorboard
)

ROLLOUT_ARGS=(
    --prompt-data "$PROJECT_DIR/data/seta_prompts.jsonl"
    --input-key text
    --rollout-shuffle
    --reward-key score
    --num-rollout 1000
    --rollout-batch-size 4
    --n-samples-per-prompt 4
    --rollout-max-response-len 2048
    --rollout-temperature 1.0
    --max-agent-turns 10
    --rollout-max-context-len 4096

    --global-batch-size 16
    --balance-data
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.001
    --kl-loss-type low_var_kl
    --entropy-coef 0.0
    --eps-clip 0.2
    --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
)

# Single GPU: no tensor/pipeline parallelism
PERF_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1

    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1

    --use-dynamic-batch-size
    --max-tokens-per-gpu 4096
    --train-memory-margin-bytes 16777216
    --use-cpu-initialization

    # Offload Adam states to CPU to save GPU VRAM (will use swap on 16GB RAM system)
    --use-precision-aware-optimizer
    --optimizer-cpu-offload
    --decoupled-weight-decay
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.50
    --sglang-disable-cuda-graph
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-backend flash
    --main-grads-dtype bf16
)

CUSTOM_ARGS=(
    --custom-generate-function-path seta.docker_environment.generate
    --custom-rm-path rewards.reward.reward_func
)

# Install missing dependencies for SETA environment
pip install camel-ai docker --quiet --root-user-action=ignore 2>&1 | tail -3

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False is set in RUNTIME_ENV_JSON
# to allow CUDA IPC sharing during weight sync. PyTorch >= 2.1 enables expandable
# segments by default, which is explicitly incompatible with CUDA IPC.
# The file copy below ensures the project's slime code overrides the container's,
# since Python adds the script dir (/root/slime) to sys.path before PYTHONPATH.
cp "$PROJECT_DIR/slime/slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py" \
   /root/slime/slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py
rm -f /root/slime/slime/backends/megatron_utils/update_weight/__pycache__/update_weight_from_tensor.cpython-*.pyc

export MASTER_ADDR="127.0.0.1"
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export RAY_DISABLE_MEMORY_MONITOR=1
export RAY_memory_usage_threshold=0.999
ray start --head \
    --node-ip-address "$MASTER_ADDR" \
    --num-gpus 1 \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"$PROJECT_DIR/slime:$PROJECT_DIR:/root/Megatron-LM\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO\": \"0\",
    \"NCCL_P2P_DISABLE\": \"1\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:False\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 /root/slime/train.py \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 1 \
    --colocate \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}" \
    "${CUSTOM_ARGS[@]}"
