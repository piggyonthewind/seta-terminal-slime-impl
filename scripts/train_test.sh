#!/bin/bash
# Train Qwen3.5-9B on SETA tasks using GRPO (2x A100-SXM4-80GB)
#
# Memory budget per GPU (TP=2, no CPU offload):
#   Model weights (bf16):        ~9 GB
#   Optimizer states (bf16 Adam): ~36 GB
#   Activations + gradients:      ~5 GB
#   SGLang KV cache:              ~24 GB  (mem_fraction_static=0.30)
#   Total:                        ~74 GB  (fits in 80 GB)

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

source <(sed 's/\r//' "$PROJECT_DIR/slime/scripts/models/qwen3.5-9B.sh")

CKPT_ARGS=(
    --hf-checkpoint "$PROJECT_DIR/models/Qwen3.5-9B"
    --ref-load "$PROJECT_DIR/models/Qwen3.5-9B-megatron"
    --save "$PROJECT_DIR/checkpoints/qwen3.5-9b-seta"
    --save-interval 999999
    --tensorboard-dir "$PROJECT_DIR/checkpoints/qwen3.5-9b-seta/tb"
    --tensorboard
)

ROLLOUT_ARGS=(
    --prompt-data "$PROJECT_DIR/data/seta_prompts.jsonl"
    --input-key text
    --rollout-shuffle
    --reward-key score
    --num-rollout 1
    --rollout-batch-size 1
    --n-samples-per-prompt 1
    --rollout-max-response-len 4096
    --rollout-temperature 1.0
    --max-agent-turns 5
    --rollout-max-context-len 8192

    --global-batch-size 1
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

# 2x A100-SXM4-80GB: TP=2 required (single GPU can't fit 9B weights + optimizer states)
PERF_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 1
    --context-parallel-size 1

    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1

    --use-dynamic-batch-size
    --max-tokens-per-gpu 16384
    --train-memory-margin-bytes 1073741824
    --use-cpu-initialization

    --use-precision-aware-optimizer
    --decoupled-weight-decay
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 2
    --sglang-mem-fraction-static 0.15
    --sglang-disable-cuda-graph
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-backend flash
    --main-grads-dtype bf16
    --loss-mask-type qwen3_5
)

WANDB_ARGS=(
    --use-wandb
    --wandb-project seta-qwen3.5-9b-a100
    --wandb-group seta-test
)

CUSTOM_ARGS=(
    --custom-generate-function-path seta.docker_environment.generate
    --custom-rm-path rewards.reward.reward_func
    --custom-rollout-log-function-path seta.log_rollout_metrics.log_rollout_metrics
    --log-passrate
)

# Install missing dependencies for SETA environment
pip install camel-ai docker --quiet --root-user-action=ignore 2>&1 | tail -3

# Fix SGLang multimodal processor compatibility with Qwen3.5 tokenizer
# (processor.tokenizer.convert_ids_to_tokens -> processor.convert_ids_to_tokens)
sed -i 's/processor\.tokenizer\.convert_ids_to_tokens/processor.convert_ids_to_tokens/g' \
    /sgl-workspace/sglang/python/sglang/srt/multimodal/processors/base_processor.py
find /sgl-workspace/sglang/python/sglang/srt/multimodal/processors/ -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False is set in RUNTIME_ENV_JSON
# to allow CUDA IPC sharing during weight sync. PyTorch >= 2.1 enables expandable
# segments by default, which is explicitly incompatible with CUDA IPC.
cp "$PROJECT_DIR/slime/slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py" \
   /root/slime/slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py
rm -f /root/slime/slime/backends/megatron_utils/update_weight/__pycache__/update_weight_from_tensor.cpython-*.pyc

export MASTER_ADDR="127.0.0.1"
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export RAY_DISABLE_MEMORY_MONITOR=1
export RAY_memory_usage_threshold=0.999
ray start --head \
    --node-ip-address "$MASTER_ADDR" \
    --num-gpus 2 \
    --object-store-memory=10000000000 \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"$PROJECT_DIR/slime:$PROJECT_DIR:/root/Megatron-LM\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO\": \"0\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:False\",
    \"WANDB_API_KEY\": \"$WANDB_API_KEY\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 /root/slime/train.py \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 2 \
    --colocate \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${CUSTOM_ARGS[@]}"
