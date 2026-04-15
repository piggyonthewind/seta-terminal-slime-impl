# SETA Training Pipeline — Debugging Log

Date: 2026-04-14 / 2026-04-15
Machine: 2x A100-SXM4-80GB, 188 GB RAM, 196 GB disk

## Overview

The pipeline (`scripts/setup_and_train.sh`) has three stages:
1. Docker pull of `slimerl/slime:latest` (~25.6 GB compressed, ~54 GB uncompressed)
2. Weight conversion: Qwen3.5-9B HuggingFace → Megatron format (TP=2)
3. GRPO training via SLIME framework

Stage 1 completed after network issues. Stage 2 required four code fixes.
Stage 3 has not yet been attempted.

---

## Stage 1: Docker Pull

### Issue 1.1 — Network failure during first pull attempt

**Symptom:** After ~2.5 hours of downloading (~14 GB pulled), the Docker Hub CDN
dropped the connection repeatedly:
```
Download failed after 5 attempts: EOF
dial tcp 103.39.76.66:443: i/o timeout
```
Docker cleaned up ALL partial downloads (overlay2 reset from 14 GB to 151 MB).

**Fix:** Manually restarted the pull:
```bash
nohup sg docker -c "docker pull slimerl/slime:latest" >> /tmp/docker_pull.log 2>&1 &
```
The second attempt completed in ~80 minutes (17:27–18:48) with rates fluctuating
between 1–69 MB/s due to CDN throttling.

### Issue 1.2 — CDN throttling pattern

Download rates followed a pattern: burst at ~69 MB/s initially, then throttled to
~1–10 MB/s for large model-weight blobs. The 68-layer image had sequential
dependency chains — extraction ("Pull complete") could only proceed when all
ancestor layers finished.

---

## Stage 2: Weight Conversion

The original conversion command in `setup_and_train.sh`:
```bash
python /root/slime/tools/convert_hf_to_torch_dist.py \
    --model-type qwen3_5 \
    --hf-checkpoint /workspace/project/models/Qwen3.5-9B \
    --save /workspace/project/models/Qwen3.5-9B-megatron \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 1
```

This command had **four separate bugs**, each discovered and fixed sequentially.

### Fix 2.1 — `--model-type` not recognized

**Error:**
```
convert_hf_to_torch_dist.py: error: unrecognized arguments: --model-type qwen3_5
```

**Root cause:** The conversion script's argparse (via Megatron's `parse_args`) has
no `--model-type` flag. Model architecture is specified through individual args
(`--spec`, `--num-layers`, `--hidden-size`, etc.), which were missing entirely.

**Fix:** Replaced `--model-type qwen3_5` with the full model architecture args
from `slime/scripts/models/qwen3.5-9B.sh`:
```bash
--spec slime_plugins.models.qwen3_5 get_qwen3_5_spec \
--disable-bias-linear --qk-layernorm --group-query-attention \
--num-attention-heads 16 --num-query-groups 4 --kv-channels 256 \
--num-layers 32 --hidden-size 4096 --ffn-hidden-size 12288 \
--use-gated-attention --normalization RMSNorm --apply-layernorm-1p \
--position-embedding-type rope --norm-epsilon 1e-6 \
--rotary-percent 0.25 --rotary-base 10000000 --swiglu \
--untie-embeddings-and-output-weights --vocab-size 248320 \
--attention-output-gate
```

**File changed:** `scripts/setup_and_train.sh`

### Fix 2.2 — World size not divisible by total_model_size (round 1)

**Error:**
```
AssertionError: world size (1) is not divisible by total_model_size (total_model_size=2)
```

**Root cause:** The original command used `python` (single process, WORLD_SIZE=1),
but TP=2 requires at least 2 processes.

**Fix:** Changed `python` to `torchrun --nproc_per_node=2`.

**File changed:** `scripts/setup_and_train.sh`

### Fix 2.3 — World size not divisible by total_model_size (round 2)

**Error:**
```
AssertionError: world size (2) is not divisible by total_model_size (total_model_size=4)
```

**Root cause:** Two sub-issues combined:

1. **Auto-PP logic in `convert_hf_to_torch_dist.py`** (lines 55–56):
   ```python
   if args.pipeline_model_parallel_size == 1 and world_size > 1:
       pp_size = world_size  # Sets PP=2
   ```
   When PP=1 and world_size > 1, the script auto-sets PP = world_size (2).
   With TP=2 already set, total_model_size = TP(2) × PP(2) × CP(1) = 4,
   which exceeds world_size(2).

2. **Missing `--context-parallel-size 1`** — Megatron defaults CP to 1, but
   adding it explicitly was needed for clarity and to rule out other defaults.

**Fix (code change):** Modified `slime/tools/convert_hf_to_torch_dist.py` to
make the auto-PP logic TP-aware:
```python
# Before (broken):
if args.pipeline_model_parallel_size == 1 and world_size > 1:
    pp_size = world_size

# After (fixed):
tp_cp = args.tensor_model_parallel_size * getattr(args, 'context_parallel_size', 1)
if args.pipeline_model_parallel_size == 1 and world_size > tp_cp:
    pp_size = world_size // tp_cp
```

With TP=2 on 2 GPUs, `world_size(2) > tp_cp(2)` is False, so auto-PP is
skipped and PP stays at 1. total_model_size = 2 × 1 × 1 = 2 = world_size. ✓

**Fix (script change):** Also changed the script to run the LOCAL copy of the
conversion script (mounted at `/workspace/project/slime/tools/`) instead of
the Docker image's copy at `/root/slime/tools/`, so the code fix takes effect.

Added `--context-parallel-size 1` to the conversion args for explicitness.

**Files changed:**
- `slime/tools/convert_hf_to_torch_dist.py` (auto-PP logic)
- `scripts/setup_and_train.sh` (script path + CP arg)

### Fix 2.4 — CUDA_DEVICE_MAX_CONNECTIONS not set

**Error:**
```
AssertionError: Using tensor model parallelism or context parallelism require
setting the environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1
```

**Root cause:** Megatron requires `CUDA_DEVICE_MAX_CONNECTIONS=1` when using
tensor parallelism. The training script (`train_5060.sh`) sets this via Ray's
runtime env, but the conversion command didn't set it.

**Fix:** Added `CUDA_DEVICE_MAX_CONNECTIONS=1` as an env var in the conversion
`bash -c` command.

**File changed:** `scripts/setup_and_train.sh`

### Fix 2.5 — `qwen3_5` model type not recognized by transformers

**Error:**
```
ValueError: The checkpoint you are trying to load has model type `qwen3_5`
but Transformers does not recognize this architecture.
```

**Root cause:** The `transformers` library inside the Docker image is too old
to recognize the `qwen3_5` model type. The HF checkpoint's `config.json` has
`"model_type": "qwen3_5"`, but there's no custom `configuration_qwen3_5.py`
in the checkpoint directory for `trust_remote_code=True` to import.

The error occurs in `AutoBridge.from_pretrained()` → `AutoConfig.from_pretrained()`
at line 119 of the conversion script, after all arg validation passes.

**Fix:** Added `pip install --upgrade transformers` before the conversion command
inside the Docker `bash -c` block.

**File changed:** `scripts/setup_and_train.sh`

### Conversion outcome

After all five fixes, the conversion completed successfully on 2026-04-14 at
~19:18 CST. Output checkpoint:
```
models/Qwen3.5-9B-megatron/
├── latest_checkpointed_iteration.txt  ("release")
└── release/
    ├── __0_0.distcp   (5.29 GB)  # TP rank 0
    ├── __0_1.distcp   (5.29 GB)
    ├── __1_0.distcp   (3.67 GB)  # TP rank 1
    ├── __1_1.distcp   (3.67 GB)
    ├── common.pt
    ├── .metadata
    ├── metadata.json
    └── modelopt_run_config.yaml
```
Total: ~17.9 GB in distributed checkpoint format.

---

## System Crash — Disk Exhaustion

**When:** Between 2026-04-14 ~19:18 and 2026-04-15 ~11:33.

**Cause:** Disk reached 95% utilization (9.7 GB free on 196 GB disk) during the
conversion stage. The breakdown:
- Docker image `slimerl/slime:latest`: ~54 GB (image layers in overlay2)
- Docker content store (compressed blobs during pull): ~25 GB
- HF model weights (`Qwen3.5-9B`): ~19 GB
- Megatron checkpoint output: ~18 GB
- System + other files: ~62 GB
- **Total: ~178 GB of 196 GB**

The remaining 9.7 GB was insufficient for Docker container overhead + pip install
during the conversion, likely causing the system to crash and reboot.

**After reboot:** Docker cleaned up temp/content store data, recovering disk to
76 GB used (112 GB free). The Docker image and Megatron checkpoint survived intact.

---

## Summary of All Changes

### `scripts/setup_and_train.sh` (cumulative diff from original)

The conversion block was rewritten from:
```bash
python /root/slime/tools/convert_hf_to_torch_dist.py \
    --model-type qwen3_5 \
    --hf-checkpoint ... --save ... \
    --tensor-model-parallel-size 2 --pipeline-model-parallel-size 1
```

To:
```bash
pip install --upgrade transformers --quiet --root-user-action=ignore 2>&1 | tail -1 &&
PYTHONPATH=/root/slime:/workspace/project:/root/Megatron-LM \
CUDA_DEVICE_MAX_CONNECTIONS=1 \
torchrun --nproc_per_node=2 /workspace/project/slime/tools/convert_hf_to_torch_dist.py \
    --spec slime_plugins.models.qwen3_5 get_qwen3_5_spec \
    [full model architecture args from qwen3.5-9B.sh] \
    --hf-checkpoint ... --save ... \
    --tensor-model-parallel-size 2 --pipeline-model-parallel-size 1 \
    --context-parallel-size 1
```

### `slime/tools/convert_hf_to_torch_dist.py`

Auto-PP logic changed to be TP-aware (lines 55–57):
```python
tp_cp = args.tensor_model_parallel_size * getattr(args, 'context_parallel_size', 1)
if args.pipeline_model_parallel_size == 1 and world_size > tp_cp:
    pp_size = world_size // tp_cp
```

---

## Stage 3: Training

Not yet attempted. Next step is to launch `train_5060.sh` inside the Docker
container. Key concerns:
- Disk space: 112 GB free should be sufficient, but checkpoints (~18 GB each,
  saved every 20 steps) will accumulate. Monitor and prune as needed.
- The `train_5060.sh` script also runs `pip install camel-ai docker` and patches
  `update_weight_from_tensor.py` before starting Ray and submitting the job.
