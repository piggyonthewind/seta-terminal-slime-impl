# Agentic RL Retool — Implementation Plan
# SETA Terminal Agent RL on Qwen3.5-2B + RTX 3090 using SLIME

Reference: https://eigent-ai.notion.site/SETA-Scaling-Environments-for-Terminal-Agents-2d2511c70ba280a9b7c0fe3e7f1b6ab8

---

## Target Results (from SETA paper, Qwen3-8B)
| Metric | Baseline | After RL |
|---|---|---|
| Pass@8 (79 tasks) | 6 | 10 |
| Avg unit test pass ratio | 0.38 | 0.45 |
| Improvement | — | +20.2% |

Our prototype: Qwen3.5-2B on single RTX 3090 — validate pipeline end-to-end.

---

## Memory Budget (Qwen3.5-2B on 3090, 24GB)

| Component | Memory | Notes |
|---|---|---|
| Model weights (BF16) | 4.0 GB | 2B × 2 bytes |
| Gradients (BF16) | 4.0 GB | |
| Optimizer states (GPU) | 4.0 GB | 20GB offloaded to CPU RAM |
| Activations (checkpointed) | 0.4 GB | --recompute-granularity full |
| SGLang KV cache (30%) | 7.2 GB | --sglang-mem-fraction-static 0.3 |
| Framework overhead | 1.0 GB | |
| **TOTAL** | **20.6 GB** | **3.4 GB safety margin** |

Docker terminal containers: **pure CPU** — no GPU needed. Runs alongside training freely.

---

## Architecture

```
RTX 3090 (24GB GPU)
│
├── Megatron-LM Training (GRPO)          ~13.4 GB
│   └── Qwen3.5-2B full-param
│
└── SGLang Inference (colocated)         ~7.2 GB
    └── Rollout generation

CPU + RAM (separate from GPU)
│
├── Docker Container 1 (Ubuntu bash)  ─┐
├── Docker Container 2 (Ubuntu bash)   ├── 2-4 parallel terminal environments
├── Docker Container 3 (Ubuntu bash)   │   each running one RL episode
└── Docker Container 4 (Ubuntu bash)  ─┘
    │
    └── run-tests.sh → /logs/verifier/ctrf.json
                     → /logs/verifier/reward.txt
```

---

## Project Structure

```
agentic_rl_retool/
├── plan.md
├── models/
│   └── Qwen3.5-2B/                    # Downloaded HF weights
├── data/
│   └── seta-env/                      # camel-ai/seta-env (400 tasks)
│       └── tasks/{task_id}/
│           ├── task.yaml              # instruction, category, difficulty
│           ├── Dockerfile             # task environment image
│           ├── run-tests.sh           # writes reward to /logs/verifier/
│           ├── solution.sh            # reference solution (optional)
│           └── tests/
│               └── test_outputs.py   # pytest validation
├── slime/                             # cloned THUDM/slime (patched)
├── examples/
│   └── seta/
│       ├── docker_environment.py      # custom generate function
│       ├── docker_reward.py           # custom reward function
│       └── docker_pool.py             # local container pool manager
└── scripts/
    └── train_3090.sh                  # launch script
```

---

## Step 1 — Clone and Patch SLIME (Single-GPU Fix)

### 1.1 Clone SLIME inside Docker container
```bash
# Inside SLIME Docker container (already pulled: slimerl/slime:latest)
docker run --gpus all --ipc=host --shm-size=16g \
  -v C:/Users/vanst/Desktop/agentic_rl_retool:/workspace \
  -it slimerl/slime:latest /bin/bash

cd /workspace
git clone https://github.com/THUDM/slime.git
```

### 1.2 Apply single-GPU patch
**File:** `slime/slime/ray/train_actor.py`
**Lines 58-59** — current code:
```python
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(f"cuda:{local_rank}")
```

**Replace with:**
```python
local_rank = int(os.environ.get("LOCAL_RANK", 0))
num_available_gpus = torch.cuda.device_count()
if local_rank >= num_available_gpus:
    logger.warning(
        f"LOCAL_RANK {local_rank} >= num_available_gpus {num_available_gpus}, "
        f"using device 0 instead"
    )
    local_rank = 0
torch.cuda.set_device(f"cuda:{local_rank}")
```

That's the **only required change** to SLIME core. No other files need modification.

---

## Step 2 — Convert Model to Megatron Format

SLIME requires Megatron torch_dist format, not raw HuggingFace weights.

```bash
# Inside SLIME Docker container
cd /workspace/slime

python tools/convert_hf_to_torch_dist.py \
    --model-type qwen \
    --hf-checkpoint /workspace/models/Qwen3.5-2B \
    --save /workspace/models/Qwen3.5-2B-megatron \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1
```

---

## Step 3 — Download SETA Dataset

```bash
# On Windows host (already have hf CLI in agentic_rl_retool env)
C:/Users/vanst/miniconda3/envs/agentic_rl_retool/Scripts/hf.exe download \
    --repo-type dataset camel-ai/seta-env \
    --local-dir "C:/Users/vanst/Desktop/agentic_rl_retool/data/seta-env"
```

### Task structure (per task):
```yaml
# task.yaml
id: task_001
instruction: "Install nginx and configure it to serve a static HTML page..."
category: system-administration   # one of 7 categories
difficulty: medium
max_agent_timeout_sec: 180
max_test_timeout_sec: 30
test_scripts:
  - setup-uv-pytest.sh
  - run-uv-pytest.sh
run_tests_in_same_shell: true
```

```bash
# run-tests.sh — writes reward signal
#!/bin/bash
pytest --ctrf /logs/verifier/ctrf.json tests/test_outputs.py
# Writes:
#   /logs/verifier/reward.txt  → "1" (pass) or "0" (fail)
#   /logs/verifier/ctrf.json   → detailed test results (pass ratio)
```

---

## Step 4 — Implement Custom Generate Function

**File:** `examples/seta/docker_environment.py`

```python
"""
Custom SLIME generate function for Docker-based terminal agent rollouts.
Wire via: --custom-generate-function-path examples.seta.docker_environment.generate
"""

import asyncio
import re
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample
import docker


async def generate(
    args,
    sample: Sample,
    sampling_params: dict,
    evaluation: bool = False,
) -> Sample:
    """
    Multi-turn terminal agent rollout inside a Docker container.
    
    Each turn:
      1. Model generates action wrapped in <bash>...</bash> tags
      2. Command executes in Docker container via docker exec
      3. stdout/stderr injected as observation (loss_mask=0)
      4. Repeat up to max_turns
    """
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # Spawn Docker container for this episode
    task_path = sample.metadata.get("task_path")
    container = _spawn_container(task_path)
    sample.metadata["container_id"] = container.id

    # Tokenize prompt
    prompt_tokens = state.tokenizer(
        sample.prompt, add_special_tokens=False
    )["input_ids"]
    response_tokens = []
    loss_masks = []
    rollout_logprobs = []

    max_turns = getattr(args, "max_agent_turns", 20)

    try:
        for turn in range(max_turns):
            current_tokens = prompt_tokens + response_tokens

            # Generate next action from model via SGLang HTTP
            payload = {
                "input_ids": current_tokens,
                "sampling_params": sampling_params,
                "return_logprob": True,
            }
            output = await post(url, payload)

            finish_type = output["meta_info"]["finish_reason"]["type"]
            if finish_type == "abort":
                sample.status = Sample.Status.ABORTED
                break

            # Extract tokens + logprobs
            if "output_token_logprobs" in output["meta_info"]:
                logprobs = [x[0] for x in output["meta_info"]["output_token_logprobs"]]
                token_ids = [x[1] for x in output["meta_info"]["output_token_logprobs"]]
                model_text = state.tokenizer.decode(token_ids)
            else:
                model_text = output["text"]
                token_ids = state.tokenizer(
                    model_text, add_special_tokens=False
                )["input_ids"]
                logprobs = [0.0] * len(token_ids)

            # Model output → trainable (loss_mask = 1)
            response_tokens.extend(token_ids)
            loss_masks.extend([1] * len(token_ids))
            rollout_logprobs.extend(logprobs)

            if finish_type == "length":
                sample.status = Sample.Status.TRUNCATED
                break

            # Parse bash command from model output
            cmd_match = re.search(r"<bash>(.*?)</bash>", model_text, re.DOTALL)
            finish_match = re.search(r"<finish>", model_text)

            if finish_match or not cmd_match:
                sample.status = Sample.Status.COMPLETED
                break

            bash_cmd = cmd_match.group(1).strip()

            # Execute in Docker
            exec_result = container.exec_run(
                ["bash", "-c", bash_cmd],
                stdout=True, stderr=True, demux=True,
            )
            stdout = exec_result.output[0].decode() if exec_result.output[0] else ""
            stderr = exec_result.output[1].decode() if exec_result.output[1] else ""
            returncode = exec_result.exit_code

            sample.metadata["last_returncode"] = returncode

            # Format observation
            observation = (
                f"<observation>\n"
                f"exit_code: {returncode}\n"
                f"stdout:\n{stdout}\n"
                f"stderr:\n{stderr}\n"
                f"</observation>\n"
            )

            # Observation → non-trainable (loss_mask = 0)
            obs_tokens = state.tokenizer(
                observation, add_special_tokens=False
            )["input_ids"]
            response_tokens.extend(obs_tokens)
            loss_masks.extend([0] * len(obs_tokens))
            rollout_logprobs.extend([0.0] * len(obs_tokens))

        # Finalize sample
        sample.tokens = prompt_tokens + response_tokens
        sample.response_length = len(response_tokens)
        sample.response = state.tokenizer.decode(response_tokens)
        sample.loss_mask = loss_masks
        sample.rollout_log_probs = rollout_logprobs

        if sample.status is None:
            sample.status = Sample.Status.COMPLETED

    finally:
        # Run tests and store results before cleanup
        _run_tests_and_store(container, sample)
        container.stop()
        container.remove()

    return sample


def _spawn_container(task_path: str):
    """Build (if needed) and start a Docker container for the task."""
    client = docker.from_env()
    image_tag = f"seta-task:{task_path.replace('/', '-')}"

    # Build image (cached after first build)
    try:
        client.images.get(image_tag)
    except docker.errors.ImageNotFound:
        client.images.build(path=task_path, tag=image_tag, rm=True)

    container = client.containers.run(
        image_tag,
        command="sleep infinity",
        detach=True,
        remove=False,
        working_dir="/workspace",
    )
    return container


def _run_tests_and_store(container, sample: Sample):
    """Run run-tests.sh and store raw results in metadata."""
    result = container.exec_run(
        ["bash", "/run-tests.sh"],
        stdout=True, stderr=True, demux=True,
    )
    stdout = result.output[0].decode() if result.output[0] else ""

    # Read reward file written by run-tests.sh
    try:
        reward_result = container.exec_run(
            ["cat", "/logs/verifier/reward.txt"],
            stdout=True, demux=True,
        )
        raw = reward_result.output[0].decode().strip() if reward_result.output[0] else "0"
        binary_reward = int(raw)
    except Exception:
        binary_reward = 0

    # Parse unit test pass ratio from ctrf.json if available
    try:
        ctrf_result = container.exec_run(
            ["cat", "/logs/verifier/ctrf.json"],
            stdout=True, demux=True,
        )
        import json
        ctrf = json.loads(ctrf_result.output[0].decode())
        passed = ctrf["results"]["summary"]["passed"]
        total = ctrf["results"]["summary"]["tests"]
        pass_ratio = passed / max(total, 1)
    except Exception:
        pass_ratio = float(binary_reward)

    sample.metadata["pass_ratio"] = pass_ratio
    sample.metadata["binary_reward"] = binary_reward
    sample.metadata["test_stdout"] = stdout
```

---

## Step 5 — Implement Custom Reward Function

**File:** `examples/seta/docker_reward.py`

```python
"""
Custom SLIME reward function implementing SETA reward formula.
Wire via: --custom-rm-path examples.seta.docker_reward.reward_func

Reward formula (from SETA paper):
    R = u + 1   if u == 1.0  (full success)
    R = u       if u < 1.0   (partial/fail)
where u = unit test pass ratio ∈ [0, 1]
Max reward = 2.0
"""

from slime.utils.types import Sample


async def reward_func(args, sample: Sample, **kwargs) -> float:
    """SETA reward: unit test pass ratio + bonus for full success."""
    u = sample.metadata.get("pass_ratio", 0.0)
    bonus = 1.0 if u == 1.0 else 0.0
    return u + bonus
```

---

## Step 6 — Local Docker Container Pool Manager

**File:** `examples/seta/docker_pool.py`

```python
"""
Manages a small pool of local Docker containers for RL rollouts.
For 3090 prototype: 2-4 containers running pure CPU bash environments.
"""

import asyncio
import threading
import time
import docker
import random
import yaml
from pathlib import Path


class LocalDockerPool:
    """
    Pool of 2-4 Docker containers for terminal agent RL episodes.
    Containers are CPU-only — no GPU needed.
    """

    def __init__(self, task_dir: str, num_containers: int = 4):
        self.task_dir = Path(task_dir)
        self.num_containers = num_containers
        self.client = docker.from_env()
        self.semaphore = asyncio.Semaphore(num_containers)

        # Load all available tasks
        self.tasks = self._load_tasks()

        # Start cleanup daemon
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True
        )
        self._cleanup_thread.start()

    def _load_tasks(self) -> list[dict]:
        tasks = []
        for task_dir in self.task_dir.iterdir():
            yaml_path = task_dir / "task.yaml"
            if yaml_path.exists():
                with open(yaml_path) as f:
                    meta = yaml.safe_load(f)
                    meta["task_path"] = str(task_dir)
                    tasks.append(meta)
        return tasks

    def sample_tasks(self, n: int, difficulty_weights=None) -> list[dict]:
        """Sample n tasks, weighted by difficulty."""
        if difficulty_weights is None:
            difficulty_weights = {"easy": 0.2, "medium": 0.7, "hard": 0.1}
        weights = [difficulty_weights.get(t.get("difficulty", "medium"), 0.5)
                   for t in self.tasks]
        return random.choices(self.tasks, weights=weights, k=n)

    async def run_episode(self, task: dict) -> dict:
        """Run one full RL episode for a task (container lifecycle managed here)."""
        async with self.semaphore:  # Limit to num_containers concurrent episodes
            return await self._run_episode_inner(task)

    async def _run_episode_inner(self, task: dict) -> dict:
        # Container spawn, episode, test, cleanup — handled in docker_environment.py
        # This just tracks metadata
        return {
            "task_id": task.get("id"),
            "task_path": task["task_path"],
            "category": task.get("category"),
            "difficulty": task.get("difficulty"),
        }

    def _cleanup_loop(self):
        """Background daemon: remove stopped containers every 2 minutes."""
        while True:
            try:
                for container in self.client.containers.list(
                    all=True, filters={"status": "exited"}
                ):
                    container.remove()
            except Exception:
                pass
            time.sleep(120)
```

---

## Step 7 — Training Launch Script

**File:** `scripts/train_3090.sh`

```bash
#!/bin/bash
# SETA Terminal Agent RL Training — Qwen3.5-2B on RTX 3090
# Run inside SLIME Docker container with GPU passthrough

set -e

WORKSPACE=/workspace
MODEL_PATH=$WORKSPACE/models/Qwen3.5-2B-megatron
TASK_DIR=$WORKSPACE/data/seta-env/tasks
CHECKPOINT_DIR=$WORKSPACE/checkpoints/qwen3.5-2b-seta

# Start Ray on single GPU
ray start --head --num-gpus 1 --num-cpus 8

# Launch training
python $WORKSPACE/slime/train.py \
    \
    `# Model` \
    --model-path $MODEL_PATH \
    --tokenizer-path $WORKSPACE/models/Qwen3.5-2B \
    \
    `# Single GPU config (patched for world_size=1)` \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 1 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --colocate \
    \
    `# Memory optimization (fits 20.6GB in 24GB)` \
    --optimizer-cpu-offload \
    --recompute-granularity full \
    --sglang-mem-fraction-static 0.3 \
    --max-tokens-per-gpu 4608 \
    \
    `# Rollout config` \
    --rollout-batch-size 2 \
    --n-samples-per-prompt 8 \
    --rollout-max-context-len 8192 \
    \
    `# Custom SETA functions` \
    --custom-generate-function-path examples.seta.docker_environment.generate \
    --custom-rm-path examples.seta.docker_reward.reward_func \
    \
    `# Agent config (passed through args)` \
    --max-agent-turns 20 \
    --docker-image ubuntu:22.04 \
    --docker-work-dir /workspace \
    \
    `# GRPO training` \
    --advantage-estimator grpo \
    --kl-loss-coef 0.0 \
    --entropy-coef 0.0 \
    --eps-clip 0.2 \
    --eps-clip-high 0.28 \
    --learning-rate 1e-6 \
    --global-batch-size 16 \
    --micro-batch-size 1 \
    \
    `# Training schedule` \
    --train-iters 500 \
    --save $CHECKPOINT_DIR \
    --save-interval 50 \
    \
    `# Data` \
    --task-dir $TASK_DIR \
    \
    `# Logging` \
    --log-interval 1 \
    --wandb-project seta-qwen3.5-2b-3090
```

---

## Step 8 — System Prompt for Terminal Agent

The model needs a system prompt defining its action space:

```python
SYSTEM_PROMPT = """You are a terminal agent. You solve tasks by executing bash commands
in an Ubuntu terminal environment.

To execute a bash command, wrap it in XML tags:
<bash>your command here</bash>

The command output will be returned as:
<observation>
exit_code: 0
stdout: ...
stderr: ...
</observation>

When you have completed the task, signal completion with:
<finish>Task completed.</finish>

Guidelines:
- Execute one command at a time and observe the output before proceeding
- You have up to 20 turns to complete the task
- The working directory is /workspace
- Write notes with: echo "note" >> /workspace/notes.txt
"""
```

---

## Implementation Order

### Week 1 — Foundation
- [ ] Download seta-env dataset
- [ ] Explore 5-10 tasks manually: read task.yaml, build Docker image, run run-tests.sh
- [ ] Apply single-GPU patch to slime/ray/train_actor.py (2 lines)
- [ ] Convert Qwen3.5-2B weights to Megatron format
- [ ] Verify SLIME starts with --actor-num-gpus-per-node 1

### Week 2 — Environment Loop
- [ ] Implement docker_pool.py (container spawn/cleanup)
- [ ] Implement docker_environment.py generate function
- [ ] Implement docker_reward.py reward function
- [ ] Test one full episode end-to-end: task → agent → bash → tests → reward
- [ ] Verify loss_mask alignment (1 for model outputs, 0 for observations)

### Week 3 — RL Training
- [ ] Wire custom functions into SLIME via CLI args
- [ ] Run 10-step smoke test, verify reward signal flows to GRPO
- [ ] Check GPU memory stays under 24GB with nvidia-smi
- [ ] Run 100-step training, observe reward curve
- [ ] Dynamic filtering: skip tasks with uniform rewards across all 8 samples

### Week 4 — Evaluation
- [ ] Download Terminal-Bench evaluation tasks (79 filtered tasks from SETA)
- [ ] Implement eval loop (Pass@8 metric)
- [ ] Compare baseline vs RL-trained Qwen3.5-2B
- [ ] Document results

---

## Key Implementation Notes

### Dynamic Filtering (SETA training trick)
Skip tasks where all 8 rollout samples have identical rewards — no learning signal:
```python
rewards = [s.reward for s in sample_group]
if len(set(rewards)) == 1:  # All same reward
    continue  # Skip this task's gradient update
```

### Retry Loop for Format Errors (SETA training trick)
If model outputs malformed tool call, inject error as observation and retry:
```python
if not cmd_match and not finish_match:
    error_obs = "<observation>\nerror: Invalid format. Use <bash>cmd</bash>.\n</observation>\n"
    # Add error as non-trainable observation, continue loop
```

### Container Resource Limits (prevent runaway tasks)
```python
container = client.containers.run(
    image_tag,
    command="sleep infinity",
    detach=True,
    mem_limit="4g",           # 4GB RAM per container
    cpu_period=100000,
    cpu_quota=200000,         # 2 CPU cores per container
    network_mode="bridge",
    remove=False,
)
```

---

## Reward Formula (from SETA paper)

```
u = unit test pass ratio (0.0 → 1.0)

R = u + 1.0   if u == 1.0   → reward = 2.0 (full success + bonus)
R = u         if u < 1.0    → reward = 0.0 to 1.0 (partial credit)

Why not binary 0/1:
Binary reward is too sparse — model gets no signal from near-misses.
Pass ratio gives credit for partially correct solutions,
enabling learning even when full task completion is rare for small models.
```

---

## References

- [SETA Blog](https://eigent-ai.notion.site/SETA-Scaling-Environments-for-Terminal-Agents-2d2511c70ba280a9b7c0fe3e7f1b6ab8)
- [SETA Repo](https://github.com/camel-ai/seta)
- [SETA Dataset](https://huggingface.co/datasets/camel-ai/seta-env)
- [SETA RL Model](https://huggingface.co/camel-ai/seta-rl-qwen3-8b)
- [SLIME Repo](https://github.com/THUDM/slime)
- [SLIME Single-GPU Issue #10](https://github.com/THUDM/slime/issues/10)
- [Terminal-Bench](https://www.tbench.ai/leaderboard)
- [terminal-bench-rl (community)](https://github.com/Danau5tin/terminal-bench-rl)
