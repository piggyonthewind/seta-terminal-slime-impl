import asyncio
import re
import json
import logging
import threading
from jinja2 import Template
from camel.toolkits.terminal_toolkit import TerminalToolkit
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

_build_locks: dict[str, threading.Lock] = {}
_build_locks_mutex = threading.Lock()
_logged_full_rollout = False

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a terminal agent. Solve the given task by using the provided "
    "tools to execute shell commands in an Ubuntu environment."
)

TOOL_TEMPLATE = """<|im_start|>system
{%- if messages[0]['role'] == 'system' %}
{{- messages[0]['content'] }}
{%- else %}
You are a helpful assistant.
{%- endif %}
{%- if tools %}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{%- for tool in tools %}
{{- tool | tojson }}
{%- endfor %}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{%- endif %}
<|im_end|>
{%- for message in messages %}
{%- if message['role'] == 'user' %}
<|im_start|>user
{{- message['content'] }}<|im_end|>
{%- elif message['role'] == 'assistant' %}
<|im_start|>assistant
{{- message['content'] }}<|im_end|>
{%- endif %}
{%- endfor %}
<|im_start|>assistant
"""

async def generate(args, sample: Sample, sampling_params:dict) -> Sample:
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    task_path = sample.metadata.get("task_path")

    loop = asyncio.get_event_loop()
    try:
        container = await loop.run_in_executor(None, lambda: _start_container(task_path))
    except Exception as e:
        logger.warning("Failed to start container for %s: %s", task_path, e)
        sample.status = Sample.Status.ABORTED
        sample.metadata["pass_ratio"] = 0.0
        sample.reward = {"score": 0.0}
        sample.remove_sample = True
        sample.tokens = state.tokenizer(sample.prompt, add_special_tokens=False)["input_ids"]
        sample.response_length = 0
        sample.response = ""
        sample.loss_mask = []
        sample.rollout_log_probs = []
        return sample
    toolkit = TerminalToolkit(use_docker_backend=True, docker_container_name=container.id, safe_mode=False)

    # Get tool schemas and build prompt
    tool_schemas = [t.get_openai_tool_schema() for t in toolkit.get_tools()]
    prompt = Template(TOOL_TEMPLATE).render(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sample.prompt},
        ],
        tools=tool_schemas,
    )
    prompt_token_ids = state.tokenizer(prompt, add_special_tokens=False)["input_ids"]

    # Pre-tokenize chat turn boundary markers (Qwen3.5 format)
    im_end_ids = state.tokenizer("<|im_end|>", add_special_tokens=False)["input_ids"]
    im_end_id = im_end_ids[0] if im_end_ids else None
    boundary_before_obs = state.tokenizer("\n<|im_start|>user\n", add_special_tokens=False)["input_ids"]
    boundary_after_obs = state.tokenizer("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]

    def _append_non_trainable(token_ids):
        """Append tokens with loss_mask=0 (observations, turn boundaries)."""
        response_token_ids.extend(token_ids)
        loss_masks.extend([0] * len(token_ids))
        rollout_log_probs.extend([0.0] * len(token_ids))

    def _append_observation(observation_text):
        """Wrap observation in proper Qwen3.5 chat turn boundaries and append."""
        # Ensure previous model output ends with <|im_end|>
        if im_end_id is not None and response_token_ids and response_token_ids[-1] != im_end_id:
            _append_non_trainable(im_end_ids)
        # <|im_end|>\n<|im_start|>user\n
        _append_non_trainable(boundary_before_obs)
        # observation content
        obs_ids = state.tokenizer(observation_text, add_special_tokens=False)["input_ids"]
        _append_non_trainable(obs_ids)
        # <|im_end|>\n<|im_start|>assistant\n
        _append_non_trainable(boundary_after_obs)

    # Initialize token tracking
    response_token_ids = []
    loss_masks = []
    rollout_log_probs = []

    max_turns = getattr(args, "max_agent_turns", 20)
    max_ctx = getattr(args, "rollout_max_context_len", 8192)

    try:
        for turn in range(max_turns):
            # Check context budget
            if len(prompt_token_ids) + len(response_token_ids) >= max_ctx:
                sample.status = Sample.Status.TRUNCATED
                break

            # Step 1: POST current context to SGLang, get model output
            payload = {
                "input_ids": prompt_token_ids + response_token_ids,
                "sampling_params": sampling_params,
                "return_logprob": True,
            }
            output = await post(url, payload)

            finish_reason = output["meta_info"]["finish_reason"]["type"]

            if finish_reason == "abort":
                current_len = len(prompt_token_ids) + len(response_token_ids)
                near_limit = current_len >= max_ctx * 0.9
                sample.status = Sample.Status.TRUNCATED if near_limit else Sample.Status.ABORTED
                break

            # Extract token ids and logprobs from SGLang response
            if "output_token_logprobs" in output["meta_info"]:
                token_lp_pairs = output["meta_info"]["output_token_logprobs"]
                turn_token_ids = [x[1] for x in token_lp_pairs]
                turn_logprobs = [x[0] for x in token_lp_pairs]
            else:
                model_text = output["text"]
                turn_token_ids = state.tokenizer(model_text, add_special_tokens=False)["input_ids"]
                turn_logprobs = [0.0] * len(turn_token_ids)

            model_text = state.tokenizer.decode(turn_token_ids)

            # Step 2: Append model output — trainable (loss_mask=1)
            response_token_ids.extend(turn_token_ids)
            loss_masks.extend([1] * len(turn_token_ids))
            rollout_log_probs.extend(turn_logprobs)

            if finish_reason == "length":
                sample.status = Sample.Status.TRUNCATED
                break

            # Step 3: Parse tool call from model output
            tool_call_match = re.search(
                r"<tool_call>\s*(\{.*?\})\s*</tool_call>", model_text, re.DOTALL
            )

            if not tool_call_match:
                # No tool call — model may think it's done, or forgot the format.
                # If it's the last turn, end cleanly; otherwise nudge it to retry.
                if turn == max_turns - 1:
                    sample.status = Sample.Status.COMPLETED
                    break
                observation = (
                    "<tool_response>\nerror: no tool call found in your response. "
                    "Please use the <tool_call>{...}</tool_call> format to call a tool, "
                    "or finish your answer if the task is complete.\n</tool_response>"
                )
                _append_observation(observation)
                continue

            try:
                tool_call = json.loads(tool_call_match.group(1))
                tool_name = tool_call["name"]
                tool_args = tool_call.get("arguments", {})
            except (json.JSONDecodeError, KeyError) as e:
                # Malformed JSON — feed error back and let the model retry
                observation = (
                    f"<tool_response>\nerror: could not parse tool call JSON: {e}. "
                    f"Please ensure your tool call is valid JSON inside <tool_call></tool_call> tags.\n</tool_response>"
                )
                _append_observation(observation)
                continue

            # Step 4: Execute tool via CAMEL toolkit
            tool_method = getattr(toolkit, tool_name, None)
            if tool_method is None:
                observation = f"<tool_response>\nerror: unknown tool '{tool_name}'\n</tool_response>"
            else:
                try:
                    result = await loop.run_in_executor(None, lambda: tool_method(**tool_args))
                    observation = f"<tool_response>\n{result}\n</tool_response>"
                except Exception as e:
                    observation = (
                        f"<tool_response>\nerror calling {tool_name}: {e}\n"
                        f"</tool_response>"
                    )

            # Step 5: Append observation with proper chat turn boundaries
            _append_observation(observation)

        # Run tests and record pass ratio.
        # Skip if the episode was cut short — the agent didn't finish the task.
        incomplete = sample.status in (Sample.Status.TRUNCATED, Sample.Status.ABORTED)
        if incomplete:
            sample.metadata["pass_ratio"] = 0.0
        else:
            pass_ratio = await loop.run_in_executor(
                None, lambda: _run_tests(container, task_path)
            )
            sample.metadata["pass_ratio"] = pass_ratio

    finally:
        # Always clean up the container
        try:
            container.stop(timeout=5)
            container.remove()
        except Exception as e:
            logger.warning(f"Container cleanup failed: {e}")

    # Finalize sample
    sample.tokens = prompt_token_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = state.tokenizer.decode(response_token_ids)
    sample.loss_mask = loss_masks
    sample.rollout_log_probs = rollout_log_probs

    if sample.status == Sample.Status.PENDING:
        sample.status = Sample.Status.COMPLETED

    sample.metadata["turn_count"] = turn + 1 if 'turn' in dir() else 0
    logger.info(
        "episode done | task=%s | status=%s | turns=%d | pass_ratio=%.3f | response_len=%d",
        task_path, sample.status, sample.metadata.get("turn_count", 0),
        sample.metadata.get("pass_ratio", 0.0), sample.response_length,
    )

    # Log one full rollout for debugging token/mask alignment
    global _logged_full_rollout
    if not _logged_full_rollout:
        _logged_full_rollout = True
        full_text = state.tokenizer.decode(sample.tokens)
        # Build annotated view: mark each token with its loss_mask value
        prompt_len = len(prompt_token_ids)
        all_ids = sample.tokens
        masks = [0] * prompt_len + loss_masks
        annotated_lines = []
        # Walk through tokens and show boundaries where mask changes
        current_mask = None
        current_chunk = []
        for i, (tid, m) in enumerate(zip(all_ids, masks)):
            if m != current_mask:
                if current_chunk:
                    chunk_text = state.tokenizer.decode(current_chunk)
                    annotated_lines.append(f"[mask={current_mask}] {chunk_text}")
                current_mask = m
                current_chunk = []
            current_chunk.append(tid)
        if current_chunk:
            chunk_text = state.tokenizer.decode(current_chunk)
            annotated_lines.append(f"[mask={current_mask}] {chunk_text}")
        annotated = "\n---\n".join(annotated_lines)
        logger.info(
            "\n========== FULL ROLLOUT DEBUG (first episode) ==========\n"
            "task: %s | tokens: %d | prompt: %d | response: %d | masks: %d\n"
            "--- ANNOTATED (mask=0: non-trainable, mask=1: trainable) ---\n%s\n"
            "========== END ROLLOUT DEBUG ==========",
            task_path, len(all_ids), prompt_len, len(response_token_ids),
            len(masks), annotated,
        )

    return sample


def _start_container(task_path: str):
    import docker
    from docker.errors import BuildError, ImageNotFound

    client = docker.from_env()
    tag = "seta-task:" + re.sub(r"[^a-z0-9]", "-", task_path.lower().strip("/"))

    # Ensure only one thread builds a given image; others wait and reuse it.
    with _build_locks_mutex:
        if tag not in _build_locks:
            _build_locks[tag] = threading.Lock()
    lock = _build_locks[tag]

    with lock:
        try:
            client.images.get(tag)
        except ImageNotFound:
            logger.info("Building Docker image %s from %s", tag, task_path)
            try:
                # Consume the generator so the build runs to completion.
                _, logs = client.images.build(path=task_path, tag=tag, rm=True)
                for chunk in logs:
                    if "stream" in chunk:
                        logger.debug("docker build: %s", chunk["stream"].rstrip())
                    elif "error" in chunk:
                        raise BuildError(chunk["error"], logs)
            except BuildError as e:
                raise RuntimeError(f"Docker build failed for {tag}: {e}") from e
            # Verify the image actually exists after the build
            client.images.get(tag)

    return client.containers.run(
        tag,
        command="sleep infinity",
        detach=True,
        remove=False,
        mem_limit="4g",
        cpu_quota=200000,
        cpu_period=100000,
        network_mode="bridge",
    )


def _run_tests(container, task_path: str) -> float:
    """Run the task's test suite inside the container and return pass ratio in [0, 1]."""
    test_script = "/run-tests.sh"

    # Copy run-tests.sh and tests/ into the container
    import tarfile, io, os

    def _copy_path_to_container(host_path: str, container_dest: str):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            tar.add(host_path, arcname=os.path.basename(host_path))
        buf.seek(0)
        container.put_archive(os.path.dirname(container_dest), buf)

    _copy_path_to_container(
        os.path.join(task_path, "run-tests.sh"), test_script
    )
    _copy_path_to_container(
        os.path.join(task_path, "tests"), "/tests"
    )

    # Make run-tests.sh executable and run it from a clean temp dir
    # to avoid conflicts with any existing pyproject.toml in the task WORKDIR.
    container.exec_run("chmod +x /run-tests.sh")
    container.exec_run("mkdir -p /tmp/test-runner")
    _, output = container.exec_run(
        "bash /run-tests.sh",
        environment={"TEST_DIR": "/tests", "HOME": "/root"},
        workdir="/tmp/test-runner",
        user="root",
    )
    stdout = output.decode("utf-8", errors="replace")
    logger.debug("Test output:\n%s", stdout)

    ratio = _parse_pytest_ratio(stdout)
    if ratio == 0.0:
        logger.info("test output tail [%s]: %s", task_path, stdout[-300:].replace('\n', ' | '))
    return ratio


def _parse_pytest_ratio(output: str) -> float:
    """Parse pytest short summary to extract passed/total ratio."""
    # pytest summary line looks like: "5 passed, 2 failed, 1 error in 3.45s"
    # or "3 passed in 1.23s"
    import re as _re
    passed = failed = error = 0

    m = _re.search(r"(\d+) passed", output)
    if m:
        passed = int(m.group(1))
    m = _re.search(r"(\d+) failed", output)
    if m:
        failed = int(m.group(1))
    m = _re.search(r"(\d+) error", output)
    if m:
        error = int(m.group(1))

    total = passed + failed + error
    if total == 0:
        if "no tests ran" in output or "collected 0 items" in output:
            logger.debug("No tests collected, pass_ratio=0")
        else:
            logger.warning("Could not parse pytest output, defaulting pass_ratio=0. Output tail: %s", output[-200:])
        return 0.0
    return passed / total