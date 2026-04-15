"""
Custom wandb rollout logging for SETA terminal agent training.

Wired via: --custom-rollout-log-function-path seta.log_rollout_metrics.log_rollout_metrics

Returns False so slime's default rollout logging (response length, perf, etc.) still runs.
"""

import logging
import numpy as np

from slime.utils import logging_utils
from slime.utils.metric_utils import compute_rollout_step, dict_add_prefix, compute_statistics
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def log_rollout_metrics(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    step = compute_rollout_step(args, rollout_id)

    pass_ratios = [s.metadata.get("pass_ratio", 0.0) for s in samples]
    scores = [s.metadata.get("score", s.metadata.get("pass_ratio", 0.0)) for s in samples]

    statuses = [s.status for s in samples]
    n = len(samples)

    log_dict = {}

    # Pass ratio distribution across the rollout batch
    log_dict |= dict_add_prefix(compute_statistics(pass_ratios), "seta/pass_ratio/")

    # Completion rate: fraction of episodes where agent passed all tests
    log_dict["seta/full_success_rate"] = float(np.mean([r == 1.0 for r in pass_ratios]))

    # Any success: fraction where at least one test passed
    log_dict["seta/any_success_rate"] = float(np.mean([r > 0.0 for r in pass_ratios]))

    # Episode status breakdown
    log_dict["seta/status/completed"] = float(np.mean([s == Sample.Status.COMPLETED for s in statuses]))
    log_dict["seta/status/truncated"] = float(np.mean([s == Sample.Status.TRUNCATED for s in statuses]))
    log_dict["seta/status/aborted"]   = float(np.mean([s == Sample.Status.ABORTED   for s in statuses]))

    # Turn count per episode (proxy for task difficulty / agent efficiency)
    turn_counts = []
    for s in samples:
        turns = s.metadata.get("turn_count", None)
        if turns is not None:
            turn_counts.append(turns)
    if turn_counts:
        log_dict |= dict_add_prefix(compute_statistics(turn_counts), "seta/turns/")

    log_dict["rollout/step"] = step
    logging_utils.log(args, log_dict, step_key="rollout/step")

    logger.info(
        "rollout %d | pass_ratio mean=%.3f | full_success=%.1f%% | n=%d",
        rollout_id,
        float(np.mean(pass_ratios)),
        log_dict["seta/full_success_rate"] * 100,
        n,
    )

    # Return False → slime's default logging (response length, perf, etc.) still runs
    return False
