import logging
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


async def reward_func(args, sample: Sample, **kwargs) -> dict:
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    if sample.status == Sample.Status.ABORTED:
        # Infrastructure failure — zero out the loss mask so this sample
        # contributes no gradient to training
        sample.remove_sample = True
        return {"score": 0.0}

    pass_ratio = sample.metadata.get("pass_ratio", 0.0)
    completion_bonus = 1.0 if pass_ratio == 1.0 else 0.0
    score = pass_ratio + completion_bonus

    logger.debug("pass_ratio=%.3f bonus=%.1f score=%.3f status=%s", pass_ratio, completion_bonus, score, sample.status)

    return {"score": score}
