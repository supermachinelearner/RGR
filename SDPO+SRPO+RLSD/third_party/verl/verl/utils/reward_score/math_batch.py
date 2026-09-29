
from .math_reward import compute_score


def compute_score_batched(data_sources, solution_strs, ground_truths, extra_infos):
    """
    This is a demonstration of how the batched reward function should look like.
    Typically, you want to use batched reward to speed up the process with parallelization
    """
    return [
        compute_score(solution_str, ground_truth)
        for solution_str, ground_truth in zip(solution_strs, ground_truths, strict=True)
    ]
