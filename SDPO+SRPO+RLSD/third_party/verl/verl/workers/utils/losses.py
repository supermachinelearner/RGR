
import torch
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import agg_loss, compute_value_loss, get_policy_loss_fn, kl_penalty
from verl.trainer.ppo.diffusion_algos import kl_penalty_image
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.metric import AggregationType, Metric
from verl.utils.torch_functional import masked_mean, masked_sum
from verl.workers.config import ActorConfig, CriticConfig
from verl.workers.utils.padding import no_padding_2_padding


def sft_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]

    log_prob = model_output["log_probs"]

    if pad_mode == DatasetPadMode.NO_PADDING:

        loss_mask = data["loss_mask"]

        log_prob_flatten = log_prob.values()
        loss_mask_flatten = loss_mask.values()

        loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)

        loss = -masked_sum(log_prob_flatten, loss_mask_flatten) / batch_num_tokens * dp_size
    else:
        response_mask = data["response_mask"].to(bool)
        loss = -masked_sum(log_prob, response_mask) / batch_num_tokens * dp_size

    return loss, {}


def ppo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)


    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    if (
        data["dp_size"] > 1
        or data["batch_num_tokens"] is not None
        or data["global_batch_size"] is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    metrics = {}

    # select fields and convert to padded tensor
    fields = ["response_mask", "old_log_probs", "advantages"]
    if "rollout_is_weights" in data:
        fields.append("rollout_is_weights")
    if "ref_log_prob" in data:
        fields.append("ref_log_prob")
    if "teacher_entropy" in data:
        fields.append("teacher_entropy")
    if "teacher_log_probs" in data:
        fields.append("teacher_log_probs")
    if "teacher_wrong_log_probs" in data:
        fields.append("teacher_wrong_log_probs")
    if "teacher_wrong_entropy" in data:
        fields.append("teacher_wrong_entropy")
    for key in (
        "teacher_correct_multi_log_probs",
        "teacher_wrong_multi_log_probs",
        "teacher_correct_multi_valid_mask",
        "teacher_wrong_multi_valid_mask",
        "teacher_correct_multi_entropy",
        "teacher_wrong_multi_entropy"):
        if key in data:
            fields.append(key)
    data = data.select(*fields).to_padded_tensor()

    response_mask = data["response_mask"].to(bool)
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]
    rollout_is_weights = data.get("rollout_is_weights", None)

    loss_agg_mode = config.loss_agg_mode

    loss_mode = config.policy_loss.get("loss_mode", "vanilla")
    teacher_log_probs = None
    teacher_entropy_vals = None
    if loss_mode in ("opsd", "sdpo", "rlsd", "srpo"):
        if "teacher_log_probs" in data:
            teacher_log_probs = data["teacher_log_probs"]
        if "teacher_entropy" in data:
            teacher_entropy_vals = data["teacher_entropy"]
    teacher_wrong_log_probs = None
    teacher_wrong_entropy_vals = None
    teacher_correct_multi_log_probs = None
    teacher_wrong_multi_log_probs = None
    teacher_correct_multi_valid_mask = None
    teacher_wrong_multi_valid_mask = None
    teacher_correct_multi_entropy_vals = None
    teacher_wrong_multi_entropy_vals = None
    if loss_mode == "rlsd_ectr":
        if "teacher_log_probs" in data:
            teacher_log_probs = data["teacher_log_probs"]
        if "teacher_wrong_log_probs" in data:
            teacher_wrong_log_probs = data["teacher_wrong_log_probs"]
        if "teacher_entropy" in data:
            teacher_entropy_vals = data["teacher_entropy"]
        if "teacher_wrong_entropy" in data:
            teacher_wrong_entropy_vals = data["teacher_wrong_entropy"]
    elif loss_mode == "rlcsd":
        if "teacher_log_probs" in data:
            teacher_log_probs = data["teacher_log_probs"]
        if "teacher_entropy" in data:
            teacher_entropy_vals = data["teacher_entropy"]
        if "teacher_wrong_multi_log_probs" in data:
            teacher_wrong_multi_log_probs = data["teacher_wrong_multi_log_probs"]
        if "teacher_wrong_multi_valid_mask" in data:
            teacher_wrong_multi_valid_mask = data["teacher_wrong_multi_valid_mask"]
        if "teacher_wrong_multi_entropy" in data:
            teacher_wrong_multi_entropy_vals = data["teacher_wrong_multi_entropy"]

    policy_loss_fn = get_policy_loss_fn(loss_mode)
    loss_kwargs = dict(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights)
    if loss_mode in ("opsd", "sdpo", "rlsd", "srpo"):
        loss_kwargs["teacher_log_probs"] = teacher_log_probs
        loss_kwargs["teacher_entropy"] = teacher_entropy_vals
    elif loss_mode == "rlsd_ectr":
        loss_kwargs["teacher_log_probs"] = teacher_log_probs
        loss_kwargs["teacher_wrong_log_probs"] = teacher_wrong_log_probs
        loss_kwargs["teacher_entropy"] = teacher_entropy_vals
        loss_kwargs["teacher_wrong_entropy"] = teacher_wrong_entropy_vals
    elif loss_mode == "rlcsd":
        loss_kwargs["teacher_log_probs"] = teacher_log_probs
        loss_kwargs["teacher_entropy"] = teacher_entropy_vals
        loss_kwargs["teacher_wrong_multi_log_probs"] = teacher_wrong_multi_log_probs
        loss_kwargs["teacher_wrong_multi_valid_mask"] = teacher_wrong_multi_valid_mask
        loss_kwargs["teacher_wrong_multi_entropy"] = teacher_wrong_multi_entropy_vals
    pg_loss, pg_metrics = policy_loss_fn(**loss_kwargs)

    pg_metrics = Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN)

    metrics.update(pg_metrics)
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=metric_aggregation)
    policy_loss = pg_loss

    if entropy is not None:
        entropy_loss = agg_loss(
            loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
        )
        entropy_coeff = config.entropy_coeff
        policy_loss -= entropy_coeff * entropy_loss
        metrics["actor/entropy_loss"] = Metric(value=entropy_loss, aggregation=metric_aggregation)

    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]
        # compute kl loss
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        kl_loss = agg_loss(
            loss_mat=kld, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode, **config.global_batch_info
        )

        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
        metrics["kl_coef"] = config.kl_loss_coef

    return policy_loss, metrics


def value_loss(config: CriticConfig, model_output, data: TensorDict, dp_group=None):

    vpreds = no_padding_2_padding(model_output["values"], data)  # (bsz, response_length)
    data = data.select("values", "returns", "response_mask").to_padded_tensor()
    values = data["values"]
    returns = data["returns"]
    response_mask = data["response_mask"].to(bool)

    vf_loss, vf_clipfrac = compute_value_loss(
        vpreds=vpreds,
        values=values,
        returns=returns,
        response_mask=response_mask,
        cliprange_value=config.cliprange_value,
        loss_agg_mode=config.loss_agg_mode)

    metrics = {}

    metrics.update(
        {
            "critic/vf_loss": vf_loss.detach().item(),
            "critic/vf_clipfrac": vf_clipfrac.detach().item(),
            "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item()}
    )

    return vf_loss, metrics


def diffusion_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    log_prob = model_output["log_probs"]

    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    metrics = {}

    response_mask = data["response_mask"].to(bool)
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]

    loss_agg_mode = config.loss_agg_mode

    loss_mode = config.policy_loss.get("loss_mode", "flow_grpo")

    policy_loss_fn = get_policy_loss_fn(loss_mode)
    pg_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=None)

    pg_metrics = Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN)

    metrics.update(pg_metrics)
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=AggregationType.MEAN)
    policy_loss = pg_loss

    if config.use_kl_loss:
        ref_prev_sample_mean = data["ref_prev_sample_mean"]
        prev_sample_mean = model_output["prev_sample_mean"]
        std_dev_t = model_output["std_dev_t"]
        kl_loss = kl_penalty_image(
            prev_sample_mean=prev_sample_mean, ref_prev_sample_mean=ref_prev_sample_mean, std_dev_t=std_dev_t
        )

        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=AggregationType.MEAN)
        metrics["kl_coef"] = config.kl_loss_coef

    gradient_accumulation_steps = tu.get_non_tensor_data(data, "gradient_accumulation_steps", default=None)
    policy_loss = policy_loss / gradient_accumulation_steps

    return policy_loss, metrics
