
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.attention_utils import index_first_axis, unpad_input


def left_right_2_no_padding(data: TensorDict) -> TensorDict:
    assert "input_ids" in data, "input_ids is required in left-right padding data"
    assert "attention_mask" in data, "attention_mask is required in left-right padding data"
    assert "response_mask" in data, "response_mask is required in left-right padding data"
    assert "position_ids" in data, "position_ids is required in left-right padding data"

    input_ids = data.pop("input_ids")
    attention_mask = data["attention_mask"]
    response_mask = data["response_mask"]
    position_ids = data["position_ids"]  # (bs, seq_len) or # (bs, 4, seq_len)

    max_seq_len, max_response_len = input_ids.shape[1], response_mask.shape[1]
    tu.assign_non_tensor_data(data, "max_seq_len", max_seq_len)
    tu.assign_non_tensor_data(data, "max_response_len", max_response_len)

    input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
    tu.assign_non_tensor_data(data, "indices", indices)

    input_ids_nested = torch.nested.nested_tensor_from_jagged(input_ids_rmpad.squeeze(-1), offsets=cu_seqlens)

    position_ids_list = []
    for i in range(attention_mask.shape[0]):
        curr_mask = attention_mask[i].bool()
        curr_pos_ids = position_ids[i]
        if curr_pos_ids.dim() == 1:  # (seq_len,)
            valid_ids = curr_pos_ids[curr_mask]
        else:  # (4, seq_len)
            valid_ids = curr_pos_ids[:, curr_mask]
        position_ids_list.append(valid_ids)
    position_ids_nested = torch.nested.as_nested_tensor(position_ids_list, layout=torch.jagged)

    data["input_ids"] = input_ids_nested
    data["position_ids"] = position_ids_nested
    data["loss_mask"] = data["response_mask"]

    routed_experts = data.get("routed_experts", None)
    if routed_experts is not None and not routed_experts.is_nested:
        if routed_experts.max() <= 255:
            routed_experts = routed_experts.to(torch.uint8)
        routed_experts_rmpad = index_first_axis(routed_experts.unsqueeze(-1).flatten(0, 1), indices)
        routed_experts_nested = torch.nested.nested_tensor_from_jagged(
            routed_experts_rmpad.squeeze(-1), offsets=cu_seqlens
        )
        data["routed_experts"] = routed_experts_nested

    # (bsz, seqlen, topk)
    teacher_logprobs = data.get("teacher_logprobs", None)
    teacher_ids = data.get("teacher_ids", None)
    if teacher_logprobs is not None and teacher_ids is not None:
        teacher_logprobs_rmpad = index_first_axis(teacher_logprobs.unsqueeze(-1).flatten(0, 1), indices)
        teacher_ids_rmpad = index_first_axis(teacher_ids.unsqueeze(-1).flatten(0, 1), indices)
        teacher_logprobs_nested = torch.nested.nested_tensor_from_jagged(
            teacher_logprobs_rmpad.squeeze(-1), offsets=cu_seqlens
        )
        teacher_ids_nested = torch.nested.nested_tensor_from_jagged(teacher_ids_rmpad.squeeze(-1), offsets=cu_seqlens)
        data["teacher_logprobs"] = teacher_logprobs_nested
        data["teacher_ids"] = teacher_ids_nested

    return data


def no_padding_2_padding(tensor: torch.Tensor, data: TensorDict) -> torch.Tensor:
    values = tensor.values() if tensor.is_nested else tensor
    prompt_ids = data["prompts"]
    response_ids = data["responses"]

    max_response_len = tu.get_non_tensor_data(data=data, key="max_response_len", default=-1)

    if prompt_ids.is_nested:
        prompt_lens = prompt_ids.offsets().diff()
        response_lens = response_ids.offsets().diff()
        if max_response_len < 0:
            max_response_len = response_lens.max().item()
    else:
        attention_mask = data["attention_mask"]
        assert not attention_mask.is_nested
        prompt_lens = attention_mask[:, : prompt_ids.shape[1]].sum(dim=1)
        response_lens = attention_mask[:, prompt_ids.shape[1] :].sum(dim=1)
        max_response_len = response_ids.shape[1]

    sequence_lens = prompt_lens + response_lens
    sequence_offsets = sequence_lens.cumsum(dim=0)
    assert sequence_offsets[-1].item() == values.shape[0]
    assert not prompt_lens.eq(0).any(), f"seq_offset - resp_len - 1 assumes prompt_len > 0. Got {prompt_lens}"

    response_list = []
    skip_padding = (0, 0) * (values.ndim - 1)
    for resp_len, seq_offset in zip(response_lens, sequence_offsets, strict=True):
        pad_size = max_response_len - resp_len
        response_list.append(F.pad(values[seq_offset - resp_len - 1 : seq_offset - 1], (*skip_padding, 0, pad_size)))

    output = torch.stack(response_list, dim=0)
    return output


def embeds_padding_2_no_padding(data: TensorDict) -> TensorDict:

    def _to_nested(embeds: torch.Tensor, mask: torch.Tensor):
        embeds_list, mask_list = [], []
        for i in range(mask.shape[0]):
            curr_mask = mask[i].bool()
            embeds_list.append(embeds[i, curr_mask, :])
            mask_list.append(curr_mask[curr_mask])
        return (
            torch.nested.as_nested_tensor(embeds_list, layout=torch.jagged),
            torch.nested.as_nested_tensor(mask_list, layout=torch.jagged),
        )

    data["prompt_embeds"], data["prompt_embeds_mask"] = _to_nested(data["prompt_embeds"], data["prompt_embeds_mask"])

    if isinstance(data.get("negative_prompt_embeds", None), torch.Tensor):
        data["negative_prompt_embeds"], data["negative_prompt_embeds_mask"] = _to_nested(
            data["negative_prompt_embeds"], data["negative_prompt_embeds_mask"]
        )

    return data


def response_from_nested(tensor: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    values, offsets = tensor.values(), tensor.offsets()
    response_lens = response_mask.offsets().diff()
    response_list = []
    for resp_len, seq_offset in zip(response_lens, offsets[1:], strict=True):
        # left-shift model output by one token for log_probs/values
        response_list.append(values[seq_offset - resp_len - 1 : seq_offset - 1])
    return torch.nested.as_nested_tensor(response_list, layout=torch.jagged)


def response_to_nested(tensor: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    assert response_mask.is_nested
    response_lens = response_mask.offsets().diff()
    response_list = []
    for i in range(tensor.shape[0]):
        response_list.append(tensor[i, : response_lens[i]])

    return torch.nested.as_nested_tensor(response_list, layout=torch.jagged)
