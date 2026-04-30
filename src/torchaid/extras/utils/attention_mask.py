import torch

__all__ = ["make_attention_mask"]

def make_attention_mask(hidden_states: torch.Tensor, lengths: torch.Tensor):
    """Creates a boolean attention mask from per-sample sequence lengths.

    For each sample in the batch, positions up to (but not including) the
    corresponding length are marked as ``True`` (attend), and positions
    beyond are marked as ``False`` (ignore).

    Args:
        hidden_states (torch.Tensor): Tensor of shape ``(B, L, *)`` whose
            first two dimensions determine the batch size ``B`` and the
            maximum sequence length ``L``.
        lengths (torch.Tensor): 1-D integer tensor of shape ``(B,)`` containing
            the actual (unpadded) length of each sequence in the batch.

    Returns:
        torch.Tensor: Boolean mask of shape ``(B, L)`` on the same device as
            ``hidden_states``. Entry ``[b, i]`` is ``True`` iff
            ``i < lengths[b]``.
    """
    batch_size, max_length = hidden_states.size(0), hidden_states.size(1)
    range_tensor = torch.arange(max_length, device=hidden_states.device).repeat(batch_size, 1)
    attention_mask = torch.as_tensor(range_tensor < lengths.unsqueeze(1), device=hidden_states.device)

    return attention_mask
