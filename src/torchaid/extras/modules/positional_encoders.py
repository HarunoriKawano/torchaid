import torch
from torch import nn

__all__ = ["PositionalEmbedding", "RelativePositionEmbedding"]

class PositionalEmbedding(nn.Module):
    """Absolute positional embedding that adds learnable position vectors to hidden states.

    Generates a position embedding for each position in the sequence and
    returns it as a broadcastable tensor, ready to be added to the input.

    Attributes:
        position_encoder (nn.Embedding): Embedding table of shape
            ``(max_length, hidden_size)``.
    """

    def __init__(self, hidden_size: int, max_length: int):
        """Initializes the positional embedding table.

        Args:
            hidden_size (int): Dimensionality of each position embedding vector.
            max_length (int): Maximum sequence length supported.
        """
        super().__init__()
        self.position_encoder = nn.Embedding(max_length, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Computes positional embeddings for the current sequence length.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape ``(B, L, D)``.
                Only the sequence length ``L`` is used; the values are not read.

        Returns:
            torch.Tensor: Position embedding tensor of shape ``(1, L, D)``
                intended to be added to the input hidden states.
        """
        max_length = hidden_states.size(1)
        position_ids = torch.arange(0, max_length, 1).to(hidden_states.device)
        position_embeddings = self.position_encoder(position_ids).unsqueeze(0)

        return position_embeddings


class RelativePositionEmbedding(nn.Module):
    """Learnable relative position embedding for attention score bias.

    Builds a pairwise distance matrix and looks up embeddings for each
    ``(query_position, key_position)`` pair. Optionally handles a CLS token
    at position 0 with a dedicated embedding index.

    Attributes:
        max_length (int): Maximum sequence length (excluding CLS if applicable).
        positional_embedding (nn.Embedding): Embedding table of size
            ``(2 * max_length + 1, hidden_size)`` covering distances in
            ``[-max_length, max_length]`` plus one extra index for the CLS token.
        with_cls (bool): Whether the first token is a CLS token.
        cls_id (int): Embedding index reserved for CLS-related positions
            (``2 * max_length``).
    """

    def __init__(self, hidden_size: int, max_length: int, with_cls: bool = True):
        """Initializes the relative position embedding table.

        Args:
            hidden_size (int): Dimensionality of each position embedding vector.
            max_length (int): Maximum sequence length. The distance table covers
                ``[-max_length, max_length]``, encoded as indices
                ``[0, 2 * max_length]``.
            with_cls (bool): If ``True``, the first token is treated as a CLS
                token. Its row and column in the distance matrix are filled with
                a dedicated ``cls_id`` index. Defaults to ``True``.
        """
        super().__init__()
        self.max_length = max_length
        self.positional_embedding = nn.Embedding(self.max_length * 2 + 1, hidden_size)
        self.with_cls = with_cls
        self.cls_id = self.max_length * 2

    def forward(self, hidden_states: torch.Tensor):
        """Computes the relative position embedding matrix.

        Constructs an ``(L, L)`` distance matrix where entry ``[i, j]`` is
        ``j - i + max_length``, then looks up the embedding for each entry.
        When ``with_cls=True`` the CLS row and column receive the ``cls_id``
        embedding.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape ``(B, L, D)``.
                Only the sequence length is used; values are not read.

        Returns:
            torch.Tensor: Relative position embeddings of shape ``(L, L, D)``.
        """
        if self.with_cls:
            hidden_states = hidden_states[:, 1:]
        range_tensor = torch.arange(hidden_states.size(1), device=hidden_states.device)
        distance_mat = range_tensor[None, :] - range_tensor[:, None] + self.max_length


        if self.with_cls:
            distance_mat = torch.cat([torch.zeros_like(distance_mat[None, 0, :]) + self.cls_id, distance_mat], dim=0)
            distance_mat = torch.cat([torch.zeros_like(distance_mat[:, None, 0]) + self.cls_id, distance_mat], dim=1)

        position_embeddings = self.positional_embedding(distance_mat)
        return position_embeddings
