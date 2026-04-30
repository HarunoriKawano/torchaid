from typing import Optional
import math

import torch
from torch import nn
from torch.nn.init import xavier_uniform_

from .positional_encoders import RelativePositionEmbedding

__all__ = ["Transformer", "TransformerWithRelativePosition"]

class Transformer(nn.Module):
    """Pre-norm Transformer encoder layer with multi-head self-attention and a feed-forward network.

    Implements the Pre-LN variant: layer normalisation is applied *before*
    each sub-layer (attention and FFN), followed by a residual connection.

    Attributes:
        layer_norm1 (nn.LayerNorm): Layer norm applied before the attention sub-layer.
        self_attn (MultiHeadSelfAttention): Multi-head self-attention module.
        layer_norm2 (nn.LayerNorm): Layer norm applied before the FFN sub-layer.
        ffn (FeedForward): Position-wise feed-forward network.
    """

    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            num_attention_heads: int,
            dropout_probability: float,
    ):
        """Initializes the Transformer encoder layer.

        Args:
            hidden_size (int): Dimensionality of the input and output hidden states.
            intermediate_size (int): Dimensionality of the inner FFN projection.
            num_attention_heads (int): Number of parallel attention heads.
                Must evenly divide ``hidden_size``.
            dropout_probability (float): Dropout probability applied in both
                the attention module and the FFN.
        """
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.self_attn = MultiHeadSelfAttention(
            hidden_size, num_attention_heads, dropout_probability
        )
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.ffn = FeedForward(hidden_size, intermediate_size, dropout_probability)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None
    ):
        """Applies one Transformer encoder layer to the input hidden states.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape ``(B, L, D)``
                where ``B`` is batch size, ``L`` is sequence length, and ``D``
                is ``hidden_size``.
            attention_mask (Optional[torch.Tensor]): Boolean or integer mask of
                shape ``(B, L)`` where ``0`` indicates positions to be masked.
                Defaults to ``None`` (no masking).

        Returns:
            torch.Tensor: Output tensor of shape ``(B, L, D)``.
        """
        residual = hidden_states.clone()
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states.clone()
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product multi-head self-attention module.

    Computes queries, keys, and values from the same input sequence and
    applies scaled dot-product attention independently across ``num_heads``
    heads. Xavier uniform initialisation is applied to Q/K/V projection weights.

    Attributes:
        head_size (int): Dimensionality per attention head (``hidden_size // num_heads``).
        num_heads (int): Number of attention heads.
        linear_q (nn.Linear): Query projection.
        linear_k (nn.Linear): Key projection.
        linear_v (nn.Linear): Value projection.
        dropout (nn.Dropout): Dropout applied to attention probabilities.
        linear_out (nn.Linear): Output projection.
    """

    def __init__(
            self,
            hidden_size: int,
            num_attention_heads: int,
            dropout_probability: float
    ):
        """Initializes the multi-head self-attention module.

        Args:
            hidden_size (int): Dimensionality of the input hidden states.
                Must be divisible by ``num_attention_heads``.
            num_attention_heads (int): Number of attention heads.
            dropout_probability (float): Dropout probability applied to
                attention weights.
        """
        super().__init__()
        self.head_size = hidden_size // num_attention_heads
        self.num_heads = num_attention_heads

        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(p=dropout_probability)
        self.linear_out = nn.Linear(hidden_size, hidden_size)

        xavier_uniform_(self.linear_q.weight)
        xavier_uniform_(self.linear_k.weight)
        xavier_uniform_(self.linear_v.weight)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Computes multi-head self-attention.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape ``(B, L, D)``.
            attention_mask (Optional[torch.Tensor]): Boolean mask of shape
                ``(B, L)``. Positions where the mask is ``0`` are filled with
                ``-inf`` before the softmax. Defaults to ``None``.

        Returns:
            torch.Tensor: Attended output tensor of shape ``(B, L, D)``.
        """
        batch_size, sequence_length, hidden_size = hidden_states.size()

        # `(B, L, D)` -> `(B, L, H, D/H)`
        query = self.linear_q(hidden_states).view(batch_size, -1, self.num_heads, self.head_size)
        key = self.linear_k(hidden_states).view(batch_size, -1, self.num_heads, self.head_size)
        value = self.linear_v(hidden_states).view(batch_size, -1, self.num_heads, self.head_size)

        # `(B, L, H, D/H)` -> `(B, L, H, D/H)`
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_size)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(attention_mask == 0, torch.finfo(scores.dtype).min)

        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        hidden_states = torch.matmul(probs, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_size)
        out = self.linear_out(hidden_states)

        return out

class TransformerWithRelativePosition(nn.Module):
    """Pre-norm Transformer encoder layer with relative-position-aware self-attention.

    Identical in structure to :class:`Transformer` but uses
    :class:`MultiHeadSelfAttentionWithRelativePosition` to incorporate
    relative position biases (Shaw et al., 2018) into the attention scores.

    Attributes:
        layer_norm1 (nn.LayerNorm): Layer norm applied before the attention sub-layer.
        self_attn (MultiHeadSelfAttentionWithRelativePosition): Relative-position-aware
            multi-head self-attention module.
        layer_norm2 (nn.LayerNorm): Layer norm applied before the FFN sub-layer.
        ffn (FeedForward): Position-wise feed-forward network.
    """

    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            num_attention_heads: int,
            dropout_probability: float,
            max_length: int,
            with_cls: bool = False
    ):
        """Initializes the relative-position Transformer encoder layer.

        Args:
            hidden_size (int): Dimensionality of hidden states.
            intermediate_size (int): Inner dimensionality of the FFN.
            num_attention_heads (int): Number of attention heads.
            dropout_probability (float): Dropout probability.
            max_length (int): Maximum sequence length used to build the relative
                position embedding table.
            with_cls (bool): If ``True``, the first token is treated as a CLS
                token and receives a dedicated position embedding. Defaults to
                ``False``.
        """
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.self_attn = MultiHeadSelfAttentionWithRelativePosition(
            hidden_size, num_attention_heads, dropout_probability, max_length, with_cls
        )
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.ffn = FeedForward(hidden_size, intermediate_size, dropout_probability)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
    ):
        """Applies one relative-position Transformer encoder layer.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape ``(B, L, D)``.
            attention_mask (Optional[torch.Tensor]): Boolean mask of shape
                ``(B, L)``. Defaults to ``None``.

        Returns:
            torch.Tensor: Output tensor of shape ``(B, L, D)``.
        """
        residual = hidden_states.clone()
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states.clone()
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class MultiHeadSelfAttentionWithRelativePosition(nn.Module):
    """Multi-head self-attention with relative position representations.

    Implements the relative position attention mechanism from
    *Self-Attention with Relative Position Representations* (Shaw et al., 2018).
    Attention scores are decomposed into content-based and position-based
    components using two learnable query bias vectors.

    Attributes:
        head_size (int): Dimensionality per attention head.
        num_heads (int): Number of attention heads.
        linear_q (nn.Linear): Query projection.
        linear_k (nn.Linear): Key projection.
        linear_v (nn.Linear): Value projection.
        dropout (nn.Dropout): Dropout on attention probabilities.
        linear_out (nn.Linear): Output projection.
        relative_position_k (RelativePositionEmbedding): Relative embeddings for keys.
        relative_position_v (RelativePositionEmbedding): Relative embeddings for values.
        query_bias1 (nn.Parameter): Learnable bias added to queries for content-to-position scores.
        query_bias2 (nn.Parameter): Learnable bias added to queries for position-to-position scores.
    """

    def __init__(
            self,
            hidden_size: int,
            num_attention_heads: int,
            dropout_probability: float,
            max_length: int,
            with_cls: bool = False
    ):
        """Initializes the relative-position multi-head self-attention module.

        Args:
            hidden_size (int): Dimensionality of hidden states.
                Must be divisible by ``num_attention_heads``.
            num_attention_heads (int): Number of attention heads.
            dropout_probability (float): Dropout probability.
            max_length (int): Maximum sequence length for the position table.
                The embedding table covers distances in
                ``[-max_length, max_length]``.
            with_cls (bool): If ``True``, the first token is treated as a CLS
                token with a dedicated position ID. Defaults to ``False``.
        """
        super().__init__()
        self.head_size = hidden_size // num_attention_heads
        self.num_heads = num_attention_heads

        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(p=dropout_probability)
        self.linear_out = nn.Linear(hidden_size, hidden_size)

        self.relative_position_k = RelativePositionEmbedding(self.head_size, max_length, with_cls)
        self.relative_position_v = RelativePositionEmbedding(self.head_size, max_length, with_cls)
        self.query_bias1 = nn.Parameter(torch.zeros(self.num_heads, self.head_size))
        self.query_bias2 = nn.Parameter(torch.zeros(self.num_heads, self.head_size))

        xavier_uniform_(self.linear_q.weight)
        xavier_uniform_(self.linear_k.weight)
        xavier_uniform_(self.linear_v.weight)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes relative-position multi-head self-attention.

        Attention scores are the sum of:

        1. Content-based scores: ``(query + bias1) @ key^T``
        2. Position-based scores: ``(query + bias2) @ relative_position_k^T``

        The output is likewise the sum of content-weighted and
        position-weighted value aggregations.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape ``(B, L, D)``.
            attention_mask (Optional[torch.Tensor]): Boolean mask of shape
                ``(B, L)``. Masked positions receive ``-inf`` before softmax.
                Defaults to ``None``.

        Returns:
            torch.Tensor: Attended output tensor of shape ``(B, L, D)``.
        """
        batch_size, length, hidden_size = hidden_states.size()

        # `(B, L, D)` -> `(B, L, H, D/H)`
        query = self.linear_q(hidden_states).view(batch_size, -1, self.num_heads, self.head_size)
        query1 = query + self.query_bias1
        query2 = query + self.query_bias2
        key = self.linear_k(hidden_states).view(batch_size, -1, self.num_heads, self.head_size)
        value = self.linear_v(hidden_states).view(batch_size, -1, self.num_heads, self.head_size)


        # `(B, L, H, D/H)` -> `(B, H, L, D/H)`
        query1 = query1.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        attention1 = torch.matmul(query1, key.transpose(-1, -2))

        # `(B, L, H, D/H)` -> `(L, B*H, D/H)`
        query2 = query2.transpose(0, 1).contiguous().view(length, -1, self.head_size)
        position_embeddings_k = self.relative_position_k(hidden_states)
        attention2 = torch.matmul(query2, position_embeddings_k.transpose(1, 2)).transpose(0, 1)
        attention2 = attention2.contiguous().view(batch_size, self.num_heads, length, length)
        attention = (attention1 + attention2) / math.sqrt(self.head_size)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            attention = attention.masked_fill(attention_mask == 0, torch.finfo(attention.dtype).min)

        probs = torch.softmax(attention, dim=-1)
        probs = self.dropout(probs)
        weight1 = torch.matmul(probs, value)

        position_embeddings_v = self.relative_position_v(hidden_states)
        weight2 = probs.permute(2, 0, 1, 3).contiguous().view(length, -1, length)
        weight2 = torch.matmul(weight2, position_embeddings_v)
        weight2 = weight2.transpose(0, 1).contiguous().view(batch_size, self.num_heads, -1, self.head_size)

        hidden_states = weight1 + weight2

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_size)
        out = self.linear_out(hidden_states)

        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network with SiLU activation and dropout.

    Applies layer normalisation, expands to ``intermediate_size``, applies
    SiLU activation and dropout, then projects back to ``hidden_size`` with
    another dropout.

    Attributes:
        layer_norm (nn.LayerNorm): Layer normalisation applied to the input.
        intermediate_dense (nn.Linear): Expansion projection.
        intermediate_act_fn (nn.SiLU): SiLU (Swish) activation function.
        intermediate_dropout (nn.Dropout): Dropout after activation.
        output_dense (nn.Linear): Contraction projection back to ``hidden_size``.
        output_dropout (nn.Dropout): Dropout on the output.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, dropout_probability: float):
        """Initializes the feed-forward network.

        Args:
            hidden_size (int): Input and output dimensionality.
            intermediate_size (int): Inner dimensionality of the expansion layer.
            dropout_probability (float): Dropout probability applied after
                activation and after the output projection.
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.intermediate_dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = nn.SiLU()
        self.intermediate_dropout = nn.Dropout(p=dropout_probability)

        self.output_dense = nn.Linear(intermediate_size, hidden_size)
        self.output_dropout = nn.Dropout(p=dropout_probability)

    def forward(self, hidden_states):
        """Applies the feed-forward transformation.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape ``(B, L, D)``.

        Returns:
            torch.Tensor: Output tensor of shape ``(B, L, D)``.
        """
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.intermediate_dropout(hidden_states)
        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states
