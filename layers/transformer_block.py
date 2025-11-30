from typing import TypeAlias, Union
from layers.sliding_window_attn import SlidingWindowAttention
from layers.sparse_expert import SparseExpertLayer
from layers.standard_layers import MLP, MultiHeadAttention
import torch  # type: ignore
from torch import nn  # type: ignore

from layers.layer_config import AttentionConfig, AttentionType, BlockConfig, MLPConfig, MLPType

AttentionLayer: TypeAlias = Union[
    MultiHeadAttention, SlidingWindowAttention
]

MLPLayer: TypeAlias = Union[MLP, SparseExpertLayer]

def build_attn_layer(attn_config: AttentionConfig) -> AttentionLayer:
    if attn_config.attention_type == AttentionType.MHA:
        return MultiHeadAttention(attn_config)
    elif attn_config.attention_type == AttentionType.SWA:
        return SlidingWindowAttention(attn_config)
    else:
        raise ValueError("unsupported attn type")

def build_mlp_layer(mlp_config: MLPConfig) -> MLPLayer:
    if mlp_config.mlp_type == MLPType.DENSE:
        return MLP(mlp_config)
    elif mlp_config.mlp_type == MLPType.SPARSE_EXPERT:
        return SparseExpertLayer(mlp_config)
    else:
        raise ValueError("unsupported mlp type")

class TransformerBlock(nn.Module):
    """
    Transformer block: communication followed by computation.
    (B, C, D) -> (B, C, D)
    """

    def __init__(self, block_config: BlockConfig):
        """
        """
        super().__init__()
        attn_config, mlp_config = block_config.attention_config, block_config.mlp_config
        if attn_config.D != mlp_config.D:
            raise ValueError(
                f"residual stream dimension mismatch: {attn_config.D=} {mlp_config.D=} ")
        self.attn = build_attn_layer(attn_config)
        self.mlp = build_mlp_layer(mlp_config)
        self.ln1 = nn.LayerNorm(attn_config.D)
        self.ln2 = nn.LayerNorm(attn_config.D)

    def forward(self, X: torch.tensor) -> torch.tensor:
        # (B, C, D) -> (B, C, D)
        # {norm, residual op} for op in {attn, ff}
        X = self.attn(self.ln1(X))
        X = self.mlp(self.ln2(X))
        return X
        