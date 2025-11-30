from dataclasses import dataclass
from enum import Enum

class MLPType(Enum):
    DENSE = "dense"  # ordinary dense MLP
    MOE = "moe"
    SPARSE_EXPERT = "sparse_expert"

class AttentionType(Enum):
    MHA = "mha"  # ordinary multi-head attention
    SWA = "swa"  # sliding-window attention

@dataclass
class MLPConfig:
    """
    Config for MLP layers. We view ordinary MLP as a limit case of MoEs.
    """
    D: int  # residual stream dim
    m: int  # total number of experts
    k: int  # active experts
    b: int  # expert width
    k_f: int  # fixed experts
    mlp_type: MLPType
    lambda_coeff: float   # layer forward returns x + lambda_coeff * mlp(x)

    @staticmethod
    def dense_default(D: int = 256):
        return MLPConfig(D=D, m=1, k=1, b=4*D, k_f=0, mlp_type=MLPType.DENSE, lambda_coeff=1.)

    @staticmethod
    def sparse_default(D: int = 256):
        return MLPConfig(
            D=D, m=128, k=8, b=64, k_f=0, mlp_type=MLPType.SPARSE_EXPERT, lambda_coeff=0.5)

@dataclass
class AttentionConfig:
    D: int  # residual stream dim
    C: int  # context length
    num_heads: int  # number of heads; H = D // num_heads is head dim
    attention_type: AttentionType

    @staticmethod
    def mha_default(D: int = 256) -> "AttentionConfig":
        return AttentionConfig(D=D, C=64, num_heads=8, attention_type=AttentionType.MHA)

    @staticmethod
    def sliding_window_default(D: int = 256) -> "AttentionConfig":
        return AttentionConfig(D=D, C=64, num_heads=8, attention_type=AttentionType.SWA)


@dataclass
class BlockConfig:
    mlp_config: MLPConfig
    attention_config: AttentionConfig

    def __post_init__(self):
        assert self.mlp_config.D == self.attention_config.D

