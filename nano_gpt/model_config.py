from dataclasses import dataclass
from enum import Enum


class TransformerType(Enum):
    BASIC = "basic"
    BASIC_SPARSE_ATTENTION = "basic_sparse_attention"

class DatasetType(Enum):
    WT2_WORD = "wt2_word"
    SHAKESPEARE = "shakespeare"

@dataclass
class ModelConfig:
    # Training hyperparameters
    batch_size: int = 32
    train_steps: int = 2 ** 13 + 1
    learning_rate: float = 2e-3
    print_every: int = 256

    # Model hyperparameters
    vocab_size: int = 65
    ff_expansion: int = 2
    dropout: float = 0.0
    embedding_dim: int = 128  # D
    context_length: int = 128  # C
    num_heads: int = 8
    num_layers: int = 12  # L
    transformer_type: TransformerType = TransformerType.BASIC
    alpha: float = 4.0
    t: int = 24
    tie_embeddings: bool = True