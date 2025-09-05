from dataclasses import dataclass, field
from enum import Enum
from typing import List


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
    sliding_window: int = 16  # Cw
    num_heads: int = 8
    num_layers: int = 12  # L
    transformer_type: TransformerType = TransformerType.BASIC
    alpha: float = 4.0
    t: int = 24
    tie_embeddings: bool = True
    use_factorized_embeddings: bool = False
    vocab_embed_dim: int = 256
    use_ste: bool = True

    # sampling controls
    sample_length: int = 500
    sample_prompts: set[str] = field(default_factory=lambda: {"Julius: ", "On thy hands he wraithed. "})

@dataclass 
class V2ModelConfig:
    # training hyperparameters
    batch_size: int = 32
    learning_rate: float = 1.3e-3
    print_every: int = 256
    train_steps: int = 20000

    # model hyperparameters
    embedding_dim: int = 256
    context_length: int = 64
    sliding_window: int = 16
    vocab_size: int = 48000
    vocab_embed_dim: int = 96
    num_heads_regular: int = 4
    num_heads_sparse: int = 1
    num_regular_blocks: int = 4
    num_sparse_blocks: int = 2
    ff_expansion: int = 2
    dropout: float = 0.0
    alpha: float = 4.0
    t: int = 48  # 48-sparse in 1024 dims
    
    use_ste: bool = True
    sample_length: int = 500
    sample_prompts: set[str] = field(default_factory=lambda: {"India is", "The United States is"})
        