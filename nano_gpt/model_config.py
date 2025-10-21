from dataclasses import dataclass, field
from enum import Enum
from re import S
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
    batch_size: int = 64
    learning_rate: float = 1.5e-3
    print_every: int = 256
    train_steps: int = 16000

    # model hyperparameters
    embedding_dim: int = 384
    vocab_size: int = 48000
    vocab_embed_dim: int = 96
    num_heads_regular: int = 8
    num_heads_sparse: int = 1
    ff_expansion: int = 2
    dropout: float = 0.0
    alpha: float = 4.0
    t: int = 64  # 64-sparse in 1536 dims

    # layers
    num_regular_init: int = 6
    sliding_window_init: int = 16
    
    num_sparse_blocks: int = 3
    context_length: int = 96

    num_regular_final: int = 4
    sliding_window_final: int = 32
    
    use_ste: bool = True
    sample_length: int = 500
    sample_prompts: set[str] = field(default_factory=lambda: {"India is", "The United States is"})

def v2_shakespeare_config() -> V2ModelConfig:
    cfg = V2ModelConfig()
    cfg.batch_size = 32
    cfg.learning_rate = 1e-3
    cfg.print_every = 512
    cfg.train_steps = 5000

    cfg.embedding_dim = 128
    cfg.vocab_embed_dim = 64
    cfg.vocab_size = 65
    cfg.num_heads_regular = 8
    cfg.num_heads_sparse = 1
    cfg.alpha = 4.0
    cfg.t: int = 32  # 32-sparse in 512 dims

    cfg.num_regular_init = 3
    cfg.sliding_window_init = 16

    cfg.num_sparse_blocks = 4
    cfg.context_length = 96

    cfg.num_regular_final = 2
    cfg.sliding_window_final = 16

    cfg.use_ste = True
    cfg.sample_length = 500
    # Set concrete prompts (avoid assigning dataclasses.field here)
    cfg.sample_prompts = [
        "Julius: ",
        "On thy hands he wraithed. ",
    ]

    return cfg

