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

def wt2_word_config() -> ModelConfig:
    return ModelConfig(
        batch_size=32,
        vocab_size=40000,
        learning_rate=1.3e-3,
        print_every=256,
        train_steps=2 ** 14 + 1,
        context_length=32,
        embedding_dim=256,
        num_heads=1,
        num_layers=6,
        ff_expansion=2,
        dropout=0.0,
        transformer_type=TransformerType.BASIC_SPARSE_ATTENTION,
        alpha=4.0,
        t=64,
        tie_embeddings=True,
        use_factorized_embeddings=True,
        vocab_embed_dim=96,
        use_ste=True,
        sample_length=500,
        sample_prompts={"India is", "The United States is"}
    )
