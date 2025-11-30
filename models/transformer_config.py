from dataclasses import dataclass
from enum import Enum

from layers.layer_config import BlockConfig
from nano_gpt.run_wiki import embedding_dim, vocab_size

@dataclass
class TransformerConfig:
    block_config: BlockConfig
    vocab_size: int
    embedding_dim: int
    num_layers: int
    use_factorized_embeddings: bool
    vocab_embed_dim: int

    def __post_init__(self):
        assert self.embedding_dim == self.block_config.mlp_config.D
