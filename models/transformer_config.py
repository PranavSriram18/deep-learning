from dataclasses import dataclass

from layers.layer_config import BlockConfig

@dataclass
class TransformerConfig:
    block_config: BlockConfig
    vocab_size: int
    embedding_dim: int
    context_length: int
    num_layers: int
    use_factorized_embeddings: bool
    vocab_embed_dim: int
    tie_embeddings: bool

    def __post_init__(self):
        assert self.embedding_dim == self.block_config.mlp_config.D
