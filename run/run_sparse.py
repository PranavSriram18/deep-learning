import torch  # type: ignore
from data.shakespeare_data_loader import ShakespeareDataLoader
from models.transformer import TransformerModel
from models.transformer_config import TransformerConfig
from layers.layer_config import BlockConfig, MLPConfig, AttentionConfig, AttentionType, MLPType
from train.trainer import Trainer, TrainConfig

def run(
    D: int = 128,
    C: int = 64,
    L: int = 2,
    num_heads: int = 8,
    m: int = 256,
    k: int = 32,
    b: int = 8,
    batch_size: int = 128,
    lr: float = 1e-3,
    print_every: int = 400,
    train_steps: int = 20000,
    sample_length: int = 250,
    aux_loss_weight: float = 0.5
):
    tie_embeddings = True
    use_factorized_embeddings = False
    vocab_embed_dim = D       # when not factorized, Dv must equal D

    # Training settings (separate config for Trainer)
    train_cfg = TrainConfig(
        batch_size=batch_size,
        learning_rate=lr,
        print_every=print_every,
        train_steps=train_steps,
        sample_length=sample_length,
        sample_prompts=[
            "Julius: ",
            "On thy hands he wraithed. ",
        ],
        char_level_tokenize=True,
        use_amp=False,
    )

    # Block-level configs (copy D/C from transformer settings)
    attention_cfg = AttentionConfig(
        D=D,
        C=C,
        num_heads=num_heads,
        attention_type=AttentionType.MHA,
    )
    mlp_cfg = MLPConfig(
        D=D,
        m=m,                        # total experts
        k=k,                          # active experts
        b=b,                         # expert width
        mlp_type=MLPType.SPARSE_EXPERT_V3,
    )
    block_cfg = BlockConfig(
        mlp_config=mlp_cfg,
        attention_config=attention_cfg,
    )

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    
    data_loader = ShakespeareDataLoader(batch_size=train_cfg.batch_size, block_size=C)
    vocab_size = data_loader.vocab_size()
    print("Loaded data")

    # Finalize model config (after vocab is known)
    transformer_cfg = TransformerConfig(
        block_config=block_cfg,
        vocab_size=vocab_size,
        embedding_dim=D,
        context_length=C,
        num_layers=L,
        use_factorized_embeddings=use_factorized_embeddings,
        vocab_embed_dim=vocab_embed_dim,
        tie_embeddings=tie_embeddings,
        aux_loss_weight=aux_loss_weight,
    )

    model = TransformerModel(config=transformer_cfg).to(device)
    print(f"Built model with config {transformer_cfg}")

    trainer = Trainer(
        model=model,
        data_loader=data_loader,
        train_config=train_cfg,
        device=device,
    )

    print("Sample before training:\n")
    trainer.print_sample()

    print("Starting training!")
    trainer.train(
        lr=train_cfg.learning_rate,
        batch_size=train_cfg.batch_size,
        steps=train_cfg.train_steps,
        print_every=train_cfg.print_every,
    )

    print("Sample after training:\n")
    trainer.print_sample()

if __name__=="__main__":
    run()