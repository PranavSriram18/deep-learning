# nano_gpt/run.py
from dataclasses import dataclass
import torch  # type: ignore

from nano_gpt.trainer import Trainer
from nano_gpt.data_loader import DataLoader
from nano_gpt.transformer_model import TransformerModel
# from nano_gpt.bigram_model import BigramModel  # optional baseline

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
    embedding_dim: int = 128
    context_length: int = 128
    num_heads: int = 8
    num_layers: int = 12

# Sampling controls (not part of ModelConfig per request)
sample_length = 500
sample_prompts = ["Julius: ", "On thy hands he wraithed. "]

def main():
    cfg = ModelConfig()

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # model = BigramModel(vocab_size=cfg.vocab_size)
    model = TransformerModel(
        vocab_size=cfg.vocab_size,
        embedding_dim=cfg.embedding_dim,
        context_length=cfg.context_length,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        ff_expansion=cfg.ff_expansion,
        dropout=cfg.dropout,
    ).to(device)
    print("Built model")

    data_loader = DataLoader(batch_size=cfg.batch_size, block_size=cfg.context_length)
    print("Loaded data")

    trainer = Trainer(
        model=model,
        data_loader=data_loader,
        char_level_tokenize=True,
        sample_prompts=sample_prompts,
        sample_length=sample_length,
        device=device,
        use_amp=True,
    )

    print("Sample before training:\n")
    trainer.print_sample()

    print("Starting training!")
    trainer.train(
        lr=cfg.learning_rate,
        batch_size=cfg.batch_size,
        steps=cfg.train_steps,
        print_every=cfg.print_every,
    )

    print("Sample after training:\n")
    trainer.print_sample()

if __name__ == "__main__":
    main()
