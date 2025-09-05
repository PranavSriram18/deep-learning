# nano_gpt/run.py
from dataclasses import dataclass
from nano_gpt.model_config import DatasetType, ModelConfig, TransformerType, wt2_word_config
import torch  # type: ignore

from nano_gpt.trainer import Trainer
from nano_gpt.data_loader import ShakespeareDataLoader
from nano_gpt.transformer_model import TransformerModel
from nano_gpt.data_wt2_word import WT2WordDataLoader

# from nano_gpt.bigram_model import BigramModel  # optional baseline

def run_transformer(transformer_type: TransformerType, dataset_type: DatasetType):
    if transformer_type == TransformerType.BASIC:
        cfg = ModelConfig(transformer_type=transformer_type)
    else:
        cfg = wt2_word_config()

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    

    vocab_cap = 50_000 if dataset_type == DatasetType.WT2_WORD else 65
    if dataset_type == DatasetType.WT2_WORD:
        data_loader = WT2WordDataLoader(batch_size=cfg.batch_size, block_size=cfg.context_length, vocab_size=vocab_cap)
    elif dataset_type == DatasetType.SHAKESPEARE:
        data_loader = ShakespeareDataLoader(batch_size=cfg.batch_size, block_size=cfg.context_length)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    cfg.vocab_size = getattr(data_loader, "vocab_size_effective", vocab_cap)
    print("Loaded data")

    model = TransformerModel(config=cfg).to(device)
    print(f"Built model with config {cfg}")

    trainer = Trainer(
        model=model,
        data_loader=data_loader,
        char_level_tokenize=dataset_type == DatasetType.SHAKESPEARE,
        sample_prompts=cfg.sample_prompts,
        sample_length=cfg.sample_length,
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
    run_transformer(
        transformer_type=TransformerType.BASIC_SPARSE_ATTENTION,
        dataset_type=DatasetType.WT2_WORD
    )
