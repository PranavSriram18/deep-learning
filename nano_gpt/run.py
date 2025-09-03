import torch # type: ignore

from nano_gpt.bigram_model import BigramModel
from nano_gpt.trainer import Trainer
from nano_gpt.data_loader import DataLoader
from nano_gpt.transformer_model import TransformerModel

# To run: from deep-learning directory:
# source ../../dl_env/bin/activate  (myenv is older one)
# python -m nano_gpt.run > out.txt

# train hyperparams
batch_size = 32
train_steps = 2 ** 13 + 1
learning_rate = 2e-3
print_every = 256

# model hyperparams
vocab_size = 65  # from inspecting dataset
ff_expansion = 2
dropout = 0.0
embedding_dim = 128
context_length = 128
num_heads = 8
num_layers = 12
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    #model = BigramModel(vocab_size=vocab_size)
    model = TransformerModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        context_length=context_length,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_expansion=ff_expansion,
        dropout=dropout,
        device=device
    )
    print("Built model")
    data_loader = DataLoader(batch_size, context_length)
    print("Loaded data")

    sample_prompts = ["Julius: ", "On thy hands he wraithed. "]
    trainer = Trainer(
        model, data_loader, char_level_tokenize=True, sample_prompts=sample_prompts, sample_length=500)
    print("Sample before training:\n")
    trainer.print_sample()
    print("Starting training!")
    trainer.train(
        lr=learning_rate, 
        batch_size=batch_size, 
        steps=train_steps, 
        print_every=print_every
    )
    print("Sample after training:\n")
    trainer.print_sample()

if __name__=="__main__":
    main()
    