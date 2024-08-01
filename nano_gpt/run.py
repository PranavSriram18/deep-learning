import torch # type: ignore

from bigram_model import BigramModel
from trainer import Trainer
from data_loader import DataLoader
from transformer_model import TransformerModel

# To run: 
# source ../../dl_env/bin/activate  (myenv is older one)
# python run.py > out.txt

# train hyperparams
batch_size = 32
train_steps = 10000
eval_interval = 300
learning_rate = 1e-2
print_every = 1000

# model hyperparams
vocab_size = 65  # from inspecting dataset
ff_expansion = 4
dropout = 0.0
embedding_dim = 64
context_length = 16
num_heads = 4
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
    trainer = Trainer(model, data_loader)
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
    