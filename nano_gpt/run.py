from bigram_model import BigramModel
from trainer import Trainer
from data_loader import DataLoader
from transformer_model import TransformerModel

import torch

# To run: 
# source ~/myenv/bin/activate
# python run.py > out.txt

# hyperparams
batch_size = 32
train_steps = 10000
eval_interval = 300
learning_rate = 1e-2
vocab_size = 65  # from inspecting dataset
dropout = 0.0
embedding_dim = 64
context_length = 8
num_heads = 4
num_layers = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__=="__main__":
    #model = BigramModel(vocab_size=vocab_size)
    model = TransformerModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        context_length=context_length,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        device=device
    )
    print("Built model")
    data_loader = DataLoader(batch_size, context_length)
    print("Loaded data")
    trainer = Trainer(model, data_loader)

    

    print("Sample before training:\n")
    print_sample()
    trainer.train(lr=learning_rate, batch_size=batch_size, steps=train_steps)
    print("Sample after training:\n")
    print_sample()
    