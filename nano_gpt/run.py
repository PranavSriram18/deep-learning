from bigram_model import BigramModel
from trainer import Trainer
from data_loader import DataLoader

import torch

# hyperparams
batch_size = 32
block_size = 8  # context length
train_steps = 10000
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
vocab_size = 65  # from inspecting dataset

if __name__=="__main__":
    model = BigramModel(vocab_size=vocab_size)
    data_loader = DataLoader(batch_size, block_size)
    trainer = Trainer(model, data_loader)

    def print_sample():
        print(
            data_loader.decode(
                model.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()
            )
        )

    print_sample()
    trainer.train(lr=learning_rate, batch_size=batch_size, steps=train_steps)
    print_sample()

    
