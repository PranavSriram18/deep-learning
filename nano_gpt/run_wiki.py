import torch # type: ignore

from data.wiki_dataloader import WikipediaDataLoader
from data.wiki_parser import parse_wikipedia_xml
from nano_gpt.generator import Generator
from nano_gpt.trainer import Trainer
from nano_gpt.transformer_model import TransformerModel

# To run: 
# from deep-learning dir:
# source ../dl_env/bin/activate
# python -m nano_gpt.run_wiki > wiki_out.txt

# train hyperparams
batch_size = 16
train_steps = 1024 * 16 + 1
learning_rate = 4e-4
print_every = 1024

# model hyperparams
vocab_size = 4096 * 3
ff_expansion = 2
dropout = 0.0
embedding_dim = 64
context_length = 64
num_heads = 4
num_layers = 12
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
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
    print("Built model", flush=True)

    articles = parse_wikipedia_xml(include_titles=False, limit_articles=500)
    print("Got articles", flush=True)
    data_loader = WikipediaDataLoader(
        articles=articles,
        context_length=context_length,
        batch_size=batch_size,
        vocab_size=vocab_size,
        shuffle=False
    )  # TODO - enable shuffle
    print("Loaded data")

    sample_prompts = ["india is", "the united states is"]
    trainer = Trainer(
        model, 
        data_loader, 
        char_level_tokenize=False, 
        sample_prompts=sample_prompts, 
        sample_length=200
    )
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

    g = Generator(model, data_loader)
    print(g.generate("India is"))
    print(g.generate("A dosa is"))

if __name__=="__main__":
    main()
    