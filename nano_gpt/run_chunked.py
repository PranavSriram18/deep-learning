import torch # type: ignore

from nano_gpt.trainer import Trainer  # type: ignore
from nano_gpt.data_loader import DataLoader  # type: ignore
from nano_gpt.chunked_transformer import ChunkedTransformer  # type: ignore

# To run: from deep-learning directory:
# source ../../dl_env/bin/activate  (myenv is older one)
# code chunked_out2.txt && python -m nano_gpt.run_chunked > chunked_out2.txt

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
context_length = 256
num_heads = 8
num_layers = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    model = ChunkedTransformer(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
    )
    print("Built model", flush=True)
    data_loader = DataLoader(batch_size, context_length)
    print("Loaded data", flush=True)

    prompt_0 = """Hark! What light through yonder window breaks? 'Tis the east, and Juliet is the sun. Arise, fair sun, and kill the envious moon, who is already sick and pale with grief that thou, her maid, art far more fair than she. Be not her maid, since she is envious; her vestal livery is but sick and green, and none but fools do wear it. Cast it off! It is my lady; O, it is my love!"""

    prompt_1 = """Be not her maid, since she is envious; her vestal livery is but sick and green, and none but fools do wear it. Cast it off! It is my lady; O, it is my love! O that she knew she were! She speaks, yet she says nothing. What of that? Her eye discourses; I will answer it. I am too bold; 'tis not to me she speaks"""

    sample_prompts = [prompt_0, prompt_1]
    trainer = Trainer(
        model, 
        data_loader, 
        char_level_tokenize=True, 
        sample_prompts=sample_prompts, 
        sample_length=500)
    print("Sample before training:\n", flush=True)
    trainer.print_sample()
    print("Starting training!", flush=True)
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
    