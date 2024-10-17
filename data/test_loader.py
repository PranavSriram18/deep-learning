
from wiki_dataloader import *

test_articles = ["This is the content of article 0. It's a short article.",
                "Here's article 1. It's also quite brief but a bit longer than the first.",
                "Here's article 2. It is somewhat less brief than the first two \
                    articles, but not by much."] 

def test_loader_basic():
    dataloader = WikipediaDataLoader(
        test_articles, context_length=4, batch_size=2, vocab_size=64)

    for i in range(2):
        print(f"Batch {i}:")
        batch = dataloader.get_text_batch("train")
        for j, text in enumerate(batch):
            print(f"  Chunk {j}: {text}")

def test_loader_encoding():
    dataloader = WikipediaDataLoader(
        test_articles, context_length=4, batch_size=2, vocab_size=64)

    for i in range(2):
        print(f"Batch {i}:")
        batch = dataloader.get_batch("train")
        print(f"current batch: {batch}")

def test_loader_full():
    titles_and_articles = parse_wikipedia_xml("simplewiki.xml")
    articles = [article for title, article in titles_and_articles]
    print("Successfully extracted articles. Building loader...\n")
    dataloader = WikipediaDataLoader(articles)
    print("Successfully built loader. Printing first 5 samples...\n")
    for i in range(5):
        print(f"Batch {i}:")
        batch = dataloader.get_text_batch('train')
        for j, text in enumerate(batch):
            print(f"    Chunk {j}: {text}")



if __name__ == "__main__":
    test_loader_basic()
    test_loader_encoding()