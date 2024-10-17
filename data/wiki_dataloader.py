from collections import Counter, defaultdict
import random
import torch
from typing import List, Tuple

from data.wiki_parser import *

class WikipediaDataLoader:
    def __init__(
            self, 
            articles: List[str], 
            context_length: int=32, 
            batch_size: int=16, 
            vocab_size: int=8192,
            shuffle: bool=False
        ):
        print("In WikipediaDataLoader __init__", flush=True)
        self.C = context_length
        self.B = batch_size
        self.V = vocab_size
        self.shuffle = shuffle
        articles = [a.lower() for a in articles]
        print("Building vocab...", flush=True)

        self.build_vocab(articles)
        self.train_articles, self.test_articles = None, None
        self.init_train_data(articles)
        print("Initialized train data", flush=True)
        self.build_test_data(articles)
        print(f"train data sample: {self.train_data[0:100]}", flush=True)
        self.curr_pos = {"train": 0, "test": 0}

    def build_vocab(self, articles: List[str]):
        print("Building vocab for wikipedia dataset...", flush=True)
        # Combine all articles into one long string, removing double spaces
        full_text = " ".join(articles).split()
        word_counts = Counter(full_text)
        print(f"Full vocab size is {len(word_counts)}", flush=True)
        allowed_words_list = [word for word, _ in word_counts.most_common(self.V-1)]
        allowed_words_list.append("<UNK>")
        self.itos = {i:s for i, s in enumerate(allowed_words_list)}
        self.stoi = defaultdict(lambda: self.V-1)  # unknown strings hence map to <UNK>
        self.stoi.update({s:i for i, s in enumerate(allowed_words_list)})
        self.allowed_words = set(allowed_words_list)

    def init_train_data(self, articles: List[str]):
        if self.train_articles is None:
            self.num_train_articles = int(0.8 * len(articles))
            self.train_articles = articles[0:self.num_train_articles]
        self.build_train_data(first_time=True)

    def build_train_data(self, first_time: bool = False):
        # when self.shuffle is true, we need to recreate the train data string
        if self.shuffle:
            random.shuffle(self.train_articles)

        if first_time or self.shuffle:
            # Combine all train articles into one long string
            full_train_str = " ".join(self.train_articles)
            
            # Remove any double spaces that might have been introduced
            full_train_list = full_train_str.split()
            self.train_data = [self.replace_word(w) for w in full_train_list]
            print("Set self.train_data", flush=True)
            if first_time:
                train_tokens = len(self.train_data)
                num_replaced = sum(1 for w in self.train_data if w == "<UNK>")
                print(f"Train data contains {train_tokens} total tokens. Replaced {num_replaced} tokens.")
                print(f"Percentage of tokens kept: {100.0 * (train_tokens - num_replaced) / train_tokens}", flush=True)

    def build_test_data(self, articles: List[str]):
        self.test_articles = articles[self.num_train_articles:]
        full_test_list = " ".join(self.test_articles).split()
        self.test_data = [self.replace_word(w) for w in full_test_list]
        test_tokens = len(self.test_data)
        num_replaced = sum(1 for w in self.test_data if w == "<UNK>")
        print(f"Test data contains {test_tokens} total tokens. Replaced {num_replaced} tokens.")
        print(f"Percentage of tokens kept: {100.0 * (test_tokens - num_replaced) / test_tokens}", flush=True)

    def replace_word(self, word: str):
        return word if word in self.allowed_words else "<UNK>"
    
    def encode(self, text: List[str]) -> List[int]:
        return [self.stoi[word] for word in text]
    
    def decode(self, vec: List[int]) -> List[str]:
        return " ".join([self.itos[val] for val in vec])
    
    def get_text_batch(self, split: str) -> List[List[str]]:
        """
        Returns a list of self.C+1 tokens from the train or test data.
        """
        # TODO - can make this nicer later; for now this is for compatibility
        # with the interface of previous loaders
        assert(split in ["train", "test"])
        training = split == "train"
        data = self.train_data if training else self.test_data
        pos = self.curr_pos[split]
        readlen = self.B * (self.C+1)

        if pos + readlen > len(data):
            if training:
                self.build_train_data()
            self.curr_pos[split] = 0
            pos = 0

        batch = [data[p:p + self.C + 1] for p in range(pos, pos + readlen, self.C+1)]        
        self.curr_pos[split] += readlen
        return batch
    
    def get_batch(self, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a pair of BxC tensors.
        """
        batch_text = self.get_text_batch(split)
        encoded = [self.encode(seq) for seq in batch_text]
        data = torch.tensor(encoded)  # B x (C+1)
        return data[:, 0:self.C], data[:, 1:self.C+1]
