import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, tokens, max_seq_len):
        self.max_seq_len = max_seq_len
        # pre-chunk all tokens into input/target pairs
        self.examples = []
        for i in range(0, len(tokens) - max_seq_len, max_seq_len):
            input_chunk  = tokens[i : i + max_seq_len]
            target_chunk = tokens[i+1 : i + max_seq_len + 1]
            self.examples.append((
                torch.tensor(input_chunk,  dtype=torch.long),
                torch.tensor(target_chunk, dtype=torch.long)
            ))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def load_wikitext(name="wikitext-2-raw-v1"):
    from datasets import load_dataset
    raw_dataset = load_dataset("wikitext", name)
    return raw_dataset["train"]["text"], raw_dataset["validation"]["text"]


def tokenize_corpus(tokenizer, corpus):
    tokens = []
    for text in corpus:
        tokens += tokenizer.encode(text)
    return tokens


def build_lm_datasets(tokenizer, train_corpus, val_corpus, max_seq_len):
    """Tokenize both corpora and chunk them into next-token-prediction datasets."""
    train_dataset = TextDataset(tokenize_corpus(tokenizer, train_corpus), max_seq_len)
    val_dataset   = TextDataset(tokenize_corpus(tokenizer, val_corpus),   max_seq_len)
    return train_dataset, val_dataset
