#Tokenizer - BPE Algorithm
from collections import defaultdict

from transformers import AutoTokenizer #will just use this for pretokenization: splitting each word in our corpus


class BPETokenizer:
  def __init__(self, vocab_size, pretokenizer=None):
    self.vocab_size = vocab_size
    self.merges = {}
    self.word_freqs = defaultdict(int)
    self.splits = {}
    self.vocab = []
    # load lazily so importing this module doesn't hit the HF hub
    self.pretokenizer = pretokenizer if pretokenizer is not None else AutoTokenizer.from_pretrained("gpt2")


  def train(self, corpus):
    self.word_freqs = defaultdict(int)
    #pre tokenize
    print("Pretokenizing...")
    for text in corpus:
        words_with_offsets = self.pretokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        new_words = [word for word, offset in words_with_offsets]
        for word in new_words:
            self.word_freqs[word] += 1

    #create vocab
    print("Creating vocab...")
    alphabet = []
    for word in self.word_freqs.keys():
        for letter in word:
            if letter not in alphabet:
              alphabet.append(letter)
        alphabet.sort()
    self.vocab = ["<|endoftext>", "[UNK]"] + alphabet.copy() # Added [UNK] to initial vocab

    self.splits = {word: [c for c in word] for word in self.word_freqs.keys()}

    def compute_pair_freqs():
      pair_freqs = defaultdict(int)
      for word, freq in self.word_freqs.items():
        split = self.splits[word]
        for i in range(len(split) - 1):
          pair = (split[i], split[i+1])
          pair_freqs[pair] += freq
      return pair_freqs

    def get_max_pair(pair_freqs):
      best_pair = ""
      max_freq = None
      for pair, freq in pair_freqs.items():
        if max_freq is None or max_freq < freq:
          best_pair = pair
          max_freq = freq
      return best_pair, max_freq

    def merge_pair(a, b):
        for word in self.word_freqs:
            split = self.splits[word]
            if len(split) == 1:
                continue

            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [a + b] + split[i + 2 :]
                else:
                    i += 1
            self.splits[word] = split

    print("Training...")
    while len(self.vocab) < self.vocab_size:
      pair_freqs = compute_pair_freqs()
      best_pair, max_freq = get_max_pair(pair_freqs)
      if max_freq is None or max_freq == 0: # Handle cases where no more merges can be made (max_freq is None means pair_freqs was empty)
          break
      merge_pair(*best_pair)
      self.merges[best_pair] = best_pair[0] + best_pair[1]
      self.vocab.append(best_pair[0] + best_pair[1])

  def encode(self, text):
    # Pre-tokenize the input text into words
    words_with_offsets = self.pretokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_words = [word for word, offset in words_with_offsets]

    encoded_tokens = []
    for word in pre_tokenized_words:
        # Start with individual characters for each word
        current_split = [c for c in word]

        # Apply learned merges greedily
        while True:
            best_merge_idx = -1
            best_merge_pair = None
            for i in range(len(current_split) - 1):
                pair = (current_split[i], current_split[i+1])
                if pair in self.merges:
                    # Find the earliest merge in the sequence to apply
                    if best_merge_idx == -1 or self.vocab.index(self.merges[pair]) < self.vocab.index(self.merges[best_merge_pair]):
                         best_merge_idx = i
                         best_merge_pair = pair

            if best_merge_idx != -1:
                merged_token = self.merges[best_merge_pair]
                current_split = current_split[:best_merge_idx] + [merged_token] + current_split[best_merge_idx + 2:]
            else:
                break # No more merges possible for this word

        # Map the final split tokens to their IDs from the vocabulary
        for token in current_split:
            if token in self.vocab:
                encoded_tokens.append(self.vocab.index(token))
            else:
                # Use [UNK] token for out-of-vocabulary (OOV) words
                if "[UNK]" not in self.vocab:
                    # Fallback: if somehow [UNK] isn't in vocab, add it.
                    self.vocab.append("[UNK]")
                encoded_tokens.append(self.vocab.index("[UNK]"))

    return encoded_tokens

  def decode(self, token_ids):
    decoded_text = []
    for token_id in token_ids:
      if 0 <= token_id < len(self.vocab):
        decoded_text.append(self.vocab[token_id])
      else:
        # Handle unknown token IDs, e.g., by skipping or using a placeholder
        decoded_text.append("[UNK]") # Placeholder for unknown IDs
    return "".join(decoded_text).replace("Ġ", " ").strip()
