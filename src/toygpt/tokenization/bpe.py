#Tokenizer - byte-level BPE Algorithm (GPT-2 style), no external tokenizer needed
import heapq
import json
from collections import defaultdict
from pathlib import Path

import regex  # like `re`, but supports \p{L} (any letter) / \p{N} (any number) unicode classes

# GPT-2's pre-tokenization rule: split text into "words" before BPE, so merges never cross word
# boundaries. Pieces: common English contractions, an optional leading space + letters,
# optional space + numbers, optional space + punctuation, then whitespace runs.
# `\s+(?!\S)` keeps the last space of a whitespace run attached to the following word.
GPT2_SPLIT_PATTERN = regex.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

def bytes_to_unicode():
    """Map each of the 256 byte values to a printable Unicode character."""
    # Bytes that are already visible, non-whitespace characters keep their own character
    bs = (list(range(ord("!"), ord("~") + 1))      # 33..126  printable ASCII
        + list(range(ord("¡"), ord("¬") + 1))      # 161..172 printable Latin-1
        + list(range(ord("®"), ord("ÿ") + 1)))     # 174..255 printable Latin-1
    cs = bs[:]
    # The other 68 bytes (control chars, space, DEL, 127-160, 173) are shifted
    # up to 256, 257, ... in order, so each gets a distinct visible character
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))

BYTE_TO_CHAR = bytes_to_unicode()
CHAR_TO_BYTE = {v: k for k, v in BYTE_TO_CHAR.items()}

# Base vocab in byte order, so token id b is exactly byte b (ids 0..255).
# (BYTE_TO_CHAR.values() is in insertion order, which starts at "!" = 33, so don't use that.)
BYTE_CHARACTERS = [BYTE_TO_CHAR[b] for b in range(256)]

SPECIAL_TOKENS = ["<|endoftext|>"]


def pretokenize(text):
    """Split text into words, then turn each word's UTF-8 bytes into byte-mapped chars.
    e.g. " café" -> "ĠcafÃ©". Same output as GPT-2's HuggingFace pre-tokenizer."""
    return ["".join(BYTE_TO_CHAR[b] for b in word.encode("utf-8"))
            for word in GPT2_SPLIT_PATTERN.findall(text)]


def merge_ids(ids, pair, new_id):
  """Replace every non-overlapping occurrence of pair in ids with new_id."""
  out = []
  i = 0
  while i < len(ids):
    if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
      out.append(new_id)
      i += 2
    else:
      out.append(ids[i])
      i += 1
  return out


def count_pairs(ids):
  """Number of times each adjacent pair occurs in one word."""
  counts = defaultdict(int)
  for pair in zip(ids, ids[1:]):
    counts[pair] += 1
  return counts


class BPETokenizer:
  def __init__(self, vocab_size, special_tokens=SPECIAL_TOKENS):
    self.vocab_size = vocab_size     # total size: 256 bytes + merges + special tokens
    self.special_token_list = list(special_tokens)
    self.merges = {}                 # (left_id, right_id) -> new_id; lower id = learned earlier = higher priority
    self.word_freqs = defaultdict(int)
    self.splits = {}                 # word -> list of token ids
    self.vocab = []                  # id -> token string (byte-mapped chars)
    self.special_tokens = {}         # special token string -> id
    self._cache = {}                 # word -> encoded ids, so repeated words are only merged once
    self._special_pattern = None


  def train(self, corpus):
    self.word_freqs = defaultdict(int)
    #pre tokenize
    print("Pretokenizing...")
    for text in corpus:
        for word in pretokenize(text):
            self.word_freqs[word] += 1

    #create vocab: always the same 256 byte tokens, whatever the corpus, so nothing is ever unknown
    print("Creating vocab...")
    self.vocab = BYTE_CHARACTERS.copy()
    self.merges = {}
    self.splits = {word: [CHAR_TO_BYTE[c] for c in word] for word in self.word_freqs}

    # Instead of recounting every pair in the corpus after each merge, count once, then only
    # update the words that actually contain the merged pair.
    #   pair_freqs[pair]     = total occurrences across the corpus (weighted by word frequency)
    #   pair_to_words[pair]  = which words contain that pair
    pair_freqs = defaultdict(int)
    pair_to_words = defaultdict(set)
    for word, freq in self.word_freqs.items():
      for pair, n in count_pairs(self.splits[word]).items():
        pair_freqs[pair] += n * freq
        pair_to_words[pair].add(word)

    # Max-heap of (-freq, pair) so we can find the most frequent pair without scanning them all.
    # Counts change as we merge; rather than updating entries in place we push a fresh entry,
    # and skip stale ones when popped (their freq no longer matches pair_freqs).
    # Ties break on the smaller pair ids, so training is deterministic.
    heap = [(-f, pair) for pair, f in pair_freqs.items()]
    heapq.heapify(heap)

    print("Training...")
    num_merges = self.vocab_size - len(BYTE_CHARACTERS) - len(self.special_token_list)
    for _ in range(num_merges):
      best_pair = None
      while heap:
        neg_freq, pair = heapq.heappop(heap)
        if pair_freqs.get(pair, 0) == -neg_freq and -neg_freq > 0:
          best_pair = pair
          break
      if best_pair is None:  # no pairs left to merge (every word is a single token)
          break

      new_id = len(self.vocab)  # 256, 257, ... in the order merges are learned
      self.merges[best_pair] = new_id
      self.vocab.append(self.vocab[best_pair[0]] + self.vocab[best_pair[1]])

      # re-split only the words containing best_pair, and adjust their pair counts
      changed = set()
      for word in list(pair_to_words[best_pair]):
        freq = self.word_freqs[word]
        old_counts = count_pairs(self.splits[word])
        new_split = merge_ids(self.splits[word], best_pair, new_id)
        new_counts = count_pairs(new_split)
        self.splits[word] = new_split

        for pair, n in old_counts.items():
          pair_freqs[pair] -= n * freq
          if pair not in new_counts:
            pair_to_words[pair].discard(word)
        for pair, n in new_counts.items():
          pair_freqs[pair] += n * freq
          pair_to_words[pair].add(word)
        changed.update(old_counts)
        changed.update(new_counts)

      del pair_freqs[best_pair], pair_to_words[best_pair]
      for pair in changed:
        if pair_freqs.get(pair, 0) > 0:
          heapq.heappush(heap, (-pair_freqs[pair], pair))

    # special tokens go last; they are never produced by a merge
    self.special_tokens = {}
    for tok in self.special_token_list:
      self.special_tokens[tok] = len(self.vocab)
      self.vocab.append(tok)
    self._reset_caches()


  def _reset_caches(self):
    self._cache = {}
    # matches any special token literally, e.g. "<|endoftext|>" (longest first, in case one contains another)
    if self.special_tokens:
      alternatives = sorted(self.special_tokens, key=len, reverse=True)
      self._special_pattern = regex.compile("(" + "|".join(regex.escape(t) for t in alternatives) + ")")
    else:
      self._special_pattern = None


  @property
  def eot_id(self):
    return self.special_tokens.get("<|endoftext|>")


  def _encode_word(self, word):
    if word in self._cache:
      return self._cache[word]
    # Start from the word's bytes: every byte is a base token, so no [UNK] is ever needed
    ids = [CHAR_TO_BYTE[c] for c in word]

    # Replay merges in the order they were learned: always apply the lowest-id
    # (earliest learned) merge present in the word, until none apply
    while len(ids) >= 2:
      pairs = set(zip(ids, ids[1:]))
      best_pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
      if best_pair not in self.merges:
        break  # no learned merge applies anymore
      ids = merge_ids(ids, best_pair, self.merges[best_pair])

    self._cache[word] = ids
    return ids


  def encode(self, text):
    # Special tokens are found first, so "<|endoftext|>" in the text becomes its single id
    # instead of being pretokenized into "<", "|", "endoftext", "|>"
    chunks = self._special_pattern.split(text) if self._special_pattern else [text]

    encoded_tokens = []
    for chunk in chunks:
      if chunk in self.special_tokens:
        encoded_tokens.append(self.special_tokens[chunk])
        continue
      for word in pretokenize(chunk):
        encoded_tokens.extend(self._encode_word(word))
    return encoded_tokens


  def decode(self, token_ids):
    special_ids = {i: t for t, i in self.special_tokens.items()}
    raw = bytearray()
    for i in token_ids:
      if i in special_ids:
        raw += special_ids[i].encode("utf-8")  # special tokens are literal text, not byte-mapped chars
      else:
        raw += bytes(CHAR_TO_BYTE[c] for c in self.vocab[i])  # "ĠcafÃ©" -> b' caf\xc3\xa9'
    return raw.decode("utf-8", errors="replace")  # ' café'; a cut-off multi-byte char becomes '�'


  def save(self, path):
    """Merges in learned order are all that's needed: the vocab is rebuilt from them."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    merges = sorted(self.merges.items(), key=lambda item: item[1])
    data = {
        "type": "byte_bpe",
        "vocab_size": self.vocab_size,
        "merges": [list(pair) for pair, _ in merges],
        "special_tokens": self.special_tokens,
    }
    with open(path, "w", encoding="utf-8") as f:
      json.dump(data, f)


  @classmethod
  def load(cls, path):
    with open(path, encoding="utf-8") as f:
      data = json.load(f)
    tok = cls(data["vocab_size"], special_tokens=list(data["special_tokens"]))
    tok.vocab = BYTE_CHARACTERS.copy()
    for a, b in data["merges"]:
      tok.merges[(a, b)] = len(tok.vocab)
      tok.vocab.append(tok.vocab[a] + tok.vocab[b])
    tok.special_tokens = {}
    for t, i in sorted(data["special_tokens"].items(), key=lambda item: item[1]):
      tok.special_tokens[t] = len(tok.vocab)
      tok.vocab.append(t)
    tok._reset_caches()
    return tok
