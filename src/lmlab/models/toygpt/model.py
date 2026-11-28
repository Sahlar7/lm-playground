from dataclasses import dataclass

import torch
from torch import nn

from lmlab.layers.toygpt_transformer_block import TransformerBlock


@dataclass
class GPTConfig:
    vocab_size:  int = 50257  # 50257 for gpt2; set to len(tokenizer.vocab) for our own BPE
    d_model:     int = 512
    n_heads:     int = 8
    d_head:      int = 64
    n_layers:    int = 6
    max_seq_len: int = 128

'''
Note:
Embedding Layer
Really just serves as a lookup table where given a token ID, we can retrieve the corresponding embedding vector.
The actual embedding vectors are learned during training of the whole transformer, its not something we train indepenently.
At this stage, we just have token ids which are unique per each word and don't consider position.
Positional encoding happens right after we get the embedding vector: given a base embedding vector for a unique token id,
we can then add positional information before we enter the transformer block.
Similarly, positional encodings are also learned.
'''

class ToyGPT(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
    self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
    self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
    self.norm = nn.LayerNorm(config.d_model)
    self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)



  def forward(self, token_ids):
    #first we get our token's vectors from the embedding model
    #(just a map from id to vector), and add the corresponding positional vectors
    #via element wise addition
    seq_len = token_ids.shape[1]
    positions = torch.arange(seq_len, device=token_ids.device)

    x = self.token_emb(token_ids) + self.pos_emb(positions)
    #x is our input sequence as embedding vectors, added with position vectors
    #we must now look through our sequence and determine how much each token
    #attends to one another, i.e pass to our attention mechanism

    #pass through transformer blocks: normalization->attention->normalization->ffn
    for block in self.blocks:
      x = block(x)
    #at this point, we have passed our input sequence through several transformer blocks,
    #iteratively refining on our original embedding vector going into our value space to have better encoded information
    #of all of the tokens in our sequence contextually, including position, the base embeddings, context via attention, non linearity and
    #expanded features via our ffn, and the benefits of separating our embedding semantics via multiheaded attention. Now we need to get
    #our final output

    x = self.norm(x) #final normalization layer, just normalize values in our vectors

    #output: logits and probabilities. Final head is just a single linear layer (we learn this too) mapping the all the vectors to logits
    #so we have an output for each token in x a vector of vocab size. This tells us for each word, how likely the next word is that at the corresponding
    #position for the logits in our vector.
    #We use logits directly for getting cross entropy loss in training, and we only care about the last logit vector position, aka
    #the logit vector for our current token, for predicting the next word and we just choose the highest value.

    logits = self.head(x)
    #probs = nn.Softmax(dim=-1)(logits) don't need to return these
    return logits
