"""ToyGPT transformer block: pre-LayerNorm attention and feed-forward, each wrapped in a residual connection."""
from torch import nn

from lmlab.layers.toygpt_attention import MultiHeadAttention
from lmlab.layers.toygpt_feedforward import FeedForward


class TransformerBlock(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.norm1 = nn.LayerNorm(config.d_model)
    self.attn = MultiHeadAttention(config)
    self.norm2 = nn.LayerNorm(config.d_model)
    self.ffn = FeedForward(config)

  def forward(self, x): #transformer block pass through. All the things we compute during attention and ffn which are basically transformations
  #and operations applied to our original x to learn some information are just added to our original sequence
    x = x + self.attn(self.norm1(x)) #calling the model like this auto calls the forward function
    x = x + self.ffn(self.norm2(x))
    return x
