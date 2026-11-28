"""ToyGPT position-wise feed-forward network: d_model -> 4*d_model -> d_model, ReLU, dropout 0.1."""
from torch import nn


class FeedForward(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(config.d_model, 4*config.d_model),
        nn.ReLU(),
        nn.Dropout(config.dropout),
        nn.Linear(4*config.d_model, config.d_model),
        nn.Dropout(config.dropout)
    )
    #ffn in a transformer generally follows a 2 (input, hidden, output) layer narrow -> wide -> narrow structure. We don't change the dimension of our vector in terms of
    #input to output, but expanding in the middle lets us discover some more complex patterns, and then in the end go into our original space

  def forward(self, x):
    return self.net(x) #just pass x, which will just be original sequence x + our output tensor
    #from multihead self attention, through the neural net
