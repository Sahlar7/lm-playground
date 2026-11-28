"""Reusable neural-network building blocks, shared across every model in the lab.

Files are named <model>_<component>.py after the model family they were written for (whose config
fields / design they assume), e.g. toygpt_attention.py, toyllama_transformer_block.py. Truly
model-agnostic layers drop the prefix. Add new variants (RoPE, RMSNorm, SwiGLU...) as new files here
next to the originals rather than editing those in place, and let each model pick the ones it uses."""
from lmlab.layers.toygpt_attention import AttentionHead, MultiHeadAttention
from lmlab.layers.toygpt_feedforward import FeedForward
from lmlab.layers.toygpt_transformer_block import TransformerBlock
