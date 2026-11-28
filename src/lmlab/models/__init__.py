"""One subfolder per model family (toygpt/, later toyllama/, ...). Each assembles shared components
from lmlab.layers / lmlab.tokenization into a concrete architecture and owns its config dataclass."""
from lmlab.models.toygpt import GPTConfig, ToyGPT
