# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

A lab/playground for building language models from scratch and replicating or creating LLM/AI/ML research. The `lmlab` package is organized so that components (layers, tokenizers, training loops, …) are reusable across many models, and each model (currently `toygpt`; later e.g. `toyllama`) is assembled from them. The code is heavily commented as learning notes — preserve those explanatory comments when editing or moving code.

## Commands

Windows machine; the virtualenv lives in `venv/` and `lmlab` is installed editable into it.

```bash
venv/Scripts/python.exe -m pip install -e ".[dev,notebook]"    # setup (src/ layout, must be installed)
venv/Scripts/python.exe -m pytest tests/                        # all tests
venv/Scripts/python.exe -m pytest tests/test_byte_bpe.py::test_round_trip   # single test
venv/Scripts/python.exe experiments/toygpt/train.py --config configs/toygpt/baseline.yaml   # train
```

Training downloads `Salesforce/wikitext` (wikitext-2-raw-v1) via HF `datasets` and uses CUDA if available. There is no linter/formatter configured.

## Structure & naming convention

```
src/lmlab/
  layers/  tokenization/  data/  training/  inference/   shared, reusable components
  models/<model>/                                         model-specific assembly + its config dataclass
  config.py                                               generic YAML <-> dataclass loading
configs/<model>/*.yaml    experiments/<model>/ (train script, notebooks)    runs/<name>/ (outputs)
```

- **Shared folders hold components; `models/<model>/` holds the architecture that wires them together.** Don't put model assembly code in shared folders or reusable layers inside a model folder.
- **Files in shared folders are named `<model>_<component>.py`** when written for (or assuming the config/design of) a specific model family — e.g. `layers/toygpt_attention.py`, `layers/toygpt_transformer_block.py`, later `layers/toyllama_transformer_block.py`. Model-agnostic files drop the prefix (`training/trainer.py`, `tokenization/byte_bpe.py`, `inference/generate.py`, `data/text.py`).
- Add new variants (RoPE, RMSNorm, SwiGLU, a new tokenizer algorithm, …) as **new files** next to the originals rather than modifying them in place, so earlier models/experiments keep reproducing. Each shared folder's `__init__.py` re-exports its public classes.
- **Adding a new model** (e.g. toyllama): `layers/toyllama_*.py` for any new components → `models/toyllama/model.py` with its config dataclass + `nn.Module`, re-exported from `models/toyllama/__init__.py` and `models/__init__.py` → `configs/toyllama/*.yaml` → `experiments/toyllama/train.py` (copy `experiments/toygpt/train.py`, note `PROJECT_ROOT = parents[2]`) → tests in `tests/`.

## Architecture

The pipeline is wired together per model in `experiments/<model>/train.py` (and mirrored in `experiments/toygpt/ToyGPT_Playground.ipynb`); library modules deliberately don't know about each other:

- **Config** (`config.py`): `load_config(path, model_config_cls)` → `(raw, model_cfg, TrainingConfig)`. YAML has top-level `name`, `tokenizer` (`gpt2` | `bpe`), `bpe_vocab_size`, plus `model:` and `training:` sections splatted into the dataclasses — unknown keys raise `TypeError` by design. `model.vocab_size` is never set in YAML; the train script fills it from the tokenizer. New experiment = copy a config and override.
- **Tokenizers**: either HF `GPT2Tokenizer` or our `BPETokenizer` (`tokenization/byte_bpe.py`). Everything downstream relies only on the duck-typed `encode(text) -> list[int]` / `decode(ids) -> str` interface. Vocab size comes from `tok.vocab_size` for GPT-2 but `len(tok.vocab)` for ours.
- **BPE invariants** (enforced by `tests/test_byte_bpe.py`): GPT-2-compatible pre-tokenization regex and byte↔unicode mapping; ids 0–255 are exactly the raw bytes (in byte order, not `bytes_to_unicode()` insertion order); merges get ids 256+ in learned order and lower id = higher priority at encode time; special tokens (`<|endoftext|>`) are appended last and split out before pre-tokenization. Training uses incremental pair counts + a lazy max-heap (stale entries skipped on pop). `save()` stores only ordered merges + special tokens; `load()` rebuilds the vocab from them. Encoding must round-trip arbitrary UTF-8 exactly.
- **ToyGPT** (`models/toygpt/model.py` + `layers/toygpt_*.py`): pre-LayerNorm transformer with learned token + positional embeddings, per-head `AttentionHead` modules concatenated in `MultiHeadAttention` (with W_O projection), ReLU FFN (4×), hard-coded dropout 0.1. Layers take the whole `GPTConfig` (`d_model`, `n_heads`, `max_seq_len`). Head dim is computed as `d_model // n_heads`; `GPTConfig.d_head` currently exists but is unused. `forward` returns logits only.
- **Data** (`data/text.py`): tokenizes the whole corpus into one flat stream and pre-chunks it into non-overlapping `(input, target shifted by 1)` pairs of length `max_seq_len`.
- **Training** (`training/trainer.py`): architecture/tokenizer-agnostic loop (Adam + cross-entropy); works with any model mapping `(B, T)` ids → `(B, T, vocab)` logits; saves `best.pt` (state_dict) by **val** loss. Fine-tuning is meant to be: load a checkpoint, call `train()` with other data.
- **Inference** (`inference/generate.py`): temperature sampling, crops context to `model.config.max_seq_len`.

## Outputs

Each run writes to `runs/<name>/` (`name` from the YAML): `best.pt`, `config.yaml` (resolved config incl. tokenizer choice), and `tokenizer.json` when using our BPE (reload with `BPETokenizer.load`). `runs/*` and `*.pt` are gitignored.

## Gotchas

- Keep the `if __name__ == "__main__"` guard in training scripts: DataLoader `num_workers > 0` spawns processes on Windows. In notebooks use `num_workers=0`.
- The notebook's first cell targets Colab (clones the repo to `/content/lm-playground` and adds `src/` to `sys.path`); locally, rely on the editable install instead.
- `experiments/toygpt/ToyGPT_Playground_original.ipynb` is the original standalone, pre-refactor notebook kept for reference; it does not import `lmlab`.
- Printing decoded text on the Windows console can raise `UnicodeEncodeError` (cp1252); set `PYTHONIOENCODING=utf-8`.
