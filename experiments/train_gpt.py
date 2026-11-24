"""Train ToyGPT from a config file.

    python experiments/train_gpt.py --config configs/gpt_baseline.yaml

Outputs go to runs/<name>/ (best.pt + config.yaml).
"""
import argparse
from pathlib import Path

import torch

from toygpt.config import load_config, save_config
from toygpt.data import load_wikitext, build_lm_datasets
from toygpt.inference import generate
from toygpt.models import ToyGPT
from toygpt.tokenization import BPETokenizer
from toygpt.training import train

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_tokenizer(raw, train_corpus, out_dir):
    kind = raw.get("tokenizer", "gpt2")
    if kind == "gpt2":
        from transformers import GPT2Tokenizer
        tok = GPT2Tokenizer.from_pretrained("gpt2")
        return tok, tok.vocab_size
    if kind == "bpe":
        tok = BPETokenizer(vocab_size=raw.get("bpe_vocab_size", 4000))
        tok.train(train_corpus)
        tok.save(Path(out_dir) / "tokenizer.json")  # reload later with BPETokenizer.load(...)
        return tok, len(tok.vocab)
    raise ValueError(f"unknown tokenizer: {kind}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    raw, model_cfg, train_cfg = load_config(args.config)
    run_name = raw.get("name", Path(args.config).stem)
    train_cfg.out_dir = str(PROJECT_ROOT / "runs" / run_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"run: {run_name}  device: {device}")

    train_corpus, val_corpus = load_wikitext()
    tokenizer, vocab_size = build_tokenizer(raw, train_corpus, train_cfg.out_dir)
    model_cfg.vocab_size = vocab_size

    save_config(train_cfg.out_dir, model=model_cfg, training=train_cfg,
                tokenizer=raw.get("tokenizer", "gpt2"), bpe_vocab_size=raw.get("bpe_vocab_size"))

    model = ToyGPT(model_cfg)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The model has {total_params:,} trainable parameters.")

    train_dataset, val_dataset = build_lm_datasets(tokenizer, train_corpus, val_corpus, model_cfg.max_seq_len)
    checkpoint = train(model, train_dataset, val_dataset, train_cfg, device)

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    for prompt in ["The history of", "In the United States", "The game began"]:
        print(generate(model, tokenizer, prompt, max_new_tokens=50))
        print()


if __name__ == "__main__":  # guard is required on Windows when num_workers > 0
    main()
