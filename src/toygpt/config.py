"""Load experiment configs (YAML) into the dataclasses, and save them next to checkpoints."""
from dataclasses import asdict
from pathlib import Path

import yaml

from toygpt.models.gpt import GPTConfig
from toygpt.training.trainer import TrainingConfig


def load_config(path):
    """Returns (raw_dict, GPTConfig, TrainingConfig). Missing keys fall back to dataclass defaults;
    unknown keys raise a TypeError so typos don't silently do nothing."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    model_cfg = GPTConfig(**raw.get("model", {}))
    train_cfg = TrainingConfig(**raw.get("training", {}))
    return raw, model_cfg, train_cfg


def save_config(out_dir, **sections):
    """e.g. save_config(run_dir, model=model_cfg, training=train_cfg, tokenizer={...})"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {k: (asdict(v) if hasattr(v, "__dataclass_fields__") else v) for k, v in sections.items()}
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
