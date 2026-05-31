from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _extract_loss(output: Any) -> torch.Tensor:
    if isinstance(output, dict):
        return output["loss"]
    if hasattr(output, "loss"):
        return output.loss
    return output


def train_one_step(
    model: torch.nn.Module, batch: dict[str, torch.Tensor], optimizer: torch.optim.Optimizer
) -> float:
    """Run one optimization step and return scalar loss."""
    model.train()
    output = model(batch)
    loss = _extract_loss(output)
    if not torch.isfinite(loss):
        raise ValueError("Loss is not finite")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())


def run_training(config: dict[str, Any], fast_train: bool = False) -> None:
    """Main training entry point."""
    from torch.utils.data import DataLoader

    from hw.dataset import MathVQADataset
    from hw.processor import MathVLMProcessor, ProcessorConfig

    data_cfg = config.get("data", {})
    proc_cfg = config.get("processor", {})
    trainer_cfg = config.get("trainer", {})

    device = torch.device(trainer_cfg.get("device", "cpu"))
    max_steps = int(trainer_cfg.get("max_steps", 3))
    if fast_train:
        max_steps = min(max_steps, 3)
    batch_size = int(trainer_cfg.get("local_batch_size", 1))
    lr = float(trainer_cfg.get("learning_rate", 5e-4))

    dataset = MathVQADataset(
        data_cfg["train_manifest"],
        split=data_cfg.get("split", "train"),
        max_samples=data_cfg.get("max_samples"),
    )

    processor = MathVLMProcessor(_FallbackTokenizer(), ProcessorConfig(**proc_cfg))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(trainer_cfg.get("num_workers", 0)),
        collate_fn=lambda batch: processor.collate([processor(s) for s in batch]),
    )

    model = _SmokeModel(processor.tokenizer.vocab_size, proc_cfg.get("num_image_tokens", 16)).to(
        device
    )
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
        weight_decay=float(trainer_cfg.get("weight_decay", 0.0)),
    )

    step = 0
    losses: list[float] = []
    while step < max_steps:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = train_one_step(model, batch, optimizer)
            losses.append(loss)
            step += 1
            print(f"step {step}/{max_steps} loss {loss:.4f}")
            if step >= max_steps:
                break

    save_path = trainer_cfg.get("save_checkpoint_path")
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
        print(f"saved checkpoint to {path}")


class _FallbackTokenizer:
    """Minimal hashing tokenizer used for CPU smoke-training without external weights."""

    pad_token_id = 0
    eos_token_id = 1
    vocab_size = 4096

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = [2 + (hash(tok) % (self.vocab_size - 2)) for tok in text.replace("\n", " ").split()]
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids


class _SmokeModel(torch.nn.Module):
    """Tiny language-only model for the CPU smoke loop."""

    def __init__(self, vocab_size: int, num_image_tokens: int) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, 32)
        self.head = torch.nn.Linear(32, vocab_size)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        ids = batch["input_ids"]
        logits = self.head(self.embed(ids))
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            batch["labels"][:, 1:].reshape(-1),
            ignore_index=-100,
        )
        return {"loss": loss}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fast-train", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    run_training(config, fast_train=args.fast_train)


if __name__ == "__main__":
    main()
