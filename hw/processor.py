from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

from hw.constants import IGNORE_INDEX, IMAGE_END_TOKEN, IMAGE_START_TOKEN, IMAGE_TOKEN
from hw.dataset import MathVQASample


@dataclass
class ProcessorConfig:
    image_size: int = 224
    num_tiles: int = 1
    tile_overlap: float = 0.0
    num_image_tokens: int = 49
    max_length: int = 512
    ignore_index: int = IGNORE_INDEX


class MathVLMProcessor:
    """Builds model inputs from MathVQASample.

    The processor owns all text/image preprocessing that must be deterministic
    across train and inference.
    """

    def __init__(self, tokenizer: Any, config: ProcessorConfig | None = None) -> None:
        self.tokenizer = tokenizer
        self.config = config or ProcessorConfig()

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Convert image to tensor with shape [num_tiles, 3, image_size, image_size]."""
        size = self.config.image_size
        image = image.convert("RGB").resize((size, size), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        tiles = tensor.unsqueeze(0).repeat(self.config.num_tiles, 1, 1, 1)
        return tiles.contiguous()

    def build_prompt(self, sample: MathVQASample, include_answer: bool) -> str:
        image_block = (
            f"{IMAGE_START_TOKEN}"
            + IMAGE_TOKEN * self.config.num_image_tokens
            + f"{IMAGE_END_TOKEN}"
        )
        options_text = "\n".join(sample.options)
        prompt = (
            f"{image_block}\n"
            "Реши визуально-математическую задачу. "
            "Выбери один вариант и напиши только букву.\n"
            f"Вопрос: {sample.question}\n"
            f"Варианты:\n{options_text}\n"
            "Ответ:"
        )
        if include_answer:
            prompt = f"{prompt} {sample.answer}"
        return prompt

    def tokenize_sample(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        prompt = self.build_prompt(sample, include_answer=False)
        full = self.build_prompt(sample, include_answer=True)

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        full_ids = self.tokenizer.encode(full, add_special_tokens=True)

        max_length = self.config.max_length
        full_ids = full_ids[:max_length]
        prompt_len = min(len(prompt_ids), len(full_ids))

        labels = list(full_ids)
        for i in range(prompt_len):
            labels[i] = self.config.ignore_index

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        attention_mask = torch.ones(len(full_ids), dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels_tensor,
        }

    def __call__(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        item = self.tokenize_sample(sample)
        item["pixel_values"] = self.preprocess_image(sample.image)
        return item

    def collate(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
        max_len = max(item["input_ids"].shape[0] for item in batch)

        input_ids = []
        attention_mask = []
        labels = []
        for item in batch:
            length = item["input_ids"].shape[0]
            pad = max_len - length
            input_ids.append(
                torch.cat([item["input_ids"], torch.full((pad,), pad_id, dtype=torch.long)])
            )
            attention_mask.append(
                torch.cat([item["attention_mask"], torch.zeros(pad, dtype=torch.long)])
            )
            labels.append(
                torch.cat(
                    [item["labels"], torch.full((pad,), self.config.ignore_index, dtype=torch.long)]
                )
            )

        out: dict[str, torch.Tensor] = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }
        if all("pixel_values" in item for item in batch):
            out["pixel_values"] = torch.stack([item["pixel_values"] for item in batch])
        return out
    def __call__(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        item = self.tokenize_sample(sample)
        item["pixel_values"] = self.preprocess_image(sample.image)
        return item

    def collate(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
        max_len = max(item["input_ids"].shape[0] for item in batch)

        input_ids = []
        attention_mask = []
        labels = []
        for item in batch:
            length = item["input_ids"].shape[0]
            pad = max_len - length
            input_ids.append(
                torch.cat([item["input_ids"], torch.full((pad,), pad_id, dtype=torch.long)])
            )
            attention_mask.append(
                torch.cat([item["attention_mask"], torch.zeros(pad, dtype=torch.long)])
            )
            labels.append(
                torch.cat(
                    [item["labels"], torch.full((pad,), self.config.ignore_index, dtype=torch.long)]
                )
            )

        out: dict[str, torch.Tensor] = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }
        if all("pixel_values" in item for item in batch):
            out["pixel_values"] = torch.stack([item["pixel_values"] for item in batch])
        return out
