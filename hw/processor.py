from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

from hw.constants import IMAGE_END_TOKEN, IMAGE_START_TOKEN, IMAGE_TOKEN, IGNORE_INDEX
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
        """Convert image to tensor with shape [num_tiles, 3, image_size, image_size].

        TODO:
            - convert to RGB;
            - resize/crop/pad;
            - split into tiles if num_tiles > 1;
            - normalize to float tensor.
        """
        size = self.config.image_size
        image = image.convert("RGB").resize((size, size), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        tiles = tensor.unsqueeze(0).repeat(self.config.num_tiles, 1, 1, 1)
        return tiles.contiguous()

    def build_prompt(self, sample: MathVQASample, include_answer: bool) -> str:
        """Build a text prompt with visual special tokens and options.

        For training, include_answer=True should append the assistant answer.
        For inference, include_answer=False should stop before the answer.
        """
        raise NotImplementedError("Implement prompt construction")

    def tokenize_sample(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        """Return input_ids, attention_mask and labels for one sample.

        labels must be IGNORE_INDEX for prompt tokens and real token ids only
        for the assistant answer.
        """
        raise NotImplementedError("Implement sample tokenization")

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
