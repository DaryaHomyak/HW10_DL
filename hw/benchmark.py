from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml

from hw.constants import CHOICES


def normalize_text(text: str) -> str:
    """Simple normalization for free-form answers."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def parse_mc_answer(text: str, choices: tuple[str, ...] = CHOICES) -> str | None:
    """Extract multiple-choice answer letter from model output."""
    if not text:
        return None
    allowed = "".join(choices)
    pattern = re.compile(rf"(?<![A-Za-z])([{allowed}])(?![A-Za-z])")
    matches = pattern.findall(text)
    if matches:
        return matches[-1].upper()
    return None


def build_benchmark_prompt(question: str, options: list[str]) -> str:
    """Build prompt for multiple-choice visual math evaluation."""
    options_text = "\n".join(options)
    return (
        "Реши визуально-математическую задачу. "
        "Выбери один вариант ответа и в конце напиши только букву.\n\n"
        f"Вопрос: {question}\n"
        f"Варианты:\n{options_text}\n"
        "Ответ:"
    )


def compute_accuracy(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Compute overall and per-subject accuracy from prediction rows."""
    if not rows:
        return {"overall": 0.0}

    total = len(rows)
    correct = sum(int(r.get("prediction") == r.get("answer")) for r in rows)
    metrics = {"overall": correct / total}

    subjects = sorted({r.get("subject", "unknown") for r in rows})
    for subject in subjects:
        sub_rows = [r for r in rows if r.get("subject", "unknown") == subject]
        sub_correct = sum(int(r.get("prediction") == r.get("answer")) for r in sub_rows)
        metrics[f"subject/{subject}"] = sub_correct / max(1, len(sub_rows))
    return metrics


def run_benchmark(config: dict[str, Any], toy: bool = False) -> dict[str, float]:
    """Run evaluation loop."""
    from hw.dataset import MathVQADataset

    data_cfg = config.get("data", {})
    infer_cfg = config.get("inference", {})

    dataset = MathVQADataset(
        data_cfg["eval_manifest"],
        split=data_cfg.get("split", "dev"),
        max_samples=data_cfg.get("max_samples"),
    )

    generator = config.get("_generator")

    rows: list[dict[str, Any]] = []
    for i in range(len(dataset)):
        sample = dataset[i]
        prompt = build_benchmark_prompt(sample.question, sample.options)
        raw_output = generator(prompt, sample) if callable(generator) else ""
        prediction = parse_mc_answer(raw_output)
        rows.append(
            {
                "id": sample.id,
                "subject": sample.subject,
                "answer": sample.answer,
                "prediction": prediction,
            }
        )

    output_path = infer_cfg.get("output_path")
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return compute_accuracy(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--toy", action="store_true")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    metrics = run_benchmark(config, toy=args.toy)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
