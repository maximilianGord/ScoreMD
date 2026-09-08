#!/usr/bin/env python3
"""Create a CSV and labelled PNG comparison grid for selected ALDP runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
import yaml


ROOT = Path("outputs/aldp")
COMPARISON_FILE = Path("comparison.json")
CSV_FILE = Path("results.csv")
IMAGE_FILE = Path("comparison.png")
PADDING = 12
LABEL_WIDTH = 220
HEADER_HEIGHT = 58


def value_for_csv(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def loss_settings(config: dict[str, Any]) -> tuple[str, str, str]:
    schedule = config.get("training_schedule", {})
    loss = schedule.get("loss", {})
    if not loss:
        losses = schedule.get("losses", [])
        loss = losses[0].get("loss", {}) if losses else {}
    return (
        str(loss.get("loss_type", "")),
        str(loss.get("tsm_type", "")),
        str(loss.get("sg_type", "")),
    )


def run_details(folder: str) -> tuple[Path, dict[str, str], dict[str, Any]]:
    run_dir = ROOT / folder.strip()
    with (run_dir / ".hydra" / "config.yaml").open() as file:
        config = yaml.safe_load(file) or {}
    loss_type, tsm_type, sg_type = loss_settings(config)
    details = {
        "folder": run_dir.name,
        "run": str(run_dir.relative_to(ROOT)),
        "coarse graining level": str(config.get("dataset", {}).get("coarse_graining_level", "")),
        "loss type": loss_type,
        "tsm type": tsm_type,
        "sg type": sg_type,
    }
    metric_path = next(
        (run_dir / "out" / name for name in ("metric.json", "metrics.json")
         if (run_dir / "out" / name).is_file()),
        None,
    )
    metrics: dict[str, Any] = {}
    if metric_path:
        with metric_path.open() as file:
            metrics = json.load(file)
    return run_dir, details, metrics


def load_image(path: Path, metric: str) -> Image.Image:
    if path.is_file():
        with Image.open(path) as image:
            return image.convert("RGB")
    image = Image.new("RGB", (640, 480), "white")
    draw = ImageDraw.Draw(image)
    draw.text((PADDING, PADDING), f"Missing:\n{metric}", fill="black", font=ImageFont.load_default())
    return image


def draw_text_centered(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, font: ImageFont.ImageFont) -> None:
    left, top, right, bottom = box
    text_box = draw.multiline_textbbox((0, 0), text, font=font, align="center")
    width, height = text_box[2] - text_box[0], text_box[3] - text_box[1]
    draw.multiline_text(
        (left + (right - left - width) // 2, top + (bottom - top - height) // 2),
        text,
        fill="black",
        font=font,
        align="center",
    )


def write_csv(details: list[dict[str, str]], metrics: list[dict[str, Any]]) -> None:
    metric_columns = sorted({key for run_metrics in metrics for key in run_metrics})
    headers = ["folder", "run", "coarse graining level", "loss type", "tsm type", "sg type", *metric_columns]
    with CSV_FILE.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        for run_details, run_metrics in zip(details, metrics):
            writer.writerow(run_details | {key: value_for_csv(run_metrics.get(key, "")) for key in metric_columns})


def write_comparison_image(
    run_dirs: list[Path],
    details: list[dict[str, str]],
    image_metrics: list[str],
    extra_columns: dict[str, str],
) -> None:
    image_names = [metric if metric.endswith(".png") else f"{metric}.png" for metric in image_metrics]
    extra_images = [(title, load_image(Path(path), title)) for title, path in extra_columns.items()]
    run_images = [[load_image(run_dir / "out" / image_name, image_name) for run_dir in run_dirs] for image_name in image_names]
    images = [[image for _, image in extra_images] + row for row in run_images]
    cell_width = max(image.width for row in images for image in row)
    cell_height = max(image.height for row in images for image in row)
    column_count = len(extra_images) + len(run_dirs)
    width = LABEL_WIDTH + column_count * (cell_width + PADDING) + PADDING
    height = HEADER_HEIGHT + len(image_names) * (cell_height + PADDING) + PADDING
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for column, (title, _) in enumerate(extra_images):
        left = LABEL_WIDTH + PADDING + column * (cell_width + PADDING)
        draw_text_centered(draw, (left, 0, left + cell_width, HEADER_HEIGHT), title, font)

    for run_column, detail in enumerate(details, start=len(extra_images)):
        left = LABEL_WIDTH + PADDING + run_column * (cell_width + PADDING)
        subtitle = " | ".join((
            f"loss={detail['loss type']}",
            f"cg={detail['coarse graining level']}",
            f"tsm={detail['tsm type']}",
            f"sg={detail['sg type']}",
        ))
        draw_text_centered(draw, (left, 0, left + cell_width, HEADER_HEIGHT), f"{detail['folder']}\n{subtitle}", font)

    for row, (metric, image_row) in enumerate(zip(image_metrics, images)):
        top = HEADER_HEIGHT + row * (cell_height + PADDING)
        draw_text_centered(draw, (0, top, LABEL_WIDTH, top + cell_height), metric, font)
        for column, image in enumerate(image_row):
            left = LABEL_WIDTH + PADDING + column * (cell_width + PADDING)
            canvas.paste(image, (left + (cell_width - image.width) // 2, top + (cell_height - image.height) // 2))

    canvas.save(IMAGE_FILE)


def main() -> None:
    with COMPARISON_FILE.open() as file:
        comparison = json.load(file)
    folders = comparison["folders"]
    image_metrics = comparison["metrics"]
    extra_columns = comparison.get("extra_col", {})

    csv_run_data = [
        run_details(str(run_dir.relative_to(ROOT)))
        for run_dir in sorted(ROOT.glob("*/*"))
        if run_dir.is_dir() and (run_dir / "out").is_dir() and any((run_dir / "out").iterdir())
    ]
    _, csv_details, csv_metrics = map(list, zip(*csv_run_data))
    write_csv(csv_details, csv_metrics)

    if folders and image_metrics:
        image_run_data = [run_details(folder) for folder in folders]
        image_run_dirs, image_details, _ = map(list, zip(*image_run_data))
        write_comparison_image(image_run_dirs, image_details, image_metrics, extra_columns)


if __name__ == "__main__":
    main()
