#!/usr/bin/env python3
"""Create CSV exports and labelled comparison grids for ALDP and Mueller–Brown runs."""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
import yaml


ALDP_ROOT = Path("outputs/aldp")
MUELLER_BROWN_ROOT = Path("outputs/mueller_brown")
COMPARISON_FILE = Path("comparison.json")
ALDP_CSV_FILE = Path("results.csv")
MUELLER_BROWN_CSV_FILE = Path("mueller_brown_results.csv")
ALDP_IMAGE_FILE = Path("comparison_aldp.png")
MUELLER_BROWN_IMAGE_FILE = Path("comparison_mueller_brown.png")
PADDING = 12
LABEL_WIDTH = 220
HEADER_HEIGHT = 58
METRIC_HEIGHT = 30
ALDP_LANGEVIN_JS_DIVERGENCE = "eval/aldp_langevin_js_divergence"


def loss_options(config: dict[str, Any]) -> dict[str, Any]:
    schedule = config.get("training_schedule", {})
    loss = schedule.get("loss", {})
    if not loss:
        losses = schedule.get("losses", [])
        loss = losses[0].get("loss", {}) if losses else {}
    return loss


def value_for_csv(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def loss_settings(config: dict[str, Any]) -> tuple[str, str, str]:
    loss = loss_options(config)
    return (
        str(loss.get("loss_type", "")),
        str(loss.get("tsm_type", "")),
        str(loss.get("sg_type", "")),
    )


def computed_mode_sigma(run_dir: Path) -> str | None:
    """Return the normalized mode variance used at runtime, when it was logged."""
    mode_pattern = re.compile(r"normalized sigma_mode_sq=([^;\s]+)")
    log_paths = [run_dir / "train.log"]
    for log_path in log_paths:
        if log_path.is_file():
            match = mode_pattern.search(log_path.read_text(errors="replace"))
            if match:
                return match.group(1)

    # Some older runs wrote to the repository-level train.log.  Match the
    # mode computation immediately after this run's date/time directory.
    try:
        run_started = datetime.strptime(
            f"{run_dir.parent.name} {run_dir.name}", "%Y-%m-%d %H-%M-%S"
        )
    except ValueError:
        return None
    timestamp_pattern = re.compile(
        r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+\].*?"
        r"normalized sigma_mode_sq=([^;\s]+)"
    )
    repository_log = Path("train.log")
    if repository_log.is_file():
        for match in timestamp_pattern.finditer(repository_log.read_text(errors="replace")):
            logged_at = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
            if run_started <= logged_at <= run_started + timedelta(minutes=15):
                return match.group(2)
    return None


def plot_label(run_dir: Path, config: dict[str, Any], run: str) -> str:
    loss = loss_options(config)
    configured_loss_type = str(loss.get("loss_type", ""))
    alpha = float(loss.get("alpha", 0.0) or 0.0)
    beta = float(loss.get("beta", 0.0) or 0.0)
    loss_type = "fp" if configured_loss_type == "dsm" and (alpha > 0 or beta > 0) else configured_loss_type
    lines = [run, f"loss_type={loss_type}"]

    if configured_loss_type == "tsm":
        lines.extend(
            f"{key}={loss.get(key, '')}"
            for key in ("tsm_type", "tsm_lambda", "tsm_sigma_max")
        )
        matching_type = str(loss.get("tsm_type", ""))
    elif configured_loss_type == "sc":
        lines.extend(
            f"{key}={loss.get(key, '')}"
            for key in ("sg_type", "sg_lambda", "sg_sigma_max")
        )
        matching_type = str(loss.get("sg_type", ""))
    else:
        matching_type = ""

    if matching_type == "mode_mixture":
        lines.append(
            "mode_var_computation="
            f"{config.get('dataset', {}).get('mode_var_computation', 'data_hessian')}"
        )
        if sigma_mode_sq := computed_mode_sigma(run_dir):
            lines.append(f"computed sigma_mode_sq={sigma_mode_sq}")
    return "\n".join(lines)


def run_details(root: Path, folder: str) -> tuple[Path, dict[str, str], dict[str, Any]]:
    run_dir = root / folder.strip()
    with (run_dir / ".hydra" / "config.yaml").open() as file:
        config = yaml.safe_load(file) or {}
    loss_type, tsm_type, sg_type = loss_settings(config)
    evaluation = config.get("evaluation", {})
    details = {
        "folder": run_dir.name,
        "run": str(run_dir.relative_to(root)),
        "coarse graining level": str(config.get("dataset", {}).get("coarse_graining_level", "")),
        "loss type": loss_type,
        "tsm type": tsm_type,
        "sg type": sg_type,
        "num langevin samples": str(evaluation.get("num_langevin_samples", "")),
        "parallel trajectories": str(evaluation.get("num_parallel_langevin_samples", "")),
        "_plot_label": plot_label(run_dir, config, str(run_dir.relative_to(root))),
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


def write_csv(
    csv_file: Path,
    details: list[dict[str, str]],
    metrics: list[dict[str, Any]],
) -> None:
    metric_columns = sorted({key for run_metrics in metrics for key in run_metrics})
    headers = [
        "folder",
        "run",
        "coarse graining level",
        "loss type",
        "tsm type",
        "sg type",
        "num langevin samples",
        "parallel trajectories",
        *metric_columns,
    ]
    with csv_file.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        for run_details, run_metrics in zip(details, metrics):
            row = run_details | {key: value_for_csv(run_metrics.get(key, "")) for key in metric_columns}
            writer.writerow({header: row.get(header, "") for header in headers})


def write_comparison_image(
    image_file: Path,
    run_dirs: list[Path],
    details: list[dict[str, str]],
    metrics: list[dict[str, Any]],
    image_metrics: list[str],
    extra_columns: dict[str, str],
    summary_metric: str | None = None,
) -> None:
    image_names = [metric if metric.endswith(".png") else f"{metric}.png" for metric in image_metrics]
    extra_images = [(title, load_image(Path(path), title)) for title, path in extra_columns.items()]
    run_images = [[load_image(run_dir / "out" / image_name, image_name) for run_dir in run_dirs] for image_name in image_names]
    images = [[image for _, image in extra_images] + row for row in run_images]
    cell_width = max(image.width for row in images for image in row)
    cell_height = max(image.height for row in images for image in row)
    column_count = len(extra_images) + len(run_dirs)
    header_height = max(
        HEADER_HEIGHT,
        max(
            ImageDraw.Draw(Image.new("RGB", (1, 1))).multiline_textbbox(
                (0, 0), detail["_plot_label"], font=ImageFont.load_default(), align="center"
            )[3]
            for detail in details
        ) + 2 * PADDING,
    )
    width = LABEL_WIDTH + column_count * (cell_width + PADDING) + PADDING
    metric_height = METRIC_HEIGHT if summary_metric else 0
    height = header_height + len(image_names) * (cell_height + PADDING) + metric_height + PADDING
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for column, (title, _) in enumerate(extra_images):
        left = LABEL_WIDTH + PADDING + column * (cell_width + PADDING)
        draw_text_centered(draw, (left, 0, left + cell_width, HEADER_HEIGHT), title, font)

    for run_column, detail in enumerate(details, start=len(extra_images)):
        left = LABEL_WIDTH + PADDING + run_column * (cell_width + PADDING)
        draw_text_centered(draw, (left, 0, left + cell_width, header_height), detail["_plot_label"], font)

    for row, (metric, image_row) in enumerate(zip(image_metrics, images)):
        top = header_height + row * (cell_height + PADDING)
        draw_text_centered(draw, (0, top, LABEL_WIDTH, top + cell_height), metric, font)
        for column, image in enumerate(image_row):
            left = LABEL_WIDTH + PADDING + column * (cell_width + PADDING)
            canvas.paste(image, (left + (cell_width - image.width) // 2, top + (cell_height - image.height) // 2))

    if summary_metric:
        top = header_height + len(image_names) * (cell_height + PADDING)
        metric_label = summary_metric.removeprefix("eval/")
        draw_text_centered(draw, (0, top, LABEL_WIDTH, top + metric_height), metric_label, font)
        for run_column, run_metrics in enumerate(metrics, start=len(extra_images)):
            left = LABEL_WIDTH + PADDING + run_column * (cell_width + PADDING)
            value = value_for_csv(run_metrics.get(summary_metric, ""))
            draw_text_centered(draw, (left, top, left + cell_width, top + metric_height), value, font)

    canvas.save(image_file)


def main() -> None:
    with COMPARISON_FILE.open() as file:
        comparison = json.load(file)
    csv_run_data = [
        run_details(ALDP_ROOT, str(run_dir.relative_to(ALDP_ROOT)))
        for run_dir in sorted(ALDP_ROOT.glob("*/*"))
        if run_dir.is_dir() and (run_dir / "out").is_dir() and any((run_dir / "out").iterdir())
    ]
    _, csv_details, csv_metrics = map(list, zip(*csv_run_data))
    write_csv(ALDP_CSV_FILE, csv_details, csv_metrics)

    mueller_brown_csv_run_data = [
        run_details(MUELLER_BROWN_ROOT, str(run_dir.relative_to(MUELLER_BROWN_ROOT)))
        for run_dir in sorted(MUELLER_BROWN_ROOT.glob("*/*"))
        if run_dir.is_dir() and (run_dir / "out").is_dir() and any((run_dir / "out").iterdir())
    ]
    if mueller_brown_csv_run_data:
        _, mueller_brown_csv_details, mueller_brown_csv_metrics = map(
            list, zip(*mueller_brown_csv_run_data)
        )
        write_csv(
            MUELLER_BROWN_CSV_FILE,
            mueller_brown_csv_details,
            mueller_brown_csv_metrics,
        )
    else:
        print("Skipping Mueller-Brown CSV: no completed runs found.")

    image_comparisons = [
        (ALDP_ROOT, ALDP_IMAGE_FILE, comparison["aldp"], ALDP_LANGEVIN_JS_DIVERGENCE),
    ]
    if mueller_brown_csv_run_data and (
        mueller_brown_comparison := comparison.get("mueller_brown")
    ):
        image_comparisons.append(
            (MUELLER_BROWN_ROOT, MUELLER_BROWN_IMAGE_FILE, mueller_brown_comparison, None)
        )

    for root, image_file, image_comparison, summary_metric in image_comparisons:
        folders = image_comparison["folders"]
        image_metrics = image_comparison["metrics"]
        extra_columns = image_comparison.get("extra_col", {})
        if folders and image_metrics:
            image_run_data = [run_details(root, folder) for folder in folders]
            image_run_dirs, image_details, image_run_metrics = map(list, zip(*image_run_data))
            write_comparison_image(
                image_file,
                image_run_dirs,
                image_details,
                image_run_metrics,
                image_metrics,
                extra_columns,
                summary_metric,
            )


if __name__ == "__main__":
    main()
