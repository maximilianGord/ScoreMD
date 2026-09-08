#!/usr/bin/env python3
"""Print a Markdown summary of completed ALDP evaluation runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path("outputs/aldp")


def value_for_table(value: Any) -> str:
    """Make metric values safe to place in one Markdown table cell."""
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


def main() -> None:
    rows: list[tuple[str, str, str, str, str, dict[str, Any]]] = []
    metric_keys: set[str] = set()

    for run_dir in sorted(path for path in ROOT.glob("*/*") if path.is_dir()):
        out_dir = run_dir / "out"
        if not out_dir.is_dir() or not any(out_dir.iterdir()):
            continue

        metric_path = next(
            (path for name in ("metric.json", "metrics.json")
             if (path := out_dir / name).is_file()),
            None,
        )
        metrics = {}
        if metric_path:
            with metric_path.open() as file:
                metrics = json.load(file)
        with (run_dir / ".hydra" / "config.yaml").open() as file:
            config = yaml.safe_load(file) or {}

        loss_type, tsm_type, sg_type = loss_settings(config)
        coarse_graining_level = str(config.get("dataset", {}).get("coarse_graining_level", ""))
        metric_keys.update(metrics)
        rows.append((run_dir.name, str(run_dir.relative_to(ROOT)), coarse_graining_level, loss_type, tsm_type, sg_type, metrics))

    metric_columns = sorted(metric_keys)
    headers = ["folder", "run", "coarse graining level", "loss type", "tsm type", "sg type", *metric_columns]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for folder, run, coarse_graining_level, loss_type, tsm_type, sg_type, metrics in rows:
        cells = [folder, run, coarse_graining_level, loss_type, tsm_type, sg_type, *(value_for_table(metrics.get(key, "")) for key in metric_columns)]
        print("| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |")


if __name__ == "__main__":
    main()
