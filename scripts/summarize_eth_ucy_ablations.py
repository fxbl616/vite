"""Summarize ETH/UCY ablation results from STAR log_curve.txt files.

The training log_curve.txt format is:
    epoch,train_loss,ADE,FDE,learning_rate

This script selects the row with the smallest FDE for each dataset/mode and
prints a markdown table plus the key comparison deltas used in the ablation
discussion.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean


DATASETS = ("eth", "hotel", "univ", "zara1", "zara2")
MODES = (
    "adaptive",
    "adaptive_no_motion",
    "adaptive_no_spatial",
    "adaptive_motion_only",
    "vite_only",
    "motion_only",
    "spatial_only",
    # Legacy ART/AIP/full branches kept for comparing against previous runs.
    "full",
    "art_only",
    "art_aip",
    "no_motion_prior",
)


def read_best_row(log_path: Path, min_epoch: int) -> dict[str, float | int] | None:
    """Return the epoch row with the lowest FDE from one log_curve.txt."""
    if not log_path.exists():
        return None

    best = None
    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            parts = raw_line.strip().split(",")
            if len(parts) < 4:
                continue
            try:
                epoch = int(float(parts[0]))
                train_loss = float(parts[1])
                ade = float(parts[2])
                fde = float(parts[3])
            except ValueError:
                continue

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "ade": ade,
                "fde": fde,
            }
            if epoch < min_epoch or ade <= 0.0 or fde <= 0.0:
                continue
            if best is None or fde < best["fde"]:
                best = row
    return best


def collect_results(
    root: Path,
    datasets: tuple[str, ...],
    min_epoch: int,
) -> dict[str, dict[str, dict[str, float | int] | None]]:
    """Collect best rows for each dataset and ablation mode."""
    results = {}
    for dataset in datasets:
        results[dataset] = {}
        for mode in MODES:
            model_name = f"ab_{dataset}_{mode}"
            log_path = root / dataset / model_name / "log_curve.txt"
            results[dataset][mode] = read_best_row(log_path, min_epoch)
    return results


def fmt(value: float | int | None) -> str:
    """Format a numeric result for markdown tables."""
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def build_summary_markdown(
    results: dict[str, dict[str, dict[str, float | int] | None]],
    datasets: tuple[str, ...],
) -> str:
    """Build a compact markdown report for the ablation results."""
    lines: list[str] = []
    lines.append("# ETH/UCY Ablation Summary")
    lines.append("")
    lines.append("Best row is selected by minimum nonzero FDE in `log_curve.txt`.")
    lines.append("")

    for dataset in datasets:
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append("| Mode | Best ADE | Best FDE | Best epoch |")
        lines.append("| --- | ---: | ---: | ---: |")
        for mode in MODES:
            row = results[dataset][mode]
            if row is None:
                lines.append(f"| {mode} | - | - | - |")
            else:
                lines.append(
                    f"| {mode} | {fmt(row['ade'])} | {fmt(row['fde'])} | {fmt(row['epoch'])} |"
                )
        lines.append("")

    lines.append("## Five-scene Average")
    lines.append("")
    lines.append("| Mode | Available scenes | Mean ADE | Mean FDE |")
    lines.append("| --- | ---: | ---: | ---: |")
    for mode in MODES:
        rows = [results[dataset][mode] for dataset in datasets if results[dataset][mode] is not None]
        ades = [float(row["ade"]) for row in rows if row is not None]
        fdes = [float(row["fde"]) for row in rows if row is not None]
        mean_ade = mean(ades) if ades else None
        mean_fde = mean(fdes) if fdes else None
        lines.append(f"| {mode} | {len(rows)} | {fmt(mean_ade)} | {fmt(mean_fde)} |")
    lines.append("")

    lines.append("## Key Deltas")
    lines.append("")
    lines.append("Delta uses FDE: negative means the first method is better.")
    lines.append("")
    lines.append("| Dataset | adaptive - vite_only | adaptive - adaptive_no_motion | adaptive - adaptive_no_spatial | adaptive - motion_only | art_aip - art_only |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for dataset in datasets:
        delta_cells = []
        pairs = (
            ("adaptive", "vite_only"),
            ("adaptive", "adaptive_no_motion"),
            ("adaptive", "adaptive_no_spatial"),
            ("adaptive", "motion_only"),
            ("art_aip", "art_only"),
        )
        for left, right in pairs:
            left_row = results[dataset][left]
            right_row = results[dataset][right]
            if left_row is None or right_row is None:
                delta_cells.append("-")
            else:
                delta_cells.append(fmt(float(left_row["fde"]) - float(right_row["fde"])))
        lines.append(f"| {dataset} | {' | '.join(delta_cells)} |")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize ETH/UCY ablation logs.")
    parser.add_argument("--root", default="output", help="Output root directory.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS),
        help="Datasets to summarize.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional markdown file to write, for example docs/eth_ucy_ablation_summary.md.",
    )
    parser.add_argument(
        "--min-epoch",
        default=10,
        type=int,
        help="Ignore rows before this epoch; early log rows have zero ADE/FDE before testing starts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = tuple(args.datasets)
    results = collect_results(Path(args.root), datasets, args.min_epoch)
    report = build_summary_markdown(results, datasets)
    print(report)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
