#!/usr/bin/env python3
"""Analyze tau2 evaluation results with Pass^k and within-task rewards.

The tau2 definition used here is:

    Pass^k(task) = C(successes, k) / C(trials, k)

It estimates the probability that all k sampled trajectories for a task
succeed. The reported dataset-level Pass^k is the mean over tasks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


TRAINING_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = TRAINING_DIR / "data/eval"
DEFAULT_OUTPUT_DIR = TRAINING_DIR / "results/eval_passk"
INFRASTRUCTURE_ERROR = "infrastructure_error"

BACKGROUND = "#FFFFFF"
TEXT = "#172033"
MUTED_TEXT = "#5D6678"
GRID = "#DCE1EA"
FRAME = "#AEB6C5"
MODEL_COLORS = ("#3B82F6", "#F59E0B", "#8B5CF6", "#14B8A6")
GROUP_COLORS = {
    "all_failure": "#E56B9F",
    "mixed": "#53B175",
    "all_success": "#3B82F6",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute tau2 Pass^1..Pass^4 and within-task reward statistics "
            "from one or more results.json files."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help=(
            "Input results.json files. When omitted, all JSON files directly "
            f"under {DEFAULT_INPUT_DIR} are analyzed."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for JSON and CSV reports (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--expected-trials",
        type=int,
        default=4,
        help="Required number of non-infrastructure trials per task (default: 4)",
    )
    return parser.parse_args()


def is_successful(reward: float) -> bool:
    """Match tau2's reward==1 success criterion with floating tolerance."""

    return (1 - 1e-6) <= reward <= (1 + 1e-6)


def pass_hat_k(num_trials: int, success_count: int, k: int) -> float:
    """Compute tau2 Pass^k for one task."""

    if num_trials < k:
        raise ValueError(f"Number of trials {num_trials} is less than k {k}")
    return math.comb(success_count, k) / math.comb(num_trials, k)


def task_sort_key(task_id: str) -> tuple[int, int | str]:
    try:
        return (0, int(task_id))
    except ValueError:
        return (1, task_id)


def require_mapping(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Expected {description} to be a JSON object")
    return value


def require_list(value: Any, description: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"Expected {description} to be a JSON array")
    return value


def extract_reward(simulation: dict[str, Any], index: int) -> float:
    reward_info = require_mapping(
        simulation.get("reward_info"), f"simulations[{index}].reward_info"
    )
    reward = reward_info.get("reward")
    if not isinstance(reward, (int, float)):
        raise ValueError(f"simulations[{index}] has a non-numeric reward: {reward!r}")
    return float(reward)


def analyze_results(
    path: Path, expected_trials: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if expected_trials < 1:
        raise ValueError("expected_trials must be positive")

    data = require_mapping(json.loads(path.read_text(encoding="utf-8")), str(path))
    info = require_mapping(data.get("info"), f"{path}: info")
    simulations = require_list(data.get("simulations"), f"{path}: simulations")

    configured_trials = info.get("num_trials")
    if configured_trials != expected_trials:
        raise ValueError(
            f"{path}: info.num_trials={configured_trials!r}, expected {expected_trials}"
        )

    infrastructure_errors = [
        simulation
        for simulation in simulations
        if simulation.get("termination_reason") == INFRASTRUCTURE_ERROR
    ]
    if infrastructure_errors:
        raise ValueError(
            f"{path}: found {len(infrastructure_errors)} infrastructure_error "
            "simulation(s). Resume/retry them before computing a complete Pass^1..4 report."
        )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    termination_counts: Counter[str] = Counter()
    trial_reward_totals: dict[int, list[float]] = defaultdict(list)

    for index, raw_simulation in enumerate(simulations):
        simulation = require_mapping(raw_simulation, f"simulations[{index}]")
        if simulation.get("task_id") is None:
            raise ValueError(f"{path}: simulations[{index}] is missing task_id")
        task_id = str(simulation["task_id"])
        trial = simulation.get("trial")
        seed = simulation.get("seed")
        if not isinstance(trial, int):
            raise ValueError(f"{path}: task {task_id} has invalid trial {trial!r}")
        reward = extract_reward(simulation, index)
        if not math.isfinite(reward):
            raise ValueError(f"{path}: task {task_id}, trial {trial} has non-finite reward")

        groups[task_id].append(
            {
                "trial": trial,
                "seed": seed,
                "reward": reward,
                "success": is_successful(reward),
                "termination_reason": str(simulation.get("termination_reason")),
            }
        )
        termination_counts[str(simulation.get("termination_reason"))] += 1
        trial_reward_totals[trial].append(reward)

    if not groups:
        raise ValueError(f"{path}: no simulations found")

    task_metadata = data.get("tasks")
    if isinstance(task_metadata, list) and len(task_metadata) != len(groups):
        raise ValueError(
            f"{path}: results contain {len(groups)} task groups but {len(task_metadata)} "
            "task metadata entries"
        )

    per_task: list[dict[str, Any]] = []
    duplicate_trials: list[str] = []
    incomplete_tasks: list[str] = []

    for task_id in sorted(groups, key=task_sort_key):
        group = sorted(groups[task_id], key=lambda item: item["trial"])
        trials = [item["trial"] for item in group]
        if len(set(trials)) != len(trials):
            duplicate_trials.append(task_id)
        if len(group) != expected_trials or set(trials) != set(range(expected_trials)):
            incomplete_tasks.append(task_id)

        rewards = [item["reward"] for item in group]
        success_count = sum(item["success"] for item in group)
        reward_mean = statistics.fmean(rewards)
        reward_variance = statistics.pvariance(rewards)
        reward_std = math.sqrt(reward_variance)

        if success_count == 0:
            group_type = "all_failure"
        elif success_count == len(group):
            group_type = "all_success"
        else:
            group_type = "mixed"

        row: dict[str, Any] = {
            "task_id": task_id,
            "num_trials": len(group),
            "success_count": success_count,
            "failure_count": len(group) - success_count,
            "reward_mean": reward_mean,
            "reward_variance": reward_variance,
            "reward_std": reward_std,
            "zero_variance": math.isclose(reward_variance, 0.0, abs_tol=1e-12),
            "group_type": group_type,
        }
        for k in range(1, expected_trials + 1):
            row[f"pass_{k}"] = pass_hat_k(len(group), success_count, k)
        for item in group:
            trial = item["trial"]
            row[f"trial_{trial}_seed"] = item["seed"]
            row[f"trial_{trial}_reward"] = item["reward"]
            row[f"trial_{trial}_success"] = item["success"]
            row[f"trial_{trial}_termination"] = item["termination_reason"]
        per_task.append(row)

    if duplicate_trials:
        raise ValueError(f"{path}: duplicate trial IDs for tasks {duplicate_trials}")
    if incomplete_tasks:
        raise ValueError(
            f"{path}: tasks do not contain trials 0..{expected_trials - 1}: "
            f"{incomplete_tasks}"
        )

    group_type_counts = Counter(row["group_type"] for row in per_task)
    success_count_distribution = Counter(row["success_count"] for row in per_task)
    total_tasks = len(per_task)
    total_simulations = len(simulations)
    total_successes = sum(row["success_count"] for row in per_task)

    agent_info = require_mapping(info.get("agent_info"), f"{path}: info.agent_info")
    user_info = require_mapping(info.get("user_info"), f"{path}: info.user_info")
    model = str(agent_info.get("llm", "unknown"))

    summary: dict[str, Any] = {
        "input_file": str(path.resolve()),
        "timestamp": data.get("timestamp"),
        "model": model,
        "user_model": user_info.get("llm"),
        "num_tasks": total_tasks,
        "expected_trials": expected_trials,
        "total_simulations": total_simulations,
        "total_successes": total_successes,
        "total_failures": total_simulations - total_successes,
        "average_reward": statistics.fmean(
            item["reward"] for group in groups.values() for item in group
        ),
        "pass_hat_k": {
            str(k): statistics.fmean(row[f"pass_{k}"] for row in per_task)
            for k in range(1, expected_trials + 1)
        },
        "group_rewards": {
            "all_failure_tasks": group_type_counts["all_failure"],
            "all_failure_rate": group_type_counts["all_failure"] / total_tasks,
            "all_failure_task_ids": [
                row["task_id"]
                for row in per_task
                if row["group_type"] == "all_failure"
            ],
            "mixed_tasks": group_type_counts["mixed"],
            "mixed_rate": group_type_counts["mixed"] / total_tasks,
            "mixed_task_ids": [
                row["task_id"]
                for row in per_task
                if row["group_type"] == "mixed"
            ],
            "all_success_tasks": group_type_counts["all_success"],
            "all_success_rate": group_type_counts["all_success"] / total_tasks,
            "all_success_task_ids": [
                row["task_id"]
                for row in per_task
                if row["group_type"] == "all_success"
            ],
            "zero_variance_tasks": sum(row["zero_variance"] for row in per_task),
            "zero_variance_rate": sum(row["zero_variance"] for row in per_task)
            / total_tasks,
            "zero_variance_task_ids": [
                row["task_id"] for row in per_task if row["zero_variance"]
            ],
            "mean_task_reward_variance": statistics.fmean(
                row["reward_variance"] for row in per_task
            ),
            "mean_task_reward_std": statistics.fmean(
                row["reward_std"] for row in per_task
            ),
            "success_count_distribution": {
                str(success_count): success_count_distribution[success_count]
                for success_count in range(expected_trials + 1)
            },
        },
        "per_trial": {
            str(trial): {
                "num_simulations": len(rewards),
                "average_reward": statistics.fmean(rewards),
                "success_count": sum(is_successful(reward) for reward in rewards),
            }
            for trial, rewards in sorted(trial_reward_totals.items())
        },
        "termination_counts": dict(sorted(termination_counts.items())),
    }
    return summary, per_task


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV report to {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a portable sans-serif font available on Linux or macOS."""

    candidates = (
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def model_display_name(model: str) -> str:
    if "Instruct" in model:
        return "Qwen3-4B-Instruct-2507"
    if "Qwen3-4B" in model:
        return "Qwen3-4B (non-thinking)"
    return model.rsplit("/", 1)[-1]


def draw_chart_header(
    draw: ImageDraw.ImageDraw,
    title: str,
    subtitle: str,
    width: int,
) -> None:
    draw.text((48, 30), title, font=load_font(32, bold=True), fill=TEXT)
    draw.text((48, 76), subtitle, font=load_font(17), fill=MUTED_TEXT)
    draw.line((48, 108, width - 48, 108), fill=GRID, width=1)


def draw_model_legend(
    draw: ImageDraw.ImageDraw,
    summaries: list[dict[str, Any]],
    x: int,
    y: int,
) -> None:
    cursor = x
    font = load_font(16)
    for index, summary in enumerate(summaries):
        color = MODEL_COLORS[index % len(MODEL_COLORS)]
        draw.ellipse((cursor, y + 3, cursor + 13, y + 16), fill=color)
        label = model_display_name(summary["model"])
        draw.text((cursor + 21, y), label, font=font, fill=TEXT)
        cursor += 31 + int(draw.textlength(label, font=font))


def make_passk_chart(
    summaries: list[dict[str, Any]], width: int = 1200, height: int = 720
) -> Image.Image:
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw_chart_header(
        draw,
        "Pass^k reliability",
        "Higher k requires all sampled trajectories to succeed",
        width,
    )
    draw_model_legend(draw, summaries, 70, 125)

    left, top, right, bottom = 105, 175, width - 48, height - 90
    all_values = [
        float(value) * 100
        for summary in summaries
        for value in summary["pass_hat_k"].values()
    ]
    y_max = max(10, math.ceil(max(all_values) * 1.15 / 10) * 10)
    num_k = max(len(summary["pass_hat_k"]) for summary in summaries)

    for tick in range(0, int(y_max) + 1, 10):
        y = bottom - (tick / y_max) * (bottom - top)
        draw.line((left, y, right, y), fill=GRID, width=1)
        draw.text(
            (left - 14, y),
            f"{tick}%",
            font=load_font(15),
            fill=MUTED_TEXT,
            anchor="rm",
        )
    draw.rectangle((left, top, right, bottom), outline=FRAME, width=1)

    x_positions = [
        left + (right - left) * index / max(1, num_k - 1)
        for index in range(num_k)
    ]
    for index, x in enumerate(x_positions, start=1):
        draw.text(
            (x, bottom + 23),
            f"Pass^{index}",
            font=load_font(16),
            fill=TEXT,
            anchor="mm",
        )

    for model_index, summary in enumerate(summaries):
        color = MODEL_COLORS[model_index % len(MODEL_COLORS)]
        values = [float(summary["pass_hat_k"][str(k)]) * 100 for k in range(1, num_k + 1)]
        points = [
            (x, bottom - (value / y_max) * (bottom - top))
            for x, value in zip(x_positions, values, strict=True)
        ]
        draw.line(points, fill=color, width=5, joint="curve")
        for x, y in points:
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color, outline=BACKGROUND, width=2)
        for point_index, ((x, y), value) in enumerate(zip(points, values, strict=True)):
            offset = -18 if model_index % 2 == 0 else 19
            draw.text(
                (x, y + offset),
                f"{value:.1f}%" if value % 1 else f"{value:.0f}%",
                font=load_font(15, bold=True),
                fill=TEXT,
                anchor="mm",
            )

    draw.text(
        (left + (right - left) / 2, height - 34),
        "Number of jointly successful samples (k)",
        font=load_font(16),
        fill=MUTED_TEXT,
        anchor="mm",
    )
    return image


def make_success_distribution_chart(
    summaries: list[dict[str, Any]], width: int = 1200, height: int = 720
) -> Image.Image:
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    expected_trials = max(int(summary["expected_trials"]) for summary in summaries)
    draw_chart_header(
        draw,
        "Within-task success-count distribution",
        f"Number of tasks succeeding 0 to {expected_trials} times across {expected_trials} trials",
        width,
    )
    draw_model_legend(draw, summaries, 70, 125)

    left, top, right, bottom = 92, 175, width - 48, height - 90
    distributions = [
        [
            int(summary["group_rewards"]["success_count_distribution"][str(count)])
            for count in range(expected_trials + 1)
        ]
        for summary in summaries
    ]
    maximum = max(value for distribution in distributions for value in distribution)
    y_max = max(5, math.ceil(maximum * 1.2 / 5) * 5)

    for tick in range(0, y_max + 1, 5):
        y = bottom - (tick / y_max) * (bottom - top)
        draw.line((left, y, right, y), fill=GRID, width=1)
        draw.text((left - 13, y), str(tick), font=load_font(15), fill=MUTED_TEXT, anchor="rm")
    draw.rectangle((left, top, right, bottom), outline=FRAME, width=1)

    group_width = (right - left) / (expected_trials + 1)
    bar_gap = max(4, int(group_width * 0.04))
    usable_width = group_width * 0.72
    bar_width = (usable_width - bar_gap * (len(summaries) - 1)) / len(summaries)
    for success_count in range(expected_trials + 1):
        center_x = left + group_width * (success_count + 0.5)
        group_start = center_x - usable_width / 2
        for model_index, distribution in enumerate(distributions):
            value = distribution[success_count]
            x0 = group_start + model_index * (bar_width + bar_gap)
            x1 = x0 + bar_width
            y0 = bottom - (value / y_max) * (bottom - top)
            draw.rounded_rectangle(
                (x0, y0, x1, bottom),
                radius=3,
                fill=MODEL_COLORS[model_index % len(MODEL_COLORS)],
            )
            draw.text(
                ((x0 + x1) / 2, y0 - 13),
                str(value),
                font=load_font(15, bold=True),
                fill=TEXT,
                anchor="mm",
            )
        draw.text(
            (center_x, bottom + 23),
            f"{success_count}/{expected_trials}",
            font=load_font(16),
            fill=TEXT,
            anchor="mm",
        )

    draw.text(
        (left + (right - left) / 2, height - 34),
        f"Successful trials per task (out of {expected_trials})",
        font=load_font(16),
        fill=MUTED_TEXT,
        anchor="mm",
    )
    return image


def make_group_structure_chart(
    summaries: list[dict[str, Any]], width: int = 1200, height: int = 720
) -> Image.Image:
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw_chart_header(
        draw,
        "Within-group reward structure",
        "Mixed groups provide relative reward signal; uniform groups have zero variance",
        width,
    )

    category_specs = (
        ("all_failure", "All failure (0/4)"),
        ("mixed", "Mixed (1-3/4)"),
        ("all_success", "All success (4/4)"),
    )
    cursor = 70
    legend_font = load_font(16)
    for category, label in category_specs:
        draw.rectangle((cursor, 129, cursor + 14, 143), fill=GROUP_COLORS[category])
        draw.text((cursor + 22, 125), label, font=legend_font, fill=TEXT)
        cursor += 34 + int(draw.textlength(label, font=legend_font))

    left, right = 330, width - 65
    bar_top = 230
    bar_height = 78
    row_gap = 150
    for model_index, summary in enumerate(summaries):
        y0 = bar_top + model_index * row_gap
        y1 = y0 + bar_height
        draw.text(
            (left - 25, (y0 + y1) / 2),
            model_display_name(summary["model"]),
            font=load_font(17, bold=True),
            fill=TEXT,
            anchor="rm",
        )
        start = 0.0
        groups = summary["group_rewards"]
        values = {
            "all_failure": float(groups["all_failure_rate"]) * 100,
            "mixed": float(groups["mixed_rate"]) * 100,
            "all_success": float(groups["all_success_rate"]) * 100,
        }
        for category, _ in category_specs:
            value = values[category]
            x0 = left + (start / 100) * (right - left)
            x1 = left + ((start + value) / 100) * (right - left)
            draw.rectangle((x0, y0, x1, y1), fill=GROUP_COLORS[category])
            if value >= 11:
                draw.text(
                    ((x0 + x1) / 2, (y0 + y1) / 2),
                    f"{value:.1f}%" if value % 1 else f"{value:.0f}%",
                    font=load_font(17, bold=True),
                    fill=TEXT,
                    anchor="mm",
                )
            start += value

    axis_y = bar_top + len(summaries) * row_gap + 5
    for tick in range(0, 101, 20):
        x = left + (tick / 100) * (right - left)
        draw.line((x, axis_y - 9, x, axis_y), fill=FRAME, width=1)
        draw.text((x, axis_y + 18), f"{tick}%", font=load_font(14), fill=MUTED_TEXT, anchor="mm")
    draw.line((left, axis_y, right, axis_y), fill=FRAME, width=1)
    draw.text(
        (left + (right - left) / 2, height - 42),
        "Share of tasks",
        font=load_font(16),
        fill=MUTED_TEXT,
        anchor="mm",
    )
    return image


def make_overview_chart(summaries: list[dict[str, Any]]) -> Image.Image:
    canvas = Image.new("RGB", (1800, 1480), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (55, 36),
        "Four-trial evaluation overview",
        font=load_font(38, bold=True),
        fill=TEXT,
    )
    draw.text(
        (55, 90),
        "Pass^1-4, task-level success distribution, and GRPO within-group signal",
        font=load_font(19),
        fill=MUTED_TEXT,
    )

    pass_chart = make_passk_chart(summaries, width=860, height=620)
    distribution_chart = make_success_distribution_chart(summaries, width=860, height=620)
    group_chart = make_group_structure_chart(summaries, width=1690, height=680)
    canvas.paste(pass_chart, (40, 150))
    canvas.paste(distribution_chart, (900, 150))
    canvas.paste(group_chart, (55, 780))
    return canvas


def write_figures(output_dir: Path, summaries: list[dict[str, Any]]) -> list[Path]:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "passk_curve.png": make_passk_chart(summaries),
        "success_count_distribution.png": make_success_distribution_chart(summaries),
        "group_reward_structure.png": make_group_structure_chart(summaries),
        "evaluation_overview.png": make_overview_chart(summaries),
    }
    paths: list[Path] = []
    for filename, image in figures.items():
        path = figures_dir / filename
        image.save(path, format="PNG", optimize=True)
        paths.append(path)
    return paths


def print_summary(summary: dict[str, Any]) -> None:
    group_rewards = summary["group_rewards"]
    print(f"\n=== {summary['model']} ===")
    print(f"Input: {summary['input_file']}")
    print(
        f"Tasks: {summary['num_tasks']}, simulations: {summary['total_simulations']}, "
        f"successes: {summary['total_successes']}, failures: {summary['total_failures']}"
    )
    print(f"Average reward: {summary['average_reward']:.4f}")
    for k, value in summary["pass_hat_k"].items():
        print(f"Pass^{k}: {value:.4f} ({value * 100:.2f}%)")
    print(
        "Groups: "
        f"all-failure={group_rewards['all_failure_tasks']} "
        f"({group_rewards['all_failure_rate'] * 100:.2f}%), "
        f"mixed={group_rewards['mixed_tasks']} "
        f"({group_rewards['mixed_rate'] * 100:.2f}%), "
        f"all-success={group_rewards['all_success_tasks']} "
        f"({group_rewards['all_success_rate'] * 100:.2f}%)"
    )
    print(
        f"Zero-variance groups: {group_rewards['zero_variance_tasks']} "
        f"({group_rewards['zero_variance_rate'] * 100:.2f}%)"
    )
    distribution = group_rewards["success_count_distribution"]
    print(
        "Success-count distribution: "
        + ", ".join(
            f"{count}/{summary['expected_trials']}={tasks}"
            for count, tasks in distribution.items()
        )
    )
    print(f"Termination counts: {summary['termination_counts']}")


def comparison_row(summary: dict[str, Any]) -> dict[str, Any]:
    groups = summary["group_rewards"]
    row = {
        "model": summary["model"],
        "input_file": summary["input_file"],
        "num_tasks": summary["num_tasks"],
        "total_simulations": summary["total_simulations"],
        "average_reward": summary["average_reward"],
        "all_failure_tasks": groups["all_failure_tasks"],
        "mixed_tasks": groups["mixed_tasks"],
        "all_success_tasks": groups["all_success_tasks"],
        "zero_variance_tasks": groups["zero_variance_tasks"],
        "zero_variance_rate": groups["zero_variance_rate"],
        "mean_task_reward_variance": groups["mean_task_reward_variance"],
    }
    for k, value in summary["pass_hat_k"].items():
        row[f"pass_{k}"] = value
    return row


def resolve_inputs(inputs: list[Path]) -> list[Path]:
    resolved = inputs or sorted(DEFAULT_INPUT_DIR.glob("*.json"))
    if not resolved:
        raise FileNotFoundError(
            f"No input JSON files supplied and none found under {DEFAULT_INPUT_DIR}"
        )
    missing = [str(path) for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Input files do not exist: {missing}")
    return resolved


def main() -> None:
    args = parse_args()
    input_paths = resolve_inputs(args.inputs)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for input_path in input_paths:
        summary, per_task = analyze_results(input_path.resolve(), args.expected_trials)
        output_stem = input_path.stem
        write_json(output_dir / f"{output_stem}_summary.json", summary)
        write_csv(output_dir / f"{output_stem}_per_task.csv", per_task)
        print_summary(summary)
        summaries.append(summary)

    write_json(output_dir / "comparison.json", summaries)
    write_csv(
        output_dir / "comparison.csv", [comparison_row(summary) for summary in summaries]
    )
    figure_paths = write_figures(output_dir, summaries)
    print(f"\nReports written to: {output_dir}")
    print("Figures written to:")
    for figure_path in figure_paths:
        print(f"  {figure_path}")


if __name__ == "__main__":
    main()
