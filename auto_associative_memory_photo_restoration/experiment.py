from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from image_utils import (
    damage_pattern,
    ensure_results_dir,
    image_to_bipolar_pattern,
    pattern_to_image,
    save_image,
    compute_accuracy,
)
from memory_network import HopfieldNetwork


def run_damage_experiment(
    original_image,
    damage_levels=None,
    output_dir="results",
    iterations=50,
    image_size=(32, 32),
    output_prefix="experiment",
):
    """Run the damage experiment across multiple levels and return the results table."""
    if damage_levels is None:
        damage_levels = [10, 20, 30, 40, 50, 60, 70]

    results_dir = ensure_results_dir(output_dir)

    original_pattern = image_to_bipolar_pattern(original_image, target_size=image_size)
    original_pattern = np.asarray(original_pattern, dtype=float).reshape(-1)

    # Save the original image for direct comparison.
    original_image = original_image.resize(image_size, resample=None)
    original_path = results_dir / f"{output_prefix}_original.png"
    save_image(original_image, original_path)

    network = HopfieldNetwork(size=original_pattern.size, iterations=iterations)
    network.store_pattern(original_pattern)

    rows = []
    for damage_percent in damage_levels:
        corrupted = damage_pattern(original_pattern, damage_percent, seed=100 + damage_percent)
        recalled = network.recall(corrupted, iterations=iterations)
        accuracy = compute_accuracy(original_pattern, recalled)

        damaged_image = pattern_to_image(corrupted, image_size)
        recalled_image = pattern_to_image(recalled, image_size)

        damaged_path = results_dir / f"{output_prefix}_damage_{damage_percent}_damaged.png"
        recalled_path = results_dir / f"{output_prefix}_damage_{damage_percent}_recalled.png"

        save_image(damaged_image, damaged_path)
        save_image(recalled_image, recalled_path)

        rows.append(
            {
                "damage_percent": int(damage_percent),
                "accuracy": round(float(accuracy), 4),
                "damaged_image": str(damaged_path),
                "recalled_image": str(recalled_path),
                "reliable_recall": bool(accuracy >= 0.75),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("damage_percent").reset_index(drop=True)
    csv_path = results_dir / "experiment_results.csv"
    df.to_csv(csv_path, index=False)

    chart_path = results_dir / "damage_accuracy_chart.png"
    plot_accuracy_curve(df, chart_path)

    failure_point = find_failure_point(df, threshold=0.75)
    return df, failure_point, csv_path, chart_path


def find_failure_point(results_df, threshold=0.75):
    """Return the first damage level where accuracy falls below the reliable recall threshold."""
    for _, row in results_df.iterrows():
        if row["accuracy"] < threshold:
            return int(row["damage_percent"])
    return None


def plot_accuracy_curve(results_df, output_path):
    """Plot the damage level versus accuracy curve."""
    plt.figure(figsize=(8, 5))
    plt.plot(
        results_df["damage_percent"],
        results_df["accuracy"],
        marker="o",
        color="tab:blue",
        linewidth=2,
    )
    plt.axhline(
        0.75,
        linestyle="--",
        color="tab:red",
        label="Reliable recall threshold (0.75)",
    )
    plt.title("Damage percentage vs reconstruction accuracy")
    plt.xlabel("Damage (%)")
    plt.ylabel("Accuracy")
    plt.xticks(results_df["damage_percent"].tolist())
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
