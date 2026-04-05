"""
Model Scaling Laws Analysis (Part 3 / instructions3.md)

For each loss function, fit a power-law scaling law relating model size (N)
to each metric, using the last checkpoint of the 17m, 32m, 68m, 150m models
as training data (all 5 subsets as individual data points).

Forecast metric values for 400m and 1b model sizes.
Report MAE and RMSE per (loss_fn, metric, model_size) for the two test models.

Plot: one figure with 6 metric subplots, all 3 loss functions shown together,
X-axis = model size (log scale), Y-axis = metric value.
"""

import json
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_PATH   = Path("C:/Users/rahul/Documents/scaling-laws-analysis/metrics_results.json")
OUTPUT_DIR  = Path("C:/Users/rahul/Documents/scaling-laws-analysis/model_scaling_plots")

# Numeric model sizes (parameters)
MODEL_SIZE_MAP = {
    "17m":  17e6,
    "32m":  32e6,
    "68m":  68e6,
    "150m": 150e6,
    "400m": 400e6,
    "1b":   1e9,
}

TRAIN_MODELS = ["17m", "32m", "68m", "150m"]
TEST_MODELS  = ["400m", "1b"]
LOSS_FNS     = ["listnet", "ranknet", "bce"]
LOSS_FN_COLORS = {"listnet": "#1f77b4", "ranknet": "#ff7f0e", "bce": "#2ca02c"}

METRIC_COLS = [
    "recall@1", "recall@2", "recall@5", "recall@10", "recall@50", "recall@100",
    "precision@1", "precision@2", "precision@5", "precision@10", "precision@50", "precision@100",
    "ndcg@1", "ndcg@2", "ndcg@5", "ndcg@10", "ndcg@50", "ndcg@100",
    "MAP", "MRR",
    "CE100", "CE32", "CE64",
]

PLOT_METRICS = ["ndcg@10", "recall@10", "MAP", "MRR", "CE64", "precision@10"]
METRIC_DISPLAY_NAMES = {"CE64": "CE"}

MODEL_SIZE_LABELS = {
    17e6:  "17m",
    32e6:  "32m",
    68e6:  "68m",
    150e6: "150m",
    400e6: "400m",
    1e9:   "1b",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def get_last_n_checkpoint_records(data: list[dict], n: int = 10) -> list[dict]:
    """Return records from the last n checkpoints of each (model_size, loss_fn)."""
    steps_by_combo = defaultdict(set)
    for rec in data:
        steps_by_combo[(rec["model_size"], rec["loss_fn"])].add(rec["step"])

    last_n_steps = {
        key: set(sorted(steps)[-n:])
        for key, steps in steps_by_combo.items()
    }

    return [
        rec for rec in data
        if rec["step"] in last_n_steps[(rec["model_size"], rec["loss_fn"])]
    ]


# ---------------------------------------------------------------------------
# Scaling law model
# ---------------------------------------------------------------------------

def power_law(x, a, b, c):
    """y = a + b * x^c"""
    return a + b * np.power(np.maximum(x, 1.0), c)


def fit_metric(sizes: np.ndarray, values: np.ndarray, metric: str):
    """Fit power law for one metric vs model size. Returns (params, cov) or None."""
    is_ce = metric.startswith("CE")
    y_max = float(np.max(values))
    y_min = float(np.min(values))

    if is_ce:
        p0 = [y_min * 0.8, y_max - y_min, -0.3]
        bounds = ([0, 0, -5], [np.inf, np.inf, 0])
    else:
        p0 = [y_max * 1.5, -(y_max - y_min + 1e-9), -0.3]
        bounds = ([0, -np.inf, -5], [1.0, 0, 0])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            params, cov = curve_fit(
                power_law, sizes, values,
                p0=p0, bounds=bounds,
                maxfev=20_000,
            )
            return params, cov
        except Exception:
            pass

    # Unconstrained fallback
    try:
        params, cov = curve_fit(
            power_law, sizes, values,
            p0=[float(np.mean(values)), 1e-3, -0.3],
            maxfev=20_000,
        )
        return params, cov
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main forecasting pipeline
# ---------------------------------------------------------------------------

def run_model_scaling(data: list[dict]) -> dict:
    """
    For each (loss_fn, metric):
      - Collect last 10 checkpoint records for train models (one value per step)
      - Fit power law
      - Predict for test models (single value per model)
      - Compare against actual last-10-checkpoint values → MAE, RMSE

    Returns:
        {loss_fn: {metric: {
            "params": ...,
            "train_sizes": ..., "train_values": ...,
            "test_actual": {model: array_of_10_vals},
            "test_pred": {model: scalar},
            "mae": {model: float}, "rmse": {model: float},
        }}}
    """
    last10 = get_last_n_checkpoint_records(data, n=10)

    # Index: {(model_size_str, loss_fn): [records]}
    idx = defaultdict(list)
    for rec in last10:
        idx[(rec["model_size"], rec["loss_fn"])].append(rec)

    results = {}

    for loss_fn in LOSS_FNS:
        results[loss_fn] = {}

        for metric in METRIC_COLS:
            # --- Build training arrays: last 10 checkpoints per train model ---
            train_sizes, train_values = [], []
            for ms in TRAIN_MODELS:
                for rec in idx[(ms, loss_fn)]:
                    if metric not in rec:
                        continue
                    train_sizes.append(MODEL_SIZE_MAP[ms])
                    train_values.append(rec[metric])

            if not train_sizes:
                continue

            train_sizes  = np.array(train_sizes,  dtype=float)
            train_values = np.array(train_values, dtype=float)

            # --- Fit ---
            fit_result = fit_metric(train_sizes, train_values, metric)
            params = fit_result[0] if fit_result is not None else None

            # --- Test: collect actual subset values and predict ---
            test_actual = {}
            test_pred   = {}
            mae_per     = {}
            rmse_per    = {}

            for ms in TEST_MODELS:
                actuals = [rec[metric] for rec in idx[(ms, loss_fn)] if metric in rec]
                if not actuals:
                    continue
                actuals = np.array(actuals, dtype=float)
                n_size  = MODEL_SIZE_MAP[ms]

                if params is not None:
                    pred = float(power_law(n_size, *params))
                else:
                    pred = float(np.mean(train_values))

                errors = actuals - pred
                mae_per[ms]  = float(np.mean(np.abs(errors)))
                rmse_per[ms] = float(np.sqrt(np.mean(errors ** 2)))

                test_actual[ms] = actuals
                test_pred[ms]   = pred

            results[loss_fn][metric] = {
                "params":       params,
                "train_sizes":  train_sizes,
                "train_values": train_values,
                "test_actual":  test_actual,
                "test_pred":    test_pred,
                "mae":          mae_per,
                "rmse":         rmse_per,
            }

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_model_scaling(results: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    n_metrics = len(PLOT_METRICS)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    axes = np.array(axes).flatten()

    fig.suptitle("Model Scaling Laws", fontsize=16, fontweight="bold", y=1.01)

    # All model sizes for the smooth fit curve
    all_sizes_numeric = sorted(MODEL_SIZE_MAP.values())
    x_fine = np.logspace(
        np.log10(min(all_sizes_numeric) * 0.8),
        np.log10(max(all_sizes_numeric) * 1.2),
        300,
    )
    x_tick_vals  = all_sizes_numeric
    x_tick_labels = [MODEL_SIZE_LABELS[v] for v in x_tick_vals]

    for ax_idx, metric in enumerate(PLOT_METRICS):
        ax = axes[ax_idx]
        display = METRIC_DISPLAY_NAMES.get(metric, metric)

        for loss_fn in LOSS_FNS:
            if metric not in results.get(loss_fn, {}):
                continue
            fc    = results[loss_fn][metric]
            color = LOSS_FN_COLORS[loss_fn]

            # Train scatter (actual)
            ax.scatter(
                fc["train_sizes"], fc["train_values"],
                color=color, alpha=0.4, s=20, marker="o",
                label=f"{loss_fn} train actual",
                zorder=2,
            )

            # Test scatter: actual (★) and predicted (✕)
            for ms in TEST_MODELS:
                n_size = MODEL_SIZE_MAP[ms]
                if ms in fc["test_actual"]:
                    ax.scatter(
                        [n_size] * len(fc["test_actual"][ms]),
                        fc["test_actual"][ms],
                        color=color, alpha=0.85, s=50, marker="*",
                        edgecolors="black", linewidths=0.4,
                        label=f"{loss_fn} {ms} actual" if ms == TEST_MODELS[0] else "_",
                        zorder=4,
                    )
                if ms in fc["test_pred"]:
                    ax.scatter(
                        [n_size], [fc["test_pred"][ms]],
                        color=color, alpha=0.95, s=80, marker="X",
                        edgecolors="black", linewidths=0.6,
                        label=f"{loss_fn} {ms} forecast" if ms == TEST_MODELS[0] else "_",
                        zorder=5,
                    )

            # Smooth fit curve
            if fc["params"] is not None:
                y_fine = power_law(x_fine, *fc["params"])
                ax.plot(
                    x_fine, y_fine,
                    color=color, linewidth=1.8, linestyle="--",
                    label=f"{loss_fn} fit",
                    zorder=3,
                )

        # Vertical separator between train and test region
        cutoff = MODEL_SIZE_MAP[TRAIN_MODELS[-1]]
        ax.axvline(cutoff, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)

        ax.set_xscale("log")
        ax.set_xticks(x_tick_vals)
        ax.set_xticklabels(x_tick_labels, fontsize=8)
        ax.set_title(display, fontsize=12)
        ax.set_xlabel("Model Size", fontsize=9)
        ax.set_ylabel("Metric Value", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3, which="both")

    # Shared legend (deduplicated, simplified)
    handles, labels = [], []
    for ax in axes[:n_metrics]:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels and not li.startswith("_"):
                handles.append(hi)
                labels.append(li)

    # Keep only fit + actuals + forecasts (drop per-model duplicates)
    keep_h, keep_l = [], []
    seen_types = set()
    for hi, li in zip(handles, labels):
        key = " ".join(li.split()[:2])   # e.g. "listnet fit", "ranknet train"
        if key not in seen_types:
            seen_types.add(key)
            keep_h.append(hi)
            keep_l.append(li)

    fig.legend(
        keep_h, keep_l,
        loc="lower center", ncol=min(len(keep_l), 6),
        fontsize=8, bbox_to_anchor=(0.5, -0.03),
        framealpha=0.9,
    )

    # Hide unused axes
    for ax_idx in range(n_metrics, len(axes)):
        axes[ax_idx].set_visible(False)

    fig.tight_layout()
    out_path = output_dir / "model_scaling.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_and_save_summary(results: dict, output_dir: Path):
    rows = []
    print(f"\n{'loss_fn':<10} {'metric':<16} {'model':<6} {'MAE':>10} {'RMSE':>10}")
    print("-" * 58)
    for loss_fn in LOSS_FNS:
        for metric in METRIC_COLS:
            if metric not in results.get(loss_fn, {}):
                continue
            fc = results[loss_fn][metric]
            for ms in TEST_MODELS:
                if ms not in fc["mae"]:
                    continue
                print(
                    f"{loss_fn:<10} {metric:<16} {ms:<6} "
                    f"{fc['mae'][ms]:>10.5f} {fc['rmse'][ms]:>10.5f}"
                )
                rows.append({
                    "loss_fn":    loss_fn,
                    "metric":     metric,
                    "model_size": ms,
                    "mae":        fc["mae"][ms],
                    "rmse":       fc["rmse"][ms],
                    "prediction": fc["test_pred"].get(ms),
                    "fit_params": fc["params"].tolist() if fc["params"] is not None else None,
                })

    summary_path = output_dir / "model_scaling_summary.json"
    with open(summary_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data...")
    data = load_data(DATA_PATH)
    print(f"  {len(data)} records loaded.")

    print("\nFitting model scaling laws...")
    results = run_model_scaling(data)

    print("\nGenerating plot...")
    plot_model_scaling(results, OUTPUT_DIR)

    print_and_save_summary(results, OUTPUT_DIR)
    print("\nDone.")


if __name__ == "__main__":
    main()
