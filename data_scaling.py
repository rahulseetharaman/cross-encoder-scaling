"""
Scaling Laws Analysis - Data Scaling (Part 2)

Task 1a: For each (model_size, loss_fn), fit a power-law scaling law to each
         metric as a function of training step. Holdout last 5 checkpoints as
         test; use the rest as training.

Task 1b: Plot forecast vs actual for each model size (6 plots total), showing
         all 3 loss functions together with key metrics as subplots.
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

DATA_PATH = Path("C:/Users/rahul/Documents/scaling-laws-analysis/metrics_results.json")
OUTPUT_DIR = Path("C:/Users/rahul/Documents/scaling-laws-analysis/scaling_law_plots")

LOSS_FNS = ["listnet", "ranknet", "bce"]
LOSS_FN_COLORS = {"listnet": "#1f77b4", "ranknet": "#ff7f0e", "bce": "#2ca02c"}

# Metrics to include in forecasting
METRIC_COLS = [
    "recall@1", "recall@2", "recall@5", "recall@10", "recall@50", "recall@100",
    "precision@1", "precision@2", "precision@5", "precision@10", "precision@50", "precision@100",
    "ndcg@1", "ndcg@2", "ndcg@5", "ndcg@10", "ndcg@50", "ndcg@100",
    "MAP", "MRR",
    "CE100", "CE32", "CE64",
]

# Key metrics to display in plots (2 rows x 3 cols per figure)
PLOT_METRICS = ["ndcg@10", "recall@10", "MAP", "MRR", "CE64", "precision@10"]
METRIC_DISPLAY_NAMES = {"CE64": "CE"}

N_TEST_CHECKPOINTS = 5


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def group_by_combo(data: list[dict]) -> dict:
    """Return {(model_size, loss_fn): list_of_records}."""
    groups = defaultdict(list)
    for rec in data:
        groups[(rec["model_size"], rec["loss_fn"])].append(rec)
    return dict(groups)


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def split_train_test(records: list[dict], n_test_checkpoints: int = N_TEST_CHECKPOINTS):
    """
    Split records into train and test sets.
    Holdout the last `n_test_checkpoints` distinct steps (all subsets) as test.
    """
    all_steps = sorted(set(r["step"] for r in records))
    test_steps = set(all_steps[-n_test_checkpoints:])
    train_steps = set(all_steps[:-n_test_checkpoints])

    train = [r for r in records if r["step"] in train_steps]
    test  = [r for r in records if r["step"] in test_steps]
    return train, test, sorted(train_steps), sorted(test_steps)


# ---------------------------------------------------------------------------
# Scaling law model
# ---------------------------------------------------------------------------

def power_law(x, a, b, c):
    """y = a + b * x^c  (Chinchilla-style power law)."""
    return a + b * np.power(np.maximum(x, 1e-9), c)


def fit_metric(steps: np.ndarray, values: np.ndarray, metric: str):
    """
    Fit a power-law scaling law for one metric.
    Returns (params, covariance) or None on failure.
    """
    # Determine whether metric improves (increases) or degrades (decreases) with steps
    is_ce = metric.startswith("CE")

    # Robust initial guesses
    y_mean = float(np.mean(values))
    y_max  = float(np.max(values))
    y_min  = float(np.min(values))
    x_mid  = float(np.median(steps))

    if is_ce:
        # CE decreases toward a floor: a=floor, b>0, c<0
        p0 = [y_min * 0.9, y_max - y_min, -0.5]
        bounds = ([0, 0, -5], [np.inf, np.inf, 0])
    else:
        # Metric increases toward a ceiling: a=ceiling, b<0, c<0
        p0 = [y_max * 1.2, -(y_max - y_min + 1e-9), -0.5]
        bounds = ([0, -np.inf, -5], [1.0, 0, 0])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            params, cov = curve_fit(
                power_law, steps, values,
                p0=p0, bounds=bounds,
                maxfev=10_000,
            )
            return params, cov
        except Exception:
            pass

    # Fallback: unconstrained fit
    try:
        params, cov = curve_fit(
            power_law, steps, values,
            p0=[y_mean, 1e-3, -0.3],
            maxfev=10_000,
        )
        return params, cov
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Forecasting pipeline per (model_size, loss_fn)
# ---------------------------------------------------------------------------

def records_to_arrays(records: list[dict], metric: str):
    """Extract (steps_array, values_array) from records for one metric."""
    steps  = np.array([r["step"]  for r in records], dtype=float)
    values = np.array([r[metric]  for r in records], dtype=float)
    return steps, values


def forecast_combo(
    train: list[dict],
    test: list[dict],
    metric: str,
) -> dict:
    """
    Fit scaling law on train, forecast on test steps.
    Returns dict with fit params, train/test arrays, and forecast arrays.
    """
    x_train, y_train = records_to_arrays(train, metric)
    x_test,  y_test  = records_to_arrays(test,  metric)

    result = fit_metric(x_train, y_train, metric)

    if result is not None:
        params, _ = result
        y_train_pred = power_law(x_train, *params)
        y_test_pred  = power_law(x_test,  *params)
    else:
        # Degenerate: predict mean of training data
        params = None
        y_train_pred = np.full_like(y_train, np.mean(y_train))
        y_test_pred  = np.full_like(y_test,  np.mean(y_train))

    return {
        "params":        params,
        "x_train":       x_train,
        "y_train":       y_train,
        "y_train_pred":  y_train_pred,
        "x_test":        x_test,
        "y_test":        y_test,
        "y_test_pred":   y_test_pred,
    }


def run_all_forecasts(data: list[dict]) -> dict:
    """
    For every (model_size, loss_fn, metric) triple, fit and forecast.

    Returns:
        {model_size: {loss_fn: {metric: forecast_dict}}}
    """
    groups = group_by_combo(data)
    results = defaultdict(lambda: defaultdict(dict))

    for (model_size, loss_fn), records in groups.items():
        train, test, train_steps, test_steps = split_train_test(records)
        print(
            f"  [{model_size:>4s} | {loss_fn:<8s}] "
            f"train steps={len(train_steps):2d}  test steps={len(test_steps):2d}"
        )
        for metric in METRIC_COLS:
            if metric not in records[0]:
                continue
            results[model_size][loss_fn][metric] = forecast_combo(train, test, metric)

    return dict(results)


# ---------------------------------------------------------------------------
# Plotting (Task 1b)
# ---------------------------------------------------------------------------

def smooth_curve(x_sorted: np.ndarray, params, metric: str) -> np.ndarray:
    """Generate a fine-grained forecast curve for plotting."""
    x_fine = np.linspace(x_sorted.min(), x_sorted.max() * 1.05, 300)
    if params is not None:
        y_fine = power_law(x_fine, *params)
    else:
        y_fine = np.full_like(x_fine, np.nan)
    return x_fine, y_fine


def plot_model_size(model_size: str, combo_results: dict, output_dir: Path):
    """
    Create one figure for `model_size` with PLOT_METRICS as subplots.
    Each subplot shows all 3 loss functions (actual scatter + forecast curve).
    """
    n_metrics = len(PLOT_METRICS)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    axes = np.array(axes).flatten()

    fig.suptitle(f"Data Scaling Laws — Model: {model_size}", fontsize=16, fontweight="bold", y=1.01)

    for ax_idx, metric in enumerate(PLOT_METRICS):
        ax = axes[ax_idx]

        for loss_fn in LOSS_FNS:
            if loss_fn not in combo_results:
                continue
            if metric not in combo_results[loss_fn]:
                continue

            fc = combo_results[loss_fn][metric]
            color = LOSS_FN_COLORS[loss_fn]

            # Scatter: actual train points
            ax.scatter(
                fc["x_train"], fc["y_train"],
                color=color, alpha=0.35, s=18, marker="o",
                label=f"{loss_fn} (train actual)",
                zorder=2,
            )
            # Scatter: actual test points
            ax.scatter(
                fc["x_test"], fc["y_test"],
                color=color, alpha=0.90, s=40, marker="*",
                edgecolors="black", linewidths=0.4,
                label=f"{loss_fn} (test actual)",
                zorder=4,
            )
            # Scatter: forecasted test points
            ax.scatter(
                fc["x_test"], fc["y_test_pred"],
                color=color, alpha=0.90, s=40, marker="X",
                edgecolors="black", linewidths=0.4,
                label=f"{loss_fn} (test forecast)",
                zorder=4,
            )

            # Smooth forecast curve (over full step range)
            all_x = np.concatenate([fc["x_train"], fc["x_test"]])
            x_fine, y_fine = smooth_curve(all_x, fc["params"], metric)
            ax.plot(
                x_fine, y_fine,
                color=color, linewidth=1.8, linestyle="--",
                label=f"{loss_fn} (fit)",
                zorder=3,
            )

        # Vertical line separating train / test
        for loss_fn in LOSS_FNS:
            if loss_fn in combo_results and metric in combo_results[loss_fn]:
                fc = combo_results[loss_fn][metric]
                cutoff = fc["x_train"].max()
                ax.axvline(cutoff, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
                ax.text(cutoff, ax.get_ylim()[0] if ax.get_ylim() else 0,
                        " train|test", color="gray", fontsize=7, va="bottom")
                break

        ax.set_title(METRIC_DISPLAY_NAMES.get(metric, metric), fontsize=12)
        ax.set_xlabel("Training Step", fontsize=9)
        ax.set_ylabel("Metric Value", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)

    # Shared legend (deduplicated)
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # Simplify legend: keep only fit + test actual + test forecast per loss_fn
    simplified = {
        k: v for k, v in by_label.items()
        if "fit" in k or "test" in k
    }
    fig.legend(
        simplified.values(), simplified.keys(),
        loc="lower center", ncol=min(len(simplified), 6),
        fontsize=8, bbox_to_anchor=(0.5, -0.03),
        framealpha=0.9,
    )

    # Hide unused axes
    for ax_idx in range(n_metrics, len(axes)):
        axes[ax_idx].set_visible(False)

    fig.tight_layout()
    out_path = output_dir / f"scaling_law_{model_size}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_all(all_results: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    model_sizes_ordered = sorted(all_results.keys())
    for model_size in model_sizes_ordered:
        print(f"  Plotting model_size={model_size} ...")
        plot_model_size(model_size, all_results[model_size], output_dir)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_summary(all_results: dict) -> list[dict]:
    """Compute MAE and RMSE on test set for every (model_size, loss_fn, metric)."""
    rows = []
    for model_size, combo in all_results.items():
        for loss_fn, metrics in combo.items():
            for metric, fc in metrics.items():
                err = fc["y_test"] - fc["y_test_pred"]
                mae  = float(np.mean(np.abs(err)))
                rmse = float(np.sqrt(np.mean(err ** 2)))
                rows.append({
                    "model_size": model_size,
                    "loss_fn":    loss_fn,
                    "metric":     metric,
                    "mae":        mae,
                    "rmse":       rmse,
                    "fit_params": fc["params"].tolist() if fc["params"] is not None else None,
                })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data...")
    data = load_data(DATA_PATH)
    print(f"  {len(data)} records loaded.")

    print("\nFitting scaling laws (Task 1a)...")
    all_results = run_all_forecasts(data)

    print("\nGenerating plots (Task 1b)...")
    plot_all(all_results, OUTPUT_DIR)

    print("\nComputing summary statistics...")
    summary = compute_summary(all_results)

    # Print a compact summary table
    print(f"\n{'model_size':<10} {'loss_fn':<10} {'metric':<16} {'MAE':>10} {'RMSE':>10}")
    print("-" * 60)
    for row in summary:
        print(
            f"{row['model_size']:<10} {row['loss_fn']:<10} "
            f"{row['metric']:<16} {row['mae']:>10.5f} {row['rmse']:>10.5f}"
        )

    # Save summary to JSON
    summary_path = OUTPUT_DIR / "forecast_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
