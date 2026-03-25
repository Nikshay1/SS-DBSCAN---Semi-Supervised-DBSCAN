"""
main.py — SS-DBSCAN Lab Project Driver
=======================================
Loads the Letters Recognition UCI dataset, runs both
DBSCAN and SS-DBSCAN, computes evaluation metrics,
and generates comparison plots saved to output/.

Usage:
    python main.py

Outputs (saved in output/ directory):
    • metrics_comparison.png   — V-measure, ARI, Silhouette bar chart
    • time_comparison.png      — execution time bar chart
    • cluster_stats.png        — cluster count and noise point comparison
    • results_summary.txt      — text summary of all metrics
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no GUI required)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.metrics import v_measure_score, adjusted_rand_score, silhouette_score
from sklearn.preprocessing import LabelEncoder

from dbscan import DBSCAN
from ss_dbscan import SS_DBSCAN, make_letters_is_important

# ═══════════════════════════════════════════════════════════════
#  Configuration (from the paper — Case Study 1)
# ═══════════════════════════════════════════════════════════════
EPS = 4
MIN_PTS = 17          # D + 1 = 16 + 1 = 17  (noiseless data)
SAMPLE_SIZE = 2000    # Use a subset to keep runtime manageable
RANDOM_SEED = 42
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# ═══════════════════════════════════════════════════════════════
#  Styling
# ═══════════════════════════════════════════════════════════════
sns.set_theme(style="whitegrid", font_scale=1.15)
COLORS = {"DBSCAN": "#3b82f6", "SS-DBSCAN": "#f97316"}


def load_letters_dataset(n_samples: int = SAMPLE_SIZE):
    """Load the Letters Recognition UCI dataset and return features + labels."""
    print("[1/5] Loading Letters Recognition dataset from OpenML...")
    data = fetch_openml(name="letter", version=1, as_frame=False, parser="auto")
    X_full = data.data.astype(float)
    y_full = data.target

    # Encode string labels to integers for metric computation
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_full)

    # Stratified random sample for tractable computation
    rng = np.random.RandomState(RANDOM_SEED)
    idx = rng.choice(len(X_full), size=min(n_samples, len(X_full)), replace=False)
    return X_full[idx], y_encoded[idx], le.classes_


def run_algorithms(X, y):
    """Run DBSCAN and SS-DBSCAN, returning a results dict."""
    results = {}

    # ── Standard DBSCAN ─────────────────────────────────────
    print("[2/5] Running standard DBSCAN ...")
    db = DBSCAN(eps=EPS, min_pts=MIN_PTS)
    db.fit(X)
    results["DBSCAN"] = evaluate(db, X, y)
    print(f"       Clusters: {db.n_clusters_}  |  Noise: {db.n_noise_}  |  "
          f"Time: {db.execution_time_:.2f}s")

    # ── SS-DBSCAN ───────────────────────────────────────────
    print("[3/5] Running SS-DBSCAN ...")
    is_imp = make_letters_is_important(X)
    ss = SS_DBSCAN(eps=EPS, min_pts=MIN_PTS, is_important_fn=is_imp)
    ss.fit(X)
    results["SS-DBSCAN"] = evaluate(ss, X, y)
    print(f"       Clusters: {ss.n_clusters_}  |  Noise: {ss.n_noise_}  |  "
          f"Time: {ss.execution_time_:.2f}s")

    return results


def evaluate(model, X, y_true):
    """Compute V-measure, ARI, Silhouette, and collect metadata."""
    labels = model.labels_
    # For metrics, treat noise as its own pseudo-cluster (-1)
    mask = labels != -1   # only clustered points for silhouette

    v = v_measure_score(y_true, labels) if model.n_clusters_ > 0 else 0.0
    ari = adjusted_rand_score(y_true, labels) if model.n_clusters_ > 0 else 0.0

    if mask.sum() > 1 and len(set(labels[mask])) > 1:
        sil = silhouette_score(X[mask], labels[mask])
    else:
        sil = 0.0

    return {
        "V-measure": round(v, 4),
        "ARI": round(ari, 4),
        "Silhouette": round(sil, 4),
        "Clusters": model.n_clusters_,
        "Noise": model.n_noise_,
        "Time (s)": round(model.execution_time_, 3),
    }


def plot_metrics(results):
    """Generate and save comparison charts."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("[4/5] Generating comparison plots ...")

    algo_names = list(results.keys())

    # ── 1. Metrics comparison bar chart ──────────────────────
    metrics = ["V-measure", "ARI", "Silhouette"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(metrics))
    width = 0.32

    for i, algo in enumerate(algo_names):
        vals = [results[algo][m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=algo,
                      color=COLORS[algo], edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10,
                    fontweight="bold")

    ax.set_ylabel("Score")
    ax.set_title("Clustering Quality Metrics — DBSCAN vs SS-DBSCAN",
                 fontweight="bold", fontsize=14)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, max(max(results[a][m] for m in metrics) for a in algo_names) + 0.12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "metrics_comparison.png"), dpi=150)
    plt.close(fig)

    # ── 2. Execution time comparison ────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    times = [results[a]["Time (s)"] for a in algo_names]
    bars = ax.bar(algo_names, times, color=[COLORS[a] for a in algo_names],
                  edgecolor="white", linewidth=0.8, width=0.45)
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{t:.3f}s", ha="center", va="bottom", fontsize=11,
                fontweight="bold")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Execution Time — DBSCAN vs SS-DBSCAN",
                 fontweight="bold", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "time_comparison.png"), dpi=150)
    plt.close(fig)

    # ── 3. Cluster / Noise statistics ───────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    clusters = [results[a]["Clusters"] for a in algo_names]
    bars = axes[0].bar(algo_names, clusters,
                       color=[COLORS[a] for a in algo_names],
                       edgecolor="white", width=0.45)
    for bar, c in zip(bars, clusters):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     str(c), ha="center", va="bottom", fontsize=12,
                     fontweight="bold")
    axes[0].set_title("Number of Clusters", fontweight="bold")
    axes[0].set_ylabel("Count")

    noise = [results[a]["Noise"] for a in algo_names]
    bars = axes[1].bar(algo_names, noise,
                       color=[COLORS[a] for a in algo_names],
                       edgecolor="white", width=0.45)
    for bar, n in zip(bars, noise):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     str(n), ha="center", va="bottom", fontsize=12,
                     fontweight="bold")
    axes[1].set_title("Noise Points", fontweight="bold")
    axes[1].set_ylabel("Count")

    fig.suptitle("Cluster Statistics — DBSCAN vs SS-DBSCAN",
                 fontweight="bold", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "cluster_stats.png"), dpi=150)
    plt.close(fig)


def save_summary(results):
    """Write a plain-text summary of all results."""
    path = os.path.join(OUTPUT_DIR, "results_summary.txt")
    with open(path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("  SS-DBSCAN Lab Project — Results Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"  Dataset : Letters Recognition (UCI)\n")
        f.write(f"  Samples : {SAMPLE_SIZE}\n")
        f.write(f"  Eps     : {EPS}\n")
        f.write(f"  MinPts  : {MIN_PTS}\n\n")

        header = f"{'Metric':<15}" + "".join(f"{a:>14}" for a in results) + "\n"
        f.write(header)
        f.write("-" * len(header) + "\n")

        for metric in ["V-measure", "ARI", "Silhouette", "Clusters", "Noise", "Time (s)"]:
            row = f"{metric:<15}" + "".join(
                f"{results[a][metric]:>14}" for a in results
            ) + "\n"
            f.write(row)

    print(f"[5/5] Summary saved to {path}")


# ═══════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    X, y, class_names = load_letters_dataset()
    results = run_algorithms(X, y)
    plot_metrics(results)
    save_summary(results)
    print("\n✅  Done!  All outputs saved in the 'output/' directory.")
