"""
marble_demo.py — Figure 1 Recreation
======================================
Recreates the marble balls clustering example from the SS-DBSCAN paper
(Figure 1): 15 marble balls of varying sizes clustered by DBSCAN vs
SS-DBSCAN.

The Is_important condition for marbles:
  A marble is important if its radius > 2 × mean(all radii)

Output: output/marble_balls_clustering.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyArrowPatch
from scipy.spatial.distance import cdist
import os

# ═══════════════════════════════════════════════════════════════
#  Generate 15 marble balls with (x, y, radius)
# ═══════════════════════════════════════════════════════════════
np.random.seed(42)

# Create 15 marbles: 3 big (important) + 12 small, arranged so DBSCAN
# merges into 1 large cluster while SS-DBSCAN produces 2 separate clusters
marbles = [
    # Big marbles (will be "important") — cluster centres
    {"x": 3.5, "y": 5.0, "r": 1.4, "label": "Big"},
    {"x": 8.5, "y": 5.0, "r": 1.3, "label": "Big"},
    {"x": 13.0, "y": 5.0, "r": 1.5, "label": "Big"},
    # Small marbles around the left big marble
    {"x": 2.0, "y": 4.0, "r": 0.35, "label": "Small"},
    {"x": 4.5, "y": 6.2, "r": 0.30, "label": "Small"},
    {"x": 2.2, "y": 6.3, "r": 0.40, "label": "Small"},
    {"x": 4.8, "y": 3.8, "r": 0.28, "label": "Small"},
    # Small marbles bridging left–centre (cause DBSCAN to merge)
    {"x": 5.8, "y": 4.5, "r": 0.32, "label": "Small"},
    {"x": 6.5, "y": 5.5, "r": 0.30, "label": "Small"},
    # Small marbles around the centre big marble
    {"x": 7.3, "y": 3.8, "r": 0.35, "label": "Small"},
    {"x": 9.8, "y": 6.0, "r": 0.38, "label": "Small"},
    # Small marbles bridging centre–right
    {"x": 10.8, "y": 5.2, "r": 0.32, "label": "Small"},
    # Small marbles around the right big marble
    {"x": 14.2, "y": 6.2, "r": 0.35, "label": "Small"},
    {"x": 14.0, "y": 3.8, "r": 0.28, "label": "Small"},
    {"x": 11.8, "y": 4.2, "r": 0.40, "label": "Small"},
]

positions = np.array([[m["x"], m["y"]] for m in marbles])
radii = np.array([m["r"] for m in marbles])
mean_radius = radii.mean()

EPS = 3.5       # large enough that DBSCAN chains everything into 1 cluster
MIN_PTS = 3

# ═══════════════════════════════════════════════════════════════
#  DBSCAN Clustering
# ═══════════════════════════════════════════════════════════════
def run_dbscan(positions, eps, min_pts, is_important_fn=None):
    n = len(positions)
    dist_matrix = cdist(positions, positions, metric="euclidean")
    labels = np.zeros(n, dtype=int)
    cluster_id = 0

    for i in range(n):
        if labels[i] != 0:
            continue
        neighbours = np.where(dist_matrix[i] <= eps)[0].tolist()
        if len(neighbours) < min_pts:
            labels[i] = -1
            continue

        cluster_id += 1
        labels[i] = cluster_id
        queue = list(neighbours)
        ptr = 0

        while ptr < len(queue):
            q = queue[ptr]
            ptr += 1
            if labels[q] == -1:
                labels[q] = cluster_id
                continue
            if labels[q] != 0:
                continue
            labels[q] = cluster_id
            q_neighbours = np.where(dist_matrix[q] <= eps)[0].tolist()

            is_core = len(q_neighbours) >= min_pts
            if is_core and is_important_fn is not None:
                is_core = is_core and is_important_fn(q)

            if is_core:
                queue.extend(q_neighbours)

    return labels, cluster_id


# Is_important: radius > 2 * mean_radius (predicate 3 from paper)
def is_important_marble(idx):
    return radii[idx] > 2 * mean_radius

dbscan_labels, dbscan_n = run_dbscan(positions, EPS, MIN_PTS, is_important_fn=None)
ss_labels, ss_n = run_dbscan(positions, EPS, MIN_PTS, is_important_fn=is_important_marble)

# Identify core points for each
dist_matrix = cdist(positions, positions, metric="euclidean")
dbscan_cores = set()
ss_cores = set()
for i in range(len(marbles)):
    nn = np.where(dist_matrix[i] <= EPS)[0]
    if len(nn) >= MIN_PTS:
        dbscan_cores.add(i)
        if is_important_marble(i):
            ss_cores.add(i)

# ═══════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════
CLUSTER_COLORS = [
    "#3b82f6", "#ef4444", "#22c55e", "#f59e0b",
    "#8b5cf6", "#ec4899", "#06b6d4", "#f97316",
]
NOISE_COLOR = "#94a3b8"

def draw_panel(ax, labels, n_clusters, cores, title, subtitle):
    ax.set_xlim(-1, 16)
    ax.set_ylim(-0.5, 9.5)
    ax.set_aspect("equal")
    ax.set_facecolor("#fafafa")

    # Draw Eps circles for core points
    for idx in cores:
        eps_circle = Circle(
            (marbles[idx]["x"], marbles[idx]["y"]), EPS,
            fill=False, linestyle="--", linewidth=0.8,
            edgecolor="#cbd5e1", alpha=0.6
        )
        ax.add_patch(eps_circle)

    # Draw marbles
    for i, m in enumerate(marbles):
        c_label = labels[i]
        if c_label == -1:
            color = NOISE_COLOR
            ec = "#64748b"
        else:
            color = CLUSTER_COLORS[(c_label - 1) % len(CLUSTER_COLORS)]
            ec = "white"

        circle = Circle(
            (m["x"], m["y"]), m["r"] * 1.8,
            facecolor=color, edgecolor=ec,
            linewidth=1.5, alpha=0.85, zorder=3
        )
        ax.add_patch(circle)

        # Mark core points with "c" and noise with "n"
        if i in cores:
            ax.text(m["x"], m["y"], "c", ha="center", va="center",
                    fontsize=8, fontweight="bold", color="white", zorder=4)
        elif c_label == -1:
            ax.text(m["x"], m["y"], "n", ha="center", va="center",
                    fontsize=7, fontweight="bold", color="white", zorder=4)

    # Title
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.text(0.5, -0.08, subtitle, transform=ax.transAxes,
            ha="center", fontsize=10, color="#64748b", style="italic")

    # Clean axes
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))
fig.suptitle("Figure 1: Marble Balls Clustering — DBSCAN vs SS-DBSCAN",
             fontsize=15, fontweight="bold", y=1.02)

draw_panel(
    ax1, dbscan_labels, dbscan_n, dbscan_cores,
    "(a) DBSCAN Clustering",
    f"{dbscan_n} cluster(s), {len(dbscan_cores)} core points"
)

draw_panel(
    ax2, ss_labels, ss_n, ss_cores,
    "(b) SS-DBSCAN Clustering",
    f"{ss_n} cluster(s), {len(ss_cores)} core points"
)

# Legend
legend_elements = [
    mpatches.Patch(facecolor=CLUSTER_COLORS[0], edgecolor="white", label="Cluster"),
    mpatches.Patch(facecolor=NOISE_COLOR, edgecolor="#64748b", label="Noise point (n)"),
    mpatches.Patch(facecolor="white", edgecolor="#cbd5e1", linestyle="--",
                   label=f"Eps radius = {EPS}"),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#3b82f6',
               markersize=10, label='Core point (c)'),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=4,
           fontsize=10, frameon=True, fancybox=True,
           edgecolor="#e2e8f0", bbox_to_anchor=(0.5, -0.04))

fig.tight_layout()
os.makedirs("output", exist_ok=True)
fig.savefig("output/marble_balls_clustering.png", dpi=180,
            bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"Mean marble radius: {mean_radius:.3f}")
print(f"Importance threshold (2 × mean): {2 * mean_radius:.3f}")
print(f"Important marbles: {[i for i in range(len(marbles)) if is_important_marble(i)]}")
print(f"\nDBSCAN:    {dbscan_n} cluster(s), {len(dbscan_cores)} core points")
print(f"SS-DBSCAN: {ss_n} cluster(s), {len(ss_cores)} core points")
print(f"\n✅  Saved to output/marble_balls_clustering.png")
