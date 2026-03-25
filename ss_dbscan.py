"""
SS-DBSCAN (Semi-Supervised DBSCAN)
===================================
Implementation based on the research paper:
  "SS-DBSCAN: Semi-Supervised Density-Based Spatial Clustering of
   Applications With Noise for Meaningful Clustering in Diverse Density Data"
  — Abdulhameed et al., IEEE Access, 2024.

Key difference from standard DBSCAN:
  A point can only be a core point if BOTH conditions are satisfied:
    1. It has at least MinPts neighbours within Eps distance   (same as DBSCAN)
    2. Is_important(point) returns True                        (NEW condition)

  The Is_important function encodes domain knowledge about the dataset.
  For the Letters Recognition dataset (Case Study 1 in the paper):
    A letter instance is "important" if ANY of its selected features
    has a value >= max(feature_column) - 2.
    Features 1, 2, 3, 4, and 10 (0-indexed: 0,1,2,3,9) are excluded.

Time complexity: O(n^2) — same as standard DBSCAN.
"""

import numpy as np
from scipy.spatial.distance import cdist
import time
from typing import Callable, Optional


class SS_DBSCAN:
    """Semi-Supervised DBSCAN clustering algorithm."""

    def __init__(self, eps: float, min_pts: int,
                 is_important_fn: Optional[Callable] = None):
        """
        Parameters
        ----------
        eps : float
            Maximum distance between two points to be considered neighbours.
        min_pts : int
            Minimum number of neighbours for core point candidacy.
        is_important_fn : callable, optional
            A function  f(point_index, X) -> bool  that returns True if the
            point satisfies the additional semi-supervised condition.
            If None, every point is considered important (reduces to DBSCAN).
        """
        self.eps = eps
        self.min_pts = min_pts
        self.is_important_fn = is_important_fn
        self.labels_ = None
        self.n_clusters_ = 0
        self.n_noise_ = 0
        self.execution_time_ = 0.0

    def fit(self, X: np.ndarray) -> "SS_DBSCAN":
        """
        Run SS-DBSCAN on the dataset X.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        start = time.perf_counter()

        n = X.shape[0]
        dist_matrix = cdist(X, X, metric="euclidean")

        labels = np.zeros(n, dtype=int)  # 0 = unvisited
        cluster_id = 0

        for i in range(n):
            if labels[i] != 0:
                continue

            neighbours = np.where(dist_matrix[i] <= self.eps)[0].tolist()

            if len(neighbours) < self.min_pts:
                labels[i] = -1   # noise
                continue

            # Start a new cluster
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

                q_neighbours = np.where(dist_matrix[q] <= self.eps)[0].tolist()

                # ──────────────────────────────────────────────
                # SS-DBSCAN MODIFICATION (Algorithm 1, line 17):
                #   is_core = len(q_neighbours) >= MinPts
                #             AND Is_important(point)
                # ──────────────────────────────────────────────
                is_core = len(q_neighbours) >= self.min_pts

                if is_core and self.is_important_fn is not None:
                    is_core = is_core and self.is_important_fn(q, X)

                if is_core:
                    queue.extend(q_neighbours)

        self.labels_ = labels
        self.n_clusters_ = cluster_id
        self.n_noise_ = int(np.sum(labels == -1))
        self.execution_time_ = time.perf_counter() - start
        return self


# ═══════════════════════════════════════════════════════════════════
#  Pre-built Is_important functions for each case study
# ═══════════════════════════════════════════════════════════════════

def make_letters_is_important(X: np.ndarray):
    """
    Build the Is_important function for the Letters Recognition dataset
    (Case Study 1, predicate 13 in the paper).

    A letter instance is important if ANY of its *selected* features
    has a value >= max(that feature column) - 2.

    Features excluded (1-indexed): 1, 2, 3, 4, 10
    → 0-indexed exclusions: 0, 1, 2, 3, 9
    """
    excluded = {0, 1, 2, 3, 9}
    selected_cols = [j for j in range(X.shape[1]) if j not in excluded]

    # Pre-compute column maximums for the selected features
    col_maxes = {j: X[:, j].max() for j in selected_cols}

    def is_important(point_idx: int, data: np.ndarray) -> bool:
        for j in selected_cols:
            if data[point_idx, j] >= col_maxes[j] - 2:
                return True
        return False

    return is_important
