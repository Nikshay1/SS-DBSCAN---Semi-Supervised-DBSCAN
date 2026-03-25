"""
DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
====================================================================
A from-scratch implementation of the standard DBSCAN algorithm.

Core idea:
  - A point is a *core point* if it has at least `MinPts` neighbours
    within `Eps` distance.
  - Clusters are formed by connecting core points that are neighbours,
    along with their border points.
  - Points that are not reachable from any core point are labelled
    as noise (-1).

Time complexity: O(n^2) in the worst case (distance matrix approach).
"""

import numpy as np
from scipy.spatial.distance import cdist
import time


class DBSCAN:
    """Standard DBSCAN clustering algorithm."""

    def __init__(self, eps: float, min_pts: int):
        """
        Parameters
        ----------
        eps : float
            Maximum distance between two points to be considered neighbours.
        min_pts : int
            Minimum number of neighbours required for a point to be a core point.
        """
        self.eps = eps
        self.min_pts = min_pts
        self.labels_ = None          # cluster labels for each point
        self.n_clusters_ = 0         # number of clusters found
        self.n_noise_ = 0            # number of noise points
        self.execution_time_ = 0.0   # wall-clock seconds

    def fit(self, X: np.ndarray) -> "DBSCAN":
        """
        Run DBSCAN on the dataset X.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        start = time.perf_counter()

        n = X.shape[0]
        # Pre-compute pairwise Euclidean distance matrix
        dist_matrix = cdist(X, X, metric="euclidean")

        labels = np.zeros(n, dtype=int)   # 0 = unvisited
        cluster_id = 0

        for i in range(n):
            if labels[i] != 0:
                continue  # already processed

            # Find all neighbours within Eps
            neighbours = np.where(dist_matrix[i] <= self.eps)[0].tolist()

            if len(neighbours) < self.min_pts:
                labels[i] = -1   # mark as noise
                continue

            # Start a new cluster
            cluster_id += 1
            labels[i] = cluster_id

            # Use a FIFO queue (list with pointer) to expand the cluster
            queue = list(neighbours)
            ptr = 0

            while ptr < len(queue):
                q = queue[ptr]
                ptr += 1

                if labels[q] == -1:
                    # Previously marked noise → border point
                    labels[q] = cluster_id
                    continue

                if labels[q] != 0:
                    continue  # already assigned to a cluster

                labels[q] = cluster_id

                # Find neighbours of q
                q_neighbours = np.where(dist_matrix[q] <= self.eps)[0].tolist()

                if len(q_neighbours) >= self.min_pts:
                    # q is also a core point → expand
                    queue.extend(q_neighbours)

        self.labels_ = labels
        self.n_clusters_ = cluster_id
        self.n_noise_ = int(np.sum(labels == -1))
        self.execution_time_ = time.perf_counter() - start
        return self
