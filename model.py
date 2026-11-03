"""
Clustering from Scratch in NumPy: HAC, K-Means++, Medoids, GMMs and Spectral Cuts

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - pairwise_distances
import numpy as np


def pairwise_distances(X, metric="euclidean"):
    """Compute the (N, N) pairwise dissimilarity matrix between rows of X.

    Supported metrics:
      - 'euclidean': L2 norm ||x_i - x_j||_2
      - 'sqeuclidean': Squared L2 norm ||x_i - x_j||_2^2
      - 'cityblock': L1 norm sum_k |x_{ik} - x_{jk}|
      - 'correlation': 1 - Pearson correlation coefficient rho(x_i, x_j)

    Parameters
    ----------
    X : array-like of shape (N, D)
        Input data matrix.
    metric : str, optional (default="euclidean")
        Distance metric to compute.

    Returns
    -------
    D : ndarray of shape (N, N), dtype float
        Symmetric dissimilarity matrix with an exact zero diagonal
        and no negative entries.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array for X, got shape {X.shape}")

    N, D = X.shape

    if metric == "sqeuclidean":
        # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * <x_i, x_j>
        dot_product = np.matmul(X, X.T)
        norms_sq = np.diag(dot_product)
        dist = norms_sq[:, None] + norms_sq[None, :] - 2.0 * dot_product
        dist = np.clip(dist, 0.0, None)

    elif metric == "euclidean":
        dot_product = np.matmul(X, X.T)
        norms_sq = np.diag(dot_product)
        dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2.0 * dot_product
        dist = np.sqrt(np.clip(dist_sq, 0.0, None))

    elif metric == "cityblock":
        # For N in the low thousands, broadcasting |x_i - x_j| across (N, N, D)
        # can demand significant memory (~N^2 * D * 8 bytes).
        # We chunk over rows to keep memory usage low and cache-friendly.
        dist = np.zeros((N, N), dtype=float)
        step = 512
        for start in range(0, N, step):
            end = min(start + step, N)
            dist[start:end, :] = np.sum(np.abs(X[start:end, None, :] - X[None, :, :]), axis=-1)

    elif metric == "correlation":
        # Pearson correlation: rho(x_i, x_j) = <u_i, u_j> / (||u_i|| * ||u_j||)
        # where u_i = x_i - mean(x_i)
        X_centered = X - np.mean(X, axis=1, keepdims=True)
        norms = np.linalg.norm(X_centered, axis=1, keepdims=True)

        # Avoid zero division for constant rows (zero variance)
        zero_mask = (norms.ravel() == 0.0)
        safe_norms = np.where(zero_mask[:, None], 1.0, norms)
        X_normed = X_centered / safe_norms

        corr = np.matmul(X_normed, X_normed.T)
        # For rows with zero variance, correlation is undefined / 0 by convention
        if np.any(zero_mask):
            corr[zero_mask, :] = 0.0
            corr[:, zero_mask] = 0.0

        corr = np.clip(corr, -1.0, 1.0)
        dist = 1.0 - corr

    else:
        raise ValueError(f"Unknown metric '{metric}'. Expected one of 'euclidean', 'sqeuclidean', 'cityblock', 'correlation'.")

    # Enforce exact symmetry, exact zero diagonal, and non-negativity
    dist = 0.5 * (dist + dist.T)
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, None)

    return dist

# Step 2 - contingency_table
import numpy as np


def contingency_table(labels_pred, labels_true):
    """Return the integer contingency table N where N[i, j] is the number of

    objects assigned to predicted cluster i that belong to true class j.

    Rows and columns are ordered by the sorted distinct values of labels_pred
    and labels_true, respectively.
    """
    labels_pred = np.asarray(labels_pred)
    labels_true = np.asarray(labels_true)

    # 1. Map each label array to 0-based integer indices based on sorted unique values
    unique_pred, pred_idx = np.unique(labels_pred, return_inverse=True)
    unique_true, true_idx = np.unique(labels_true, return_inverse=True)

    # 2. Allocate an integer array of shape (n_pred_clusters, n_true_classes)
    table = np.zeros((len(unique_pred), len(unique_true)), dtype=int)

    # 3. Accumulate counts at each (pred_idx, true_idx) pair using np.add.at
    np.add.at(table, (pred_idx, true_idx), 1)

    return table

# Step 3 - pair_counts
import numpy as np


def pair_counts(N):
    """Compute pair counting quantities (TP, FP, FN, TN) from a contingency table.

    A pair of distinct objects is:
      - TP: in the same predicted cluster AND in the same true class
      - FP: in the same predicted cluster BUT in different true classes
      - FN: in different predicted clusters BUT in the same true class
      - TN: in different predicted clusters AND in different true classes

    Parameters
    ----------
    N : array-like of shape (R, C)
        Contingency table where N[i, j] is the count of items in predicted
        cluster i and true class j.

    Returns
    -------
    TP, FP, FN, TN : tuple of int
    """
    N = np.asarray(N, dtype=np.int64)

    def comb2(arr):
        """Vectorized n * (n - 1) // 2."""
        return int(np.sum(arr * (arr - 1) // 2))

    # Total number of items n, and total number of pairs comb(n, 2)
    n_samples = int(np.sum(N))
    total_pairs = n_samples * (n_samples - 1) // 2

    # Pairs in the same predicted cluster (row sums) = TP + FP
    row_sums = np.sum(N, axis=1)
    same_pred = comb2(row_sums)

    # Pairs in the same true class (column sums) = TP + FN
    col_sums = np.sum(N, axis=0)
    same_true = comb2(col_sums)

    # Pairs in both the same predicted cluster and same true class = TP
    TP = comb2(N)

    # Derive remaining quantities
    FP = same_pred - TP
    FN = same_true - TP
    TN = total_pairs - TP - FP - FN

    return int(TP), int(FP), int(FN), int(TN)

