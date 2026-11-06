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

# Step 4 - cluster_scores
import numpy as np


def cluster_scores(labels_pred, labels_true):
    """Compute external evaluation scores between predicted clusterings and true labels.

    Returns a dict of floats containing:
      - 'purity': (1 / n) * sum_i max_j N_ij
      - 'rand': (TP + TN) / (TP + FP + FN + TN)
      - 'ari': Adjusted Rand Index
      - 'mi': Mutual Information I(U, V) in nats
      - 'nmi': Normalized Mutual Information 2 * I(U, V) / (H(U) + H(V))
    """
    N = contingency_table(labels_pred, labels_true)
    TP, FP, FN, TN = pair_counts(N)

    n_samples = float(np.sum(N))

    # 1. Purity: (1 / n) * sum_i max_j N_ij
    purity = float(np.sum(np.max(N, axis=1)) / n_samples)

    # 2. Rand Index: (TP + TN) / Total Pairs
    total_pairs = float(TP + FP + FN + TN)
    rand = float((TP + TN) / total_pairs) if total_pairs > 0 else 1.0

    # 3. Adjusted Rand Index (ARI)
    # E = (TP + FP) * (TP + FN) / total_pairs
    same_pred = TP + FP
    same_true = TP + FN
    expected_index = float(same_pred * same_true) / total_pairs if total_pairs > 0 else 0.0
    max_index = 0.5 * (same_pred + same_true)
    ari_denom = max_index - expected_index

    if ari_denom == 0.0:
        ari = 1.0
    else:
        ari = float((TP - expected_index) / ari_denom)

    # 4. Mutual Information (MI) and Normalized Mutual Information (NMI)
    # Row and column marginal sums
    row_sums = np.sum(N, axis=1)  # shape: (R,)
    col_sums = np.sum(N, axis=0)  # shape: (C,)

    # Only compute log terms for non-zero entries (0 * log(0) = 0)
    nz_i, nz_j = np.nonzero(N)
    nz_vals = N[nz_i, nz_j].astype(float)

    # I(U, V) = sum_{i,j} (N_ij / n) * log((n * N_ij) / (a_i * b_j))
    #         = sum_{i,j} (N_ij / n) * (log(n * N_ij) - log(a_i) - log(b_j))
    mi_terms = (nz_vals / n_samples) * (
        np.log(n_samples * nz_vals) - np.log(row_sums[nz_i]) - np.log(col_sums[nz_j])
    )
    mi = float(np.sum(mi_terms))
    mi = max(0.0, mi)  # Clip potential negative rounding noise

    # Entropies H(U) and H(V) in nats
    p_u = row_sums[row_sums > 0].astype(float) / n_samples
    h_u = -np.sum(p_u * np.log(p_u))

    p_v = col_sums[col_sums > 0].astype(float) / n_samples
    h_v = -np.sum(p_v * np.log(p_v))

    denom_nmi = 0.5 * (h_u + h_v)
    if denom_nmi == 0.0:
        nmi = 1.0
    else:
        nmi = float(mi / denom_nmi)

    return {
        "purity": purity,
        "rand": rand,
        "ari": ari,
        "mi": mi,
        "nmi": nmi,
    }

# Step 5 - agglomerate
import numpy as np


def agglomerate(D, linkage="average"):
    """Perform hierarchical agglomerative clustering on a dissimilarity matrix D.

    Parameters
    ----------
    D : array-like of shape (N, N)
        Pairwise dissimilarity matrix. Will not be modified.
    linkage : {"single", "complete", "average"}, default="average"
        The linkage criterion determining cluster distance.

    Returns
    -------
    Z : ndarray of shape (N - 1, 4), dtype float
        Linkage matrix where row t is [a, b, height, size] with a < b,
        representing the clusters merged at step t.
    """
    if linkage not in ("single", "complete", "average"):
        raise ValueError(
            f"Unknown linkage '{linkage}'. Expected 'single', 'complete', or 'average'."
        )

    D = np.asarray(D, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square 2D matrix.")

    N = D.shape[0]
    if N <= 1:
        return np.empty((0, 4), dtype=float)

    # Work on a copy of dissimilarities between active clusters
    dist = D.copy()
    np.fill_diagonal(dist, np.inf)

    # Track active cluster IDs and sizes
    cluster_ids = np.arange(N, dtype=int)
    cluster_sizes = np.ones(N, dtype=float)

    Z = np.empty((N - 1, 4), dtype=float)

    for t in range(N - 1):
        k = N - t  # Current number of active clusters
        sub_dist = dist[:k, :k]

        min_val = np.min(sub_dist)

        # Inspect upper triangle of active submatrix for min distance
        i_indices, j_indices = np.where(np.triu(sub_dist, k=1) == min_val)

        # Map active slot indices to global cluster IDs
        c_i = cluster_ids[i_indices]
        c_j = cluster_ids[j_indices]

        a_cand = np.minimum(c_i, c_j)
        b_cand = np.maximum(c_i, c_j)

        # Tie-breaker: smallest a, then smallest b
        tie_order = np.lexsort((b_cand, a_cand))
        best_idx = tie_order[0]

        slot_a = i_indices[best_idx]
        slot_b = j_indices[best_idx]
        if slot_a > slot_b:
            slot_a, slot_b = slot_b, slot_a

        id_a = cluster_ids[slot_a]
        id_b = cluster_ids[slot_b]
        if id_a > id_b:
            id_a, id_b = id_b, id_a

        size_a = cluster_sizes[slot_a]
        size_b = cluster_sizes[slot_b]
        new_size = size_a + size_b
        new_id = N + t

        Z[t] = [float(id_a), float(id_b), float(min_val), float(new_size)]

        # Distance updates to remaining clusters
        mask = np.ones(k, dtype=bool)
        mask[[slot_a, slot_b]] = False

        d_a = sub_dist[slot_a, mask]
        d_b = sub_dist[slot_b, mask]

        if linkage == "single":
            d_new = np.minimum(d_a, d_b)
        elif linkage == "complete":
            d_new = np.maximum(d_a, d_b)
        elif linkage == "average":
            d_new = (size_a * d_a + size_b * d_b) / new_size

        # Update slot_a with merged cluster data inside the active slice
        cluster_ids[slot_a] = new_id
        cluster_sizes[slot_a] = new_size

        sub_dist[slot_a, mask] = d_new
        sub_dist[mask, slot_a] = d_new
        sub_dist[slot_a, slot_a] = np.inf

        # Swap last active slot (k - 1) into slot_b to shrink active matrix to k - 1
        last_slot = k - 1
        if slot_b != last_slot:
            cluster_ids[slot_b] = cluster_ids[last_slot]
            cluster_sizes[slot_b] = cluster_sizes[last_slot]

            sub_dist[slot_b, :k] = sub_dist[last_slot, :k]
            sub_dist[:k, slot_b] = sub_dist[:k, last_slot]
            sub_dist[slot_b, slot_b] = np.inf

    return Z

# Step 6 - cut_tree
import numpy as np


def cut_tree(Z, n_clusters=None, height=None):
    """Cut a hierarchical clustering tree Z into flat cluster labels.

    Parameters
    ----------
    Z : array-like of shape (N - 1, 4)
        Linkage matrix as returned by agglomerate / scipy.cluster.hierarchy.linkage.
    n_clusters : int, optional
        Desired number of clusters (1 <= n_clusters <= N).
    height : float, optional
        Maximum threshold on merge height. Merges with height <= height are retained.

    Returns
    -------
    labels : ndarray of shape (N,), dtype int
        Cluster assignments with labels 0, 1, ... numbered by first appearance.
    """
    # 1. Validate arguments: exactly one must be provided
    if (n_clusters is None and height is None) or (
        n_clusters is not None and height is not None
    ):
        raise ValueError("Exactly one of 'n_clusters' or 'height' must be specified.")

    Z = np.asarray(Z, dtype=float)
    N = Z.shape[0] + 1  # Total original objects

    if N == 1:
        return np.zeros(1, dtype=int)

    # 2. Determine how many merges from the beginning of Z to apply
    if n_clusters is not None:
        if not (1 <= n_clusters <= N):
            raise ValueError(f"'n_clusters' must be between 1 and {N}, got {n_clusters}")
        n_merges = N - n_clusters
    else:
        # Keep all merges where merge height <= height
        n_merges = int(np.sum(Z[:, 2] <= height))

    # 3. Disjoint-Set / Union-Find structure to trace cluster membership
    parent = list(range(2 * N))

    def find(i):
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        for node in path:
            parent[node] = i
        return i

    # Apply the first n_merges
    for t in range(n_merges):
        a = int(Z[t, 0])
        b = int(Z[t, 1])
        new_id = N + t
        root_a = find(a)
        root_b = find(b)
        parent[root_a] = new_id
        parent[root_b] = new_id

    # 4. Find the root for each original object 0 .. N - 1
    roots = [find(i) for i in range(N)]

    # 5. Assign labels 0, 1, ... in order of first appearance
    root_to_label = {}
    labels = np.empty(N, dtype=int)
    current_label = 0

    for i, r in enumerate(roots):
        if r not in root_to_label:
            root_to_label[r] = current_label
            current_label += 1
        labels[i] = root_to_label[r]

    return labels

