"""
Reusable clustering functions for the DCD pipeline.

This module contains all clustering logic: sparsification, distance computation,
clustering methods (hierarchical agglomerative, hierarchical divisive, k-means),
dimensionality reduction (PCA, Truncated SVD), and representative finding.

Used by experiments/dcd/03_run_clustering.py (cluster stage).
"""

import json
import os
from itertools import product
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import silhouette_score
from tqdm import tqdm


# ---------------------- JSON Encoder ----------------------

class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return super().default(obj)


# ---------------------- Sparsification ----------------------

def _pdist_jaccard(data: np.ndarray) -> np.ndarray:
    """Compute jaccard pairwise distances without a float64 copy.

    scipy.spatial.distance.pdist with metric='jaccard' dispatches to an
    optimised C implementation when the input dtype is bool, avoiding the
    O(n*d) float64 conversion that occurs for uint8 inputs.  Reinterpreting
    the bytes as bool_ is zero-copy for binary (0/1) uint8 arrays.
    """
    if data.dtype != np.bool_:
        data = data.view(np.bool_)
    return pdist(data, metric="jaccard")


def sparsify_binary(e: np.ndarray, gamma: float) -> np.ndarray:
    """Binarize an edge score vector by keeping top edges that account for
    gamma fraction of total absolute attribution.

    Args:
        e: Raw edge attribution scores, shape (n_edges,).
        gamma: Fraction of total attribution to retain (e.g. 0.99).

    Returns:
        Binary mask of shape (n_edges,) with dtype uint8.
    """
    e_abs = np.abs(e)
    total = np.sum(e_abs)

    if total == 0:
        return np.zeros_like(e, dtype=np.uint8)

    sorted_idx = np.argsort(e_abs)[::-1]
    cumsum = np.cumsum(e_abs[sorted_idx])
    cutoff = np.searchsorted(cumsum, gamma * total) + 1

    mask = np.zeros_like(e, dtype=np.uint8)
    mask[sorted_idx[:cutoff]] = 1
    return mask


def binarize_matrix(edge_scores: np.ndarray, gamma: float) -> np.ndarray:
    """Apply sparsify_binary to each row of an edge score matrix.

    Args:
        edge_scores: Raw edge scores, shape (n_samples, n_edges).
        gamma: Fraction of total attribution to retain per example.

    Returns:
        Binary matrix of shape (n_samples, n_edges) with dtype uint8.
    """
    n_samples = edge_scores.shape[0]
    binary = np.zeros_like(edge_scores, dtype=np.uint8)
    for i in tqdm(range(n_samples), desc="Binarizing edge scores", leave=False):
        binary[i] = sparsify_binary(edge_scores[i], gamma)
    return binary


def sparsify_matrix(edge_scores: np.ndarray, gamma: float) -> np.ndarray:
    """Zero out edges below the gamma cumulative-attribution threshold,
    but keep raw magnitudes for the retained edges.

    Unlike binarize_matrix, the output preserves the original float values
    of the edges that pass the threshold — useful when feeding into
    dimensionality reduction where edge magnitude carries signal.

    Args:
        edge_scores: Raw edge scores, shape (n_samples, n_edges).
        gamma: Fraction of total attribution to retain per example.

    Returns:
        Sparse float matrix of shape (n_samples, n_edges), same dtype as input.
    """
    n_samples = edge_scores.shape[0]
    out = np.zeros_like(edge_scores)
    for i in tqdm(range(n_samples), desc="Sparsifying edge scores", leave=False):
        row = edge_scores[i]
        mask = sparsify_binary(row, gamma).astype(bool)
        out[i, mask] = row[mask]
    return out

def save_cluster_examples(
    groups: np.ndarray,
    train_data: pd.DataFrame,
    save_dir: str,
    k: int,
) -> None:
    """Save the examples belonging to each cluster as separate CSVs.

    Args:
        groups: Cluster assignments, shape (n_samples,).
        train_data: Original training data with prompt_type column.
        save_dir: Directory for this clustering method.
        k: Number of clusters.
    """
    k_dir = os.path.join(save_dir, f"k_{k}")
    os.makedirs(k_dir, exist_ok=True)

    for cid in np.unique(groups):
        cluster_data = train_data[groups == cid]
        cluster_data.to_csv(
            os.path.join(k_dir, f"cluster_{cid}.csv"),
            index=False,
        )


# ---------------------- Dimensionality Reduction ----------------------

def reduce_dimensions(
    data: np.ndarray,
    method: Optional[str],
    n_components: Optional[int],
    random_state: int = 42,
) -> Tuple[np.ndarray, object]:
    """Apply dimensionality reduction to the data.

    Args:
        data: Input matrix, shape (n_samples, n_features).
        method: One of "pca", "truncated_svd", or None (no reduction).
        n_components: Number of components to keep. Required if method is not None.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (reduced_data, fitted_reducer). If method is None, returns
        (data, None).
    """
    if method is None:
        return data, None

    if n_components is None:
        raise ValueError(f"n_components required when using reduction method '{method}'")

    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=random_state)
    elif method == "truncated_svd":
        reducer = TruncatedSVD(n_components=n_components, random_state=random_state)
    else:
        raise ValueError(f"Unknown reduction method: {method}. Use 'pca' or 'truncated_svd'.")

    reduced = reducer.fit_transform(data)
    return reduced, reducer


# ---------------------- Clustering Methods ----------------------

def hierarchical_agglomerative(
    data: np.ndarray,
    max_clusters: int,
    distance_metric: str = "jaccard",
    linkage_method: str = "average",
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """Run hierarchical agglomerative clustering for k=2..max_clusters.

    Args:
        data: Input matrix, shape (n_samples, n_features). Should be binary
              (uint8) if using Jaccard distance.
        max_clusters: Maximum number of clusters to try.
        distance_metric: Distance metric for pdist (e.g. "jaccard", "euclidean", "cosine").
        linkage_method: Linkage method (e.g. "average", "complete", "single").
              Ward linkage requires Euclidean distance.

    Returns:
        Tuple of (results, Z) where results maps k -> cluster assignments array
        of shape (n_samples,), and Z is the scipy linkage matrix (n-1, 4).
    """
    dist_vec = _pdist_jaccard(data) if distance_metric == "jaccard" else pdist(data, metric=distance_metric)
    Z = linkage(dist_vec, method=linkage_method)

    results = {}
    for k in range(2, max_clusters + 1):
        results[k] = fcluster(Z, t=k, criterion="maxclust")

    return results, Z


def hierarchical_divisive(
    data: np.ndarray,
    max_clusters: int,
    distance_metric: str = "jaccard",
    linkage_method: str = "average",
) -> Dict[int, np.ndarray]:
    """Run top-down (divisive) hierarchical clustering for k=2..max_clusters.

    Recursively bisects the largest cluster using agglomerative clustering
    with k=2 at each step.

    Args:
        data: Input matrix, shape (n_samples, n_features). Should be binary
              (uint8) if using Jaccard distance.
        max_clusters: Maximum number of clusters to try.
        distance_metric: Distance metric for pdist.
        linkage_method: Linkage method for bisection steps.

    Returns:
        Dict mapping k -> cluster assignments array of shape (n_samples,).
    """
    n_samples = data.shape[0]

    # Start: all samples in one cluster
    labels = np.ones(n_samples, dtype=int)
    next_label = 2

    results = {}

    for k in tqdm(range(2, max_clusters + 1), desc="Divisive clustering k sweep", leave=False):
        # Find the largest cluster to split
        unique, counts = np.unique(labels, return_counts=True)
        largest_cluster = unique[np.argmax(counts)]
        largest_idx = np.where(labels == largest_cluster)[0]

        if len(largest_idx) < 2:
            # Cannot split a singleton; keep previous assignments for remaining k
            results[k] = labels.copy()
            continue

        # Bisect the largest cluster using agglomerative with k=2
        subset = data[largest_idx]
        dist_vec = _pdist_jaccard(subset) if distance_metric == "jaccard" else pdist(subset, metric=distance_metric)
        Z = linkage(dist_vec, method=linkage_method)
        sub_labels = fcluster(Z, t=2, criterion="maxclust")

        # Assign new labels: keep original label for one group, new label for the other
        new_labels = labels.copy()
        split_mask = sub_labels == 2
        new_labels[largest_idx[split_mask]] = next_label
        next_label += 1

        labels = new_labels
        results[k] = labels.copy()

    # Relabel each result to use consecutive integers starting from 1
    for k in results:
        unique_labels = np.unique(results[k])
        label_map = {old: new for new, old in enumerate(unique_labels, start=1)}
        results[k] = np.array([label_map[l] for l in results[k]])

    return results


def kmeans_clustering(
    data: np.ndarray,
    max_clusters: int,
    random_state: int = 42,
    return_inertias: bool = False,
) -> Union[Dict[int, np.ndarray], Tuple[Dict[int, np.ndarray], Dict[int, float]]]:
    """Run k-means clustering for k=2..max_clusters.

    Note: K-means requires continuous data (not binary). Apply dimensionality
    reduction before calling this on binarized edge scores.

    Args:
        data: Input matrix, shape (n_samples, n_features).
        max_clusters: Maximum number of clusters to try.
        random_state: Random seed for reproducibility.
        return_inertias: If True, also return a dict mapping k -> inertia value.

    Returns:
        Dict mapping k -> cluster assignments array of shape (n_samples,).
        Labels are 1-indexed to be consistent with hierarchical methods.
        If return_inertias is True, returns (labels_dict, inertias_dict).
    """
    labels_dict: Dict[int, np.ndarray] = {}
    inertias_dict: Dict[int, float] = {}
    for k in tqdm(range(2, max_clusters + 1), desc="K-means k sweep", leave=False):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(data)
        labels_dict[k] = labels + 1  # 1-indexed to match fcluster convention
        inertias_dict[k] = float(km.inertia_)

    if return_inertias:
        return labels_dict, inertias_dict
    return labels_dict


# ---------------------- Representatives ----------------------

def find_representatives(
    edge_scores_binary: np.ndarray,
    groups: np.ndarray,
    distance_metric: str = "jaccard",
    precomputed_dist: Optional[np.ndarray] = None,
) -> Dict[int, int]:
    """Find the medoid (most central example) in each cluster.

    The medoid is the example with the smallest sum of distances to all
    other examples in the same cluster.

    Args:
        edge_scores_binary: Binary edge score matrix, shape (n_samples, n_edges).
        groups: Cluster assignments, shape (n_samples,).
        distance_metric: Distance metric for pairwise computation.
        precomputed_dist: Optional precomputed square distance matrix of shape
            (n_samples, n_samples). When provided, sub-matrices are sliced
            directly instead of recomputing pairwise distances per cluster.

    Returns:
        Dict mapping cluster_id -> index of representative example.
    """
    reps = {}
    for cluster_id in np.unique(groups):
        idx = np.where(groups == cluster_id)[0]

        if len(idx) == 1:
            reps[int(cluster_id)] = int(idx[0])
            continue

        if precomputed_dist is not None:
            sub_dists = precomputed_dist[np.ix_(idx, idx)]
        else:
            circuits = edge_scores_binary[idx]
            sub_dists = squareform(_pdist_jaccard(circuits) if distance_metric == "jaccard" else pdist(circuits, metric=distance_metric))
        rep = idx[np.argmin(sub_dists.sum(axis=1))]
        reps[int(cluster_id)] = int(rep)

    return reps


# ---------------------- Evaluation ----------------------

def compute_silhouette(
    data: np.ndarray,
    groups: np.ndarray,
    distance_metric: str = "jaccard",
    precomputed_dist: Optional[np.ndarray] = None,
) -> float:
    """Compute silhouette score for a clustering result.

    Args:
        data: Input matrix used for clustering, shape (n_samples, n_features).
        groups: Cluster assignments, shape (n_samples,).
        distance_metric: Distance metric matching what was used for clustering.
        precomputed_dist: Optional precomputed square distance matrix of shape
            (n_samples, n_samples). When provided, used directly with
            metric="precomputed" to avoid recomputing pairwise distances.

    Returns:
        Silhouette score (float between -1 and 1).
    """
    n_unique = len(np.unique(groups))
    if n_unique < 2 or n_unique >= len(groups):
        return -1.0

    if precomputed_dist is not None:
        return silhouette_score(precomputed_dist, groups, metric="precomputed")
    return silhouette_score(data, groups, metric=distance_metric)

# ---------------------- Runner ----------------------

def run_clustering_method(
    edge_scores: np.ndarray,
    method_config: dict,
    global_config: dict,
) -> dict:
    """Run a single clustering method with its configuration.

    This is the main entry point called by the experiment script. It handles
    dimensionality reduction, clustering, representative finding, and
    silhouette evaluation.

    Args:
        edge_scores: Edge score matrix the caller has already preprocessed
            (raw, binarized, or sparsified), shape (n_samples, n_edges).
            This is the only input matrix — representatives and silhouette
            are computed in whatever space the caller passes in.
        method_config: Config for this specific method, containing:
            - name: str (e.g. "hierarchical-agglomerative")
            - reduction: Optional[str] ("pca", "truncated_svd", or null)
            - n_components: Optional[int or List[int]]
            - linkage: Optional[str] (for hierarchical methods)
            - preprocessing: Optional[str] — one of "raw", "binary", "sparse".
              Tells the runner which preprocessing the caller already applied,
              so the default distance metric can be chosen accordingly.
              Defaults to "raw".
        global_config: Global clustering config containing:
            - max_clusters: int
            - distance_metric: Optional[str] — if None, defaults to "jaccard"
              for binary preprocessing and "euclidean" otherwise.
            - sparsity_gamma: Optional[float]

    Returns:
        Dict with results for each k value, including cluster assignments,
        silhouette scores, representative indices, and cluster sizes.
    """
    method_name = method_config["name"]
    reduction = method_config.get("reduction", None)
    n_components_list = method_config.get("n_components", [None])
    max_clusters = global_config["max_clusters"]
    linkage_method = method_config.get("linkage", "average")
    random_state = method_config.get("random_state", 42)
    preprocessing = method_config.get("preprocessing", "raw")
    if preprocessing not in ("raw", "binary", "sparse"):
        raise ValueError(
            f"preprocessing must be 'raw', 'binary', or 'sparse', got {preprocessing!r}"
        )

    # Default distance metric: jaccard for binary, euclidean for raw/sparse.
    default_metric = "jaccard" if preprocessing == "binary" else "euclidean"
    distance_metric = global_config.get("distance_metric") or default_metric

    # Normalize n_components_list to always be a list
    if not isinstance(n_components_list, list):
        n_components_list = [n_components_list]

    all_results = {}

    for n_comp in n_components_list:
        # Determine the data to cluster and the appropriate distance metric
        if reduction is not None:
            clustering_data, _ = reduce_dimensions(
                edge_scores.astype(np.float32),
                method=reduction,
                n_components=n_comp,
            )
            # After reduction, data is continuous — use euclidean
            clustering_distance = "euclidean"
            result_key = f"{method_name}-{reduction}-{n_comp}"
        else:
            clustering_data = edge_scores
            clustering_distance = distance_metric
            result_key = method_name

        # Precompute distance matrix once — reused across all k evaluations
        # for both silhouette and representative selection.
        print(f"  Precomputing distance matrix ({clustering_distance})...")
        _pdist_cd = _pdist_jaccard(clustering_data) if clustering_distance == "jaccard" else pdist(clustering_data, metric=clustering_distance)
        dist_matrix = squareform(_pdist_cd)

        # Run the appropriate clustering algorithm
        inertias: Dict[int, float] = {}
        elbow_k: Optional[int] = None
        if "agglomerative" in method_name:
            cluster_results, linkage_Z = hierarchical_agglomerative(
                clustering_data,
                max_clusters=max_clusters,
                distance_metric=clustering_distance,
                linkage_method=linkage_method,
            )
            elbow_k = find_elbow_k_hierarchical(linkage_Z, k_min=2, k_max=max_clusters)
        elif "divisive" in method_name:
            cluster_results = hierarchical_divisive(
                clustering_data,
                max_clusters=max_clusters,
                distance_metric=clustering_distance,
                linkage_method=linkage_method,
            )
            # Pseudo-inertia: sum of within-cluster pairwise distances per k
            pseudo_inertias: Dict[int, float] = {}
            for k, groups in cluster_results.items():
                total = 0.0
                for cid in np.unique(groups):
                    idx = np.where(groups == cid)[0]
                    if len(idx) > 1:
                        sub = dist_matrix[np.ix_(idx, idx)]
                        total += float(sub.sum()) / 2.0
                pseudo_inertias[k] = total
            elbow_k = find_elbow_k(pseudo_inertias)
        elif "kmeans" in method_name:
            cluster_results, inertias = kmeans_clustering(
                clustering_data,
                max_clusters=max_clusters,
                random_state=random_state,
                return_inertias=True,
            )
        else:
            raise ValueError(f"Unknown clustering method: {method_name}")

        # Evaluate each k
        k_results = {}
        for k, groups in tqdm(cluster_results.items(), desc="Evaluating k values", leave=False):
            sil_score = compute_silhouette(
                clustering_data, groups,
                distance_metric=clustering_distance,
                precomputed_dist=dist_matrix,
            )

            # Representatives are found in the same space and metric used
            # for clustering, so the medoid is consistent with the distances
            # that drove the cluster assignments.
            reps = find_representatives(
                clustering_data, groups,
                distance_metric=clustering_distance,
                precomputed_dist=dist_matrix,
            )

            cluster_sizes = {
                int(cid): int(np.sum(groups == cid))
                for cid in np.unique(groups)
            }

            k_entry: dict = {
                "silhouette_score": sil_score,
                "cluster_sizes": cluster_sizes,
                "representatives": reps,
                "groups": groups,
            }
            if k in inertias:
                k_entry["inertia"] = inertias[k]

            k_results[k] = k_entry

        if elbow_k is not None:
            k_results["elbow_k"] = elbow_k

        all_results[result_key] = k_results

    return all_results


def save_clustering_results(
    results: dict,
    save_dir: str,
) -> None:
    """Save clustering results to disk.

    For each method, saves a JSON file with scores, cluster sizes, and
    representatives, plus a separate .npy file for cluster assignments
    (since they're arrays).

    Args:
        results: Output from run_clustering_method().
        save_dir: Base directory for saving results.
    """
    for method_key, k_results in results.items():
        method_dir = os.path.join(save_dir, method_key)
        os.makedirs(method_dir, exist_ok=True)

        # Separate groups (numpy arrays) from JSON-serializable data
        # Pop elbow_k first — it's a scalar, not a per-k dict
        elbow_k = k_results.pop("elbow_k", None)

        json_results = {}
        for k, data in k_results.items():
            groups = data.pop("groups")
            np.save(
                os.path.join(method_dir, f"groups_k{k}.npy"),
                groups,
            )
            json_results[f"{k}-clusters"] = data

        if elbow_k is not None:
            json_results["elbow_k"] = int(elbow_k)

        json_path = os.path.join(method_dir, "results.json")
        with open(json_path, "w") as f:
            json.dump(json_results, f, indent=2, cls=NumpyJSONEncoder)


# ---------------------- Hyperparameter Grid Utilities ----------------------

# Fixed iteration order and combo-name prefixes for list-valued params.
_LIST_PARAM_ORDER = ["sparsity_gamma", "linkage", "n_components"]
_COMBO_NAME_PREFIX = {
    "sparsity_gamma": "gamma",
    "linkage": "linkage-",
    "n_components": "ncomp",
}
_PARAM_RENAME: dict = {}


def get_hyperparameter_combos(method_cfg: dict) -> List[dict]:
    """Generate all single-valued combinations of list-valued hyperparameters.

    Iterates over the Cartesian product of every list-valued field in
    method_cfg (sparsity_gamma, linkage, n_components, random_seeds).
    Scalar fields are copied unchanged into every combo.

    Args:
        method_cfg: Method config dict from clustering_grid.yaml, where
            list-valued fields represent axes to sweep.

    Returns:
        List of dicts, one per combination. Each dict contains resolved
        single values for all fields, plus a ``combo_name`` string that
        encodes the varying parameters.
    """
    scalar: dict = {}
    list_params: Dict[str, list] = {}

    for key, val in method_cfg.items():
        if isinstance(val, list):
            list_params[key] = val
        else:
            scalar[key] = val

    # random_seeds is never swept: resolve to a single random_state scalar.
    # Use the first seed if a list is provided, the scalar value if given
    # directly, or 42 if absent.  Never add to combo_name.
    if "random_seeds" in list_params:
        scalar["random_state"] = list_params.pop("random_seeds")[0]
    elif "random_seeds" in scalar:
        scalar["random_state"] = scalar.pop("random_seeds")
    else:
        scalar["random_state"] = 42

    # Iterate in fixed order so combo_name is deterministic
    ordered_keys = [k for k in _LIST_PARAM_ORDER if k in list_params]
    # Any extra list-valued keys not in the known order (future-proofing)
    ordered_keys += [k for k in list_params if k not in _LIST_PARAM_ORDER]

    combos: List[dict] = []
    for combo_vals in product(*[list_params[k] for k in ordered_keys]):
        combo = dict(scalar)
        name_parts: List[str] = []
        for key, val in zip(ordered_keys, combo_vals):
            out_key = _PARAM_RENAME.get(key, key)
            combo[out_key] = val
            prefix = _COMBO_NAME_PREFIX.get(key, key)
            name_parts.append(f"{prefix}{val}")
        combo["combo_name"] = "_".join(name_parts)
        combos.append(combo)

    return combos


def find_elbow_k(inertias: Dict[int, float]) -> int:
    """Find the elbow point in a k-means inertia curve.

    Uses a geometric approach: normalises the (k, inertia) points to [0, 1]²,
    draws a straight line from the first to the last point, and returns the k
    with the maximum perpendicular distance from that line.

    Args:
        inertias: Dict mapping k (int) -> inertia (float).

    Returns:
        The k value at the elbow.
    """
    ks = sorted(inertias.keys())
    if len(ks) < 2:
        return ks[0]

    ks_arr = np.array(ks, dtype=float)
    vals = np.array([inertias[k] for k in ks], dtype=float)

    # Normalise both axes to [0, 1]
    k_range = ks_arr[-1] - ks_arr[0]
    v_range = vals[0] - vals[-1]  # inertia decreases, so start > end
    ks_norm = (ks_arr - ks_arr[0]) / (k_range if k_range > 0 else 1.0)
    vals_norm = (vals - vals[-1]) / (v_range if v_range > 0 else 1.0)

    # Line from P1=(0, 1) to P2=(1, 0) after normalisation; general form: x + y = 1
    # Perpendicular distance from (x0, y0) to ax + by + c = 0 is |ax0+by0+c|/sqrt(a²+b²)
    # Here: 1*x + 1*y - 1 = 0  →  a=1, b=1, c=-1
    distances = np.abs(ks_norm + vals_norm - 1.0) / np.sqrt(2.0)

    return int(ks[int(np.argmax(distances))])


def _within_cluster_dispersion(data: np.ndarray, groups: np.ndarray) -> float:
    """Pooled within-cluster sum of squared distances to centroid.

    W(k) = sum_r (1/n_r) * sum_{i in r} ||x_i - mu_r||^2

    Args:
        data: Feature matrix, shape (n_samples, n_features).
        groups: Cluster assignments, shape (n_samples,).

    Returns:
        W(k) value (non-negative float).
    """
    W = 0.0
    for cid in np.unique(groups):
        mask = groups == cid
        cluster_data = data[mask].astype(np.float64)
        n_r = len(cluster_data)
        if n_r <= 1:
            continue
        centroid = cluster_data.mean(axis=0)
        W += float(np.sum((cluster_data - centroid) ** 2)) / n_r
    return W


def compute_gap_statistic(
    data: np.ndarray,
    groups_by_k: Dict[int, np.ndarray],
    n_refs: int = 20,
    random_state: int = 42,
) -> Tuple[Dict[int, dict], int]:
    """Compute the gap statistic and select the optimal k.

    Implements Tibshirani, Walther & Hastie (2001). For each k, computes
    Gap(k) = E*[log W_ref(k)] - log W(k), where W is the pooled within-cluster
    dispersion and the expectation is over B uniform reference datasets drawn in
    the bounding box of the observed data. Reference datasets are clustered with
    k-means regardless of the original algorithm (standard post-hoc approximation).

    The optimal k is the smallest k satisfying the one-standard-error rule:
        Gap(k) >= Gap(k+1) - SE(k+1)
    where SE(k) = sd(k) * sqrt(1 + 1/B).

    Note: This function expects continuous (e.g. PCA/SVD-reduced) data.
    Methods that cluster in the original binary space should be skipped.

    Args:
        data: Continuous feature matrix used for clustering,
              shape (n_samples, n_features).
        groups_by_k: Dict mapping k -> cluster assignment array, as produced
                     by loading the saved groups_k{k}.npy files.
        n_refs: Number of uniform reference datasets (B in the paper).
        random_state: Seed for reference data generation and k-means runs.

    Returns:
        Tuple of:
        - gap_results: Dict mapping k -> {"gap": float, "se": float,
          "log_wk": float}.
        - optimal_k: The gap-statistic-selected k (int).
    """
    rng = np.random.default_rng(random_state)
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    k_values = sorted(groups_by_k.keys())

    log_wks: Dict[int, float] = {}
    for k in k_values:
        wk = _within_cluster_dispersion(data, groups_by_k[k])
        log_wks[k] = float(np.log(max(wk, 1e-10)))

    ref_log_wks: Dict[int, List[float]] = {k: [] for k in k_values}
    for _ in range(n_refs):
        ref_data = rng.uniform(data_min, data_max, size=data.shape).astype(np.float32)
        ref_seed = int(rng.integers(1_000_000))
        for k in k_values:
            km = KMeans(n_clusters=k, random_state=ref_seed, n_init=5)
            ref_groups = km.fit_predict(ref_data)
            wk_ref = _within_cluster_dispersion(ref_data, ref_groups)
            ref_log_wks[k].append(float(np.log(max(wk_ref, 1e-10))))

    gap_results: Dict[int, dict] = {}
    for k in k_values:
        ref_vals = np.array(ref_log_wks[k])
        gap = float(ref_vals.mean() - log_wks[k])
        se = float(ref_vals.std(ddof=0) * np.sqrt(1.0 + 1.0 / n_refs))
        gap_results[k] = {"gap": gap, "se": se, "log_wk": log_wks[k]}

    optimal_k = k_values[-1]
    for i, k in enumerate(k_values[:-1]):
        k_next = k_values[i + 1]
        if gap_results[k]["gap"] >= gap_results[k_next]["gap"] - gap_results[k_next]["se"]:
            optimal_k = k
            break

    return gap_results, optimal_k


def find_elbow_k_hierarchical(Z: np.ndarray, k_min: int = 2, k_max: int = 20) -> int:
    """Find the optimal k from a hierarchical agglomerative linkage matrix.

    Uses the largest gap between consecutive merge heights in the dendrogram.
    A large gap means the next merge would join clusters that are far apart —
    cutting there preserves the most distinct groupings.

    Args:
        Z: Linkage matrix from scipy.cluster.hierarchy.linkage, shape (n-1, 4).
            Z[:, 2] contains the merge heights in ascending order.
        k_min: Minimum number of clusters to consider.
        k_max: Maximum number of clusters to consider.

    Returns:
        The k value with the largest merge-height gap.
    """
    n = len(Z) + 1  # number of samples
    heights = Z[:, 2]

    best_k = k_min
    best_gap = -1.0

    for k in range(k_min, k_max + 1):
        # For k clusters, the last merge performed is at index n-k-1.
        # The merge that would reduce to k-1 clusters is at index n-k.
        idx_below = n - k - 1  # height just inside the cut
        idx_above = n - k      # height just above the cut
        if idx_below < 0 or idx_above >= len(heights):
            continue
        gap = heights[idx_above] - heights[idx_below]
        if gap > best_gap:
            best_gap = gap
            best_k = k

    return best_k