# === Core scientific stack ===
import numpy as np
import pandas as pd

# === Machine learning & clustering ===
from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    SpectralClustering,
    DBSCAN
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

# === Hierarchical utilities (for dendrograms, cophenetic distances, etc.) ===
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist

# === Plotting & visualization ===
import matplotlib.pyplot as plt
import seaborn as sns

# === Typing and utilities ===
from typing import List, Tuple, Optional

def consensus_matrix_kmeans(
        data : np.ndarray,
        K_values : List[int] = [2, 3, 4, 5, 6, 7], 
        iterations : int = 100, 
        random_state : int = 42
    ):
    n_samples = data.shape[0]
    rng = np.random.default_rng(random_state)
    stability_scores = []
    fig, axes = plt.subplots(1, len(K_values), figsize=(3*len(K_values),3))
    axes = axes.flatten()
    for idx, K in enumerate(K_values):
        consensus_matrix = np.zeros((n_samples, n_samples))
        sampled_matrix = np.zeros((n_samples, n_samples))
        for n in range(iterations):
            train_idx = rng.choice(n_samples, size=int(0.7 * n_samples), replace=False)
            X_train = data[train_idx]
            km = KMeans(n_clusters=K, init="k-means++", random_state=int(rng.integers(1e6)), n_init=1)
            cluster_labels = km.fit_predict(X_train)
            sampled_matrix[np.ix_(train_idx, train_idx)] += 1
            co_members = np.equal.outer(cluster_labels, cluster_labels)
            consensus_matrix[np.ix_(train_idx, train_idx)] += co_members
        with np.errstate(divide='ignore', invalid='ignore'):
            consensus_matrix = np.divide(consensus_matrix, sampled_matrix, where=(sampled_matrix != 0))
            consensus_matrix = np.nan_to_num(consensus_matrix)
        final_km = KMeans(n_clusters=K, n_init=10, random_state=42)
        final_clusters = final_km.fit_predict(data)
        order_idx = np.argsort(final_clusters)
        consensus_sorted = consensus_matrix[order_idx][:, order_idx]
        sns.heatmap(consensus_sorted, ax=axes[idx], cmap='viridis',
                    xticklabels=False, yticklabels=False, cbar=False)
        axes[idx].set_title(f'Consensus Matrix (K={K})')
        upper = np.triu_indices(n_samples, k=1)
        stability_score = np.mean(consensus_matrix[upper] * (1 - consensus_matrix[upper]))
        stability_scores.append(stability_score)
    fig.suptitle('K-Means', fontsize=20, y=1.02)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 3))
    plt.plot(K_values, stability_scores, marker='o', color='b', label='Stability Score')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Stability Score (Avg. Variance)')
    plt.title('Cluster Stability vs. K (K-Means Bootstrap Consensus)')
    plt.legend()
    plt.grid(True)
    plt.show()

def consensus_matrix_gmm(
        data: np.ndarray,
        K_values: List[int] = [2, 3, 4, 5, 6, 7],
        iterations: int = 100,
        random_state: int = 42
    ):
    n_samples = data.shape[0]
    rng = np.random.default_rng(random_state)
    stability_scores = []

    fig, axes = plt.subplots(1, len(K_values), figsize=(3 * len(K_values), 3))
    axes = axes.flatten()

    for idx, K in enumerate(K_values):
        consensus_matrix = np.zeros((n_samples, n_samples))
        sampled_matrix = np.zeros((n_samples, n_samples))

        for n in range(iterations):
            train_idx = rng.choice(n_samples, size=int(0.7 * n_samples), replace=False)
            X_train = data[train_idx]

            # --- Gaussian Mixture model instead of KMeans ---
            gmm = GaussianMixture(n_components=K, covariance_type="full", random_state=int(rng.integers(1e6)))
            cluster_labels = gmm.fit_predict(X_train)

            sampled_matrix[np.ix_(train_idx, train_idx)] += 1
            co_members = np.equal.outer(cluster_labels, cluster_labels)
            consensus_matrix[np.ix_(train_idx, train_idx)] += co_members

        # --- normalize consensus matrix ---
        with np.errstate(divide='ignore', invalid='ignore'):
            consensus_matrix = np.divide(consensus_matrix, sampled_matrix, where=(sampled_matrix != 0))
            consensus_matrix = np.nan_to_num(consensus_matrix)

        # --- final GMM for sorting and visualization ---
        final_gmm = GaussianMixture(n_components=K, covariance_type="full", random_state=42)
        final_clusters = final_gmm.fit_predict(data)
        order_idx = np.argsort(final_clusters)
        consensus_sorted = consensus_matrix[order_idx][:, order_idx]

        # --- plot consensus heatmap ---
        sns.heatmap(consensus_sorted, ax=axes[idx], cmap='viridis',
                    xticklabels=False, yticklabels=False, cbar=False)
        axes[idx].set_title(f'Consensus Matrix (K={K})')

        # --- compute stability score ---
        upper = np.triu_indices(n_samples, k=1)
        stability_score = np.mean(consensus_matrix[upper] * (1 - consensus_matrix[upper]))
        stability_scores.append(stability_score)

    fig.suptitle('Gaussian Mixture Models (GMM)', fontsize=20, y=1.02)
    plt.tight_layout()
    plt.show()

    # --- stability curve ---
    plt.figure(figsize=(6, 3))
    plt.plot(K_values, stability_scores, marker='o', color='b', label='Stability Score')
    plt.xlabel('Number of Components (K)')
    plt.ylabel('Stability Score (Avg. Variance)')
    plt.title('Cluster Stability vs. K (GMM Bootstrap Consensus)')
    plt.legend()
    plt.grid(True)
    plt.show()

def consensus_matrix_spectral(
        data: np.ndarray,
        K_values: List[int] = [2, 3, 4, 5, 6, 7],
        iterations: int = 100,
        random_state: int = 42
    ):
    n_samples = data.shape[0]
    rng = np.random.default_rng(random_state)
    stability_scores = []

    fig, axes = plt.subplots(1, len(K_values), figsize=(3 * len(K_values), 3))
    axes = axes.flatten()

    for idx, K in enumerate(K_values):
        consensus_matrix = np.zeros((n_samples, n_samples))
        sampled_matrix = np.zeros((n_samples, n_samples))

        for n in range(iterations):
            train_idx = rng.choice(n_samples, size=int(0.7 * n_samples), replace=False)
            X_train = data[train_idx]

            # --- Spectral clustering model ---
            sc = SpectralClustering(
                n_clusters=K,
                affinity='nearest_neighbors',  # stable option for UMAP/t-SNE embeddings
                assign_labels='kmeans',
                random_state=int(rng.integers(1e6))
            )
            cluster_labels = sc.fit_predict(X_train)

            # Update consensus matrices
            sampled_matrix[np.ix_(train_idx, train_idx)] += 1
            co_members = np.equal.outer(cluster_labels, cluster_labels)
            consensus_matrix[np.ix_(train_idx, train_idx)] += co_members

        # --- Normalize consensus matrix ---
        with np.errstate(divide='ignore', invalid='ignore'):
            consensus_matrix = np.divide(consensus_matrix, sampled_matrix, where=(sampled_matrix != 0))
            consensus_matrix = np.nan_to_num(consensus_matrix)

        # --- Final spectral clustering on full data (for ordering) ---
        final_sc = SpectralClustering(
            n_clusters=K,
            affinity='nearest_neighbors',
            assign_labels='kmeans',
            random_state=42
        )
        final_clusters = final_sc.fit_predict(data)
        order_idx = np.argsort(final_clusters)
        consensus_sorted = consensus_matrix[order_idx][:, order_idx]

        # --- Plot heatmap ---
        sns.heatmap(consensus_sorted, ax=axes[idx], cmap='viridis',
                    xticklabels=False, yticklabels=False, cbar=False)
        axes[idx].set_title(f'Consensus Matrix (K={K})')

        # --- Stability score ---
        upper = np.triu_indices(n_samples, k=1)
        stability_score = np.mean(consensus_matrix[upper] * (1 - consensus_matrix[upper]))
        stability_scores.append(stability_score)

    fig.suptitle('Spectral Clustering', fontsize=20, y=1.02)
    plt.tight_layout()
    plt.show()

    # --- Stability curve ---
    plt.figure(figsize=(6, 3))
    plt.plot(K_values, stability_scores, marker='o', color='b', label='Stability Score')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Stability Score (Avg. Variance)')
    plt.title('Cluster Stability vs. K (Spectral Bootstrap Consensus)')
    plt.legend()
    plt.grid(True)
    plt.show()

def consensus_matrix_hierarchical(
        data: np.ndarray,
        linkage: str = "ward", 
        K_values: List[int] = [2, 3, 4, 5, 6, 7],
        iterations: int = 100,
        random_state: int = 42
    ):
    n_samples = data.shape[0]
    rng = np.random.default_rng(random_state)
    stability_scores = []

    # --- prepare figure grid ---
    fig, axes = plt.subplots(1, len(K_values), figsize=(3 * len(K_values), 3))
    axes = axes.flatten()

    for idx, K in enumerate(K_values):
        consensus_matrix = np.zeros((n_samples, n_samples))
        sampled_matrix = np.zeros((n_samples, n_samples))

        for n in range(iterations):
            # --- bootstrap sample ---
            train_idx = rng.choice(n_samples, size=int(0.7 * n_samples), replace=False)
            X_train = data[train_idx]

            # --- Agglomerative model ---
            model = AgglomerativeClustering(n_clusters=K, linkage=linkage)
            cluster_labels = model.fit_predict(X_train)

            # update matrices
            sampled_matrix[np.ix_(train_idx, train_idx)] += 1
            co_members = np.equal.outer(cluster_labels, cluster_labels)
            consensus_matrix[np.ix_(train_idx, train_idx)] += co_members

        # --- normalize consensus matrix ---
        with np.errstate(divide='ignore', invalid='ignore'):
            consensus_matrix = np.divide(consensus_matrix, sampled_matrix, where=(sampled_matrix != 0))
            consensus_matrix = np.nan_to_num(consensus_matrix)

        # --- final fit for sorting ---
        final_model = AgglomerativeClustering(n_clusters=K, linkage=linkage)
        final_clusters = final_model.fit_predict(data)
        order_idx = np.argsort(final_clusters)
        consensus_sorted = consensus_matrix[order_idx][:, order_idx]

        # --- plot consensus heatmap ---
        sns.heatmap(consensus_sorted, ax=axes[idx], cmap='viridis',
                    xticklabels=False, yticklabels=False, cbar=False)
        axes[idx].set_title(f'K={K}')

        # --- stability score ---
        upper = np.triu_indices(n_samples, k=1)
        stability_score = np.mean(consensus_matrix[upper] * (1 - consensus_matrix[upper]))
        stability_scores.append(stability_score)

    # --- finalize figure ---
    fig.suptitle(f'Agglomerative Clustering (linkage="{linkage}")', fontsize=20, y=1.02)
    plt.tight_layout()
    plt.show()

    # --- plot stability curve ---
    plt.figure(figsize=(6, 3))
    plt.plot(K_values, stability_scores, marker='o', color='b', label='Stability Score')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Stability Score (Avg. Variance)')
    plt.title(f'Stability vs. K ({linkage.capitalize()} Linkage)')
    plt.legend()
    plt.grid(True)
    plt.show()

def consensus_matrix_dbscan(
        data: np.ndarray,
        eps: float = 0.35,
        min_samples: int = 5,
        iterations: int = 100,
        random_state: int = 42,
        plot : bool = False
    ):
    """
    Bootstrap-based consensus matrix for DBSCAN clustering stability.
    """
    n_samples = data.shape[0]
    rng = np.random.default_rng(random_state)
    stability_scores = []
    consensus_matrix = np.zeros((n_samples, n_samples))
    sampled_matrix = np.zeros((n_samples, n_samples))

    # --- Bootstrap iterations ---
    for n in range(iterations):
        train_idx = rng.choice(n_samples, size=int(0.7 * n_samples), replace=False)
        X_train = data[train_idx]

        model = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = model.fit_predict(X_train)

        sampled_matrix[np.ix_(train_idx, train_idx)] += 1

        # Only count non-noise points in same cluster
        co_members = np.equal.outer(cluster_labels, cluster_labels)
        valid_mask = (cluster_labels[:, None] != -1) & (cluster_labels[None, :] != -1)
        consensus_matrix[np.ix_(train_idx, train_idx)] += (co_members & valid_mask)

    # --- Normalize ---
    with np.errstate(divide='ignore', invalid='ignore'):
        consensus_matrix = np.divide(consensus_matrix, sampled_matrix, where=(sampled_matrix != 0))
        consensus_matrix = np.nan_to_num(consensus_matrix)

    # --- Compute stability score ---
    upper = np.triu_indices(n_samples, k=1)
    stability_score = np.mean(consensus_matrix[upper] * (1 - consensus_matrix[upper]))
    if plot:
        # --- Plot ---
        plt.figure(figsize=(5, 4))
        sns.heatmap(consensus_matrix, cmap="viridis", xticklabels=False, yticklabels=False)
        plt.title(f"DBSCAN Consensus (eps={eps}, min_samples={min_samples})\nStability={stability_score:.4f}")
        plt.show()

    return stability_score, consensus_matrix

def dbscan_stability_grid(data, eps_values, min_samples_values, iterations=50):
    results = np.zeros((len(eps_values), len(min_samples_values)))
    noise_fracs = np.zeros_like(results)

    for i, eps in enumerate(eps_values):
        for j, ms in enumerate(min_samples_values):
            # --- Compute stability ---
            stability, _ = consensus_matrix_dbscan(
                data, eps=eps, min_samples=ms, iterations=iterations
            )
            results[i, j] = stability

            # --- Compute noise fraction (single DBSCAN fit) ---
            model = DBSCAN(eps=eps, min_samples=ms)
            labels = model.fit_predict(data)
            noise_fracs[i, j] = np.mean(labels == -1)

    # --- Plot both heatmaps side by side ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    sns.heatmap(
        results, annot=True, fmt=".3f", 
        xticklabels=min_samples_values, yticklabels=eps_values, 
        cmap="viridis", ax=axes[0]
    )
    axes[0].set_title("DBSCAN Stability Heatmap (variance measure)")
    axes[0].set_xlabel("min_samples"); axes[0].set_ylabel("eps")

    sns.heatmap(
        noise_fracs, annot=True, fmt=".2f",
        xticklabels=min_samples_values, yticklabels=eps_values,
        cmap="magma", ax=axes[1]
    )
    axes[1].set_title("DBSCAN Noise Fraction Heatmap")
    axes[1].set_xlabel("min_samples"); axes[1].set_ylabel("")

    plt.tight_layout()
    plt.show()