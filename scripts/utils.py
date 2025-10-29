import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    SpectralClustering,
    DBSCAN
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

def plot_label_across_methods(label_name, y, embeddings_dict, palette, save_path):

    methods = list(embeddings_dict.keys())
    embeddings_ = list(embeddings_dict.values())
    label_values = y[label_name].values

    # --- Create grid (3x2 layout) ---
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=methods,
        specs=[[{}, {}],
               [{}, {}],
               [{}, None]],
        horizontal_spacing=0.08,
        vertical_spacing=0.10
    )

    # --- Add scatterplots per method ---
    for i, (name, X_emb) in enumerate(zip(methods, embeddings_)):
        row = i // 2 + 1
        col = i % 2 + 1
        df_plot = pd.DataFrame({
            "x": X_emb[:, 0],
            "y": X_emb[:, 1],
            label_name: label_values
        })

        # Add one scatter trace per unique label
        unique_vals = df_plot[label_name].unique()
        for val in unique_vals:
            sub = df_plot[df_plot[label_name] == val]
            color = palette.get(val, "#999999")  # fallback gray if unseen
            if len(sub) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=sub["x"],
                        y=sub["y"],
                        mode="markers",
                        marker=dict(size=5, color=color, opacity=0.85),
                        name=f"{label_name}: {val}",
                        showlegend=(i == 0)
                    ),
                    row=row, col=col
                )

    # --- Layout ---
    fig.update_layout(
        template="plotly_white",
        height=1000,
        width=900,
        title_text=f"Unsupervised Methods ({label_name})",
        title_x=0.5,
        font=dict(size=13),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.12,
            xanchor="center",
            x=0.5,
            title=None
        ),
        margin=dict(l=40, r=40, t=80, b=90)
    )

    # --- Export & show ---
    fig.write_html(save_path, include_plotlyjs="cdn")
    fig.show()


## part A
def fit_kmeans(X, k=5, seed=42):
    return KMeans(n_clusters=k, random_state=seed).fit_predict(X)

def fit_hierarchical(X, k=5, linkage="ward"):
    return AgglomerativeClustering(n_clusters=k, linkage=linkage).fit_predict(X)

def fit_spectral(X, k=5, seed=42):
    return SpectralClustering(n_clusters=k, affinity="nearest_neighbors", random_state=seed).fit_predict(X)

def fit_gmm(X, k=5, seed=42):
    return GaussianMixture(n_components=k, random_state=seed).fit_predict(X)

def fit_dbscan(X, eps=0.5, min_samples=5):
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)

def create_cluster_dict(X_dimension_reduction : np.ndarray):
    labels = (
        {
            "KMeans": fit_kmeans(X_dimension_reduction),
            "Spectral": fit_spectral(X_dimension_reduction),
            "GMM": fit_gmm(X_dimension_reduction),
            "DBSCAN": fit_dbscan(X_dimension_reduction)
        } 
        | 
        {
            f"Hierarchical | {link.capitalize()}": fit_hierarchical(X_dimension_reduction, linkage=link) 
                for link in ['ward','single','average','complete']
        }
        )
    return labels, X_dimension_reduction

def plot_results(labels: dict[str, np.ndarray], X: np.ndarray, path: str = None, title: str = None):
    fig, axes = plt.subplots(1, len(labels), figsize=(4 * len(labels), 4))
    for ax, (name, lbl) in zip(axes, labels.items()):
        ax.scatter(X[:, 0], X[:, 1], c=lbl, cmap='tab10', s=15)
        ax.set_title(name)
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    if title:
        fig.suptitle(title, fontsize=20, y=1.10)
    if path:
        plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.show()

## part B
def silhouette_analysis(X, method, params=None, k_min=2, k_max=15, random_state=42, verbose = True):
    if params is None:
        params = {}
    scores = {}
    # fix random_state outside loop and randomize but reproducible inside loop
    rng = np.random.default_rng(random_state) 
    for k in range(k_min, k_max + 1):
        try:
        # handle KMeans, Spectral, Agglomerative
            model = method(n_clusters=k, random_state=rng.integers(1e6), **params)
        except TypeError:
            try:
            # handle GaussianMixture
                model = method(n_components=k, random_state=rng.integers(1e6), **params)
            except TypeError:
            # methods without K (e.g. DBSCAN)
                model = method(**params)

        labels = model.fit_predict(X)

        if len(set(labels)) <= 1: # skip if only one cluster (silhouette undefined)
            continue

        score = silhouette_score(X, labels)
        scores[k] = score
        if verbose:
            print(f"K={k:2d} | silhouette={score:.4f}")

    best_k = max(scores, key=scores.get)
    best_score = scores[best_k]
    if verbose:
        print(f"\nBest K = {best_k} with silhouette score = {best_score:.4f}")

    return best_k, scores

def plot_silhouette(scores, best_k, title="Silhouette Analysis"):
    plt.figure(figsize=(6,4))
    plt.plot(list(scores.keys()), list(scores.values()), marker='o', linestyle='-')
    plt.axvline(best_k, color='red', linestyle='--', label=f"Best K = {best_k}")
    plt.title(title)
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.legend()
    plt.grid(True)
    plt.show()

# not robust - relativelyhardcoded for specific case
def agglomerative_jitter_silhouette_scores(X, random_state, linkage):
    rng = np.random.default_rng(42)
    scores_jittered = []

    for k in range(2, 15):
        # add small Gaussian noise before fitting
        X_jittered = X + rng.normal(0, 1e-3, X.shape)
        model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
        labels = model.fit_predict(X_jittered)
        if len(set(labels)) > 1:
            scores_jittered.append(silhouette_score(X_jittered, labels))
        else:
            scores_jittered.append(np.nan)

    plt.figure(figsize=(6,4))
    plt.plot(range(2, 15), scores_jittered, marker='o')
    plt.axvline(2, color='red', linestyle='--', label="Best K ≈ 2")
    plt.title(f"Agglomerative ({linkage.capitalize()}) Silhouette (UMAP) with Random Perturbation")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.legend()
    plt.grid(True)
    plt.show()