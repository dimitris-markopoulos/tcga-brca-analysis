import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    SpectralClustering,
    DBSCAN
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.neighbors import NearestNeighbors

import warnings
from sklearn.manifold import spectral_embedding

import warnings

warnings.filterwarnings(
    "ignore",
    message="Graph is not fully connected",
    category=UserWarning,
    module="sklearn.manifold._spectral_embedding"
)



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

def create_cluster_dict(X_dimension_reduction : np.ndarray, K : int):
    labels = (
        {
            "KMeans": fit_kmeans(X_dimension_reduction, K),
            "Spectral": fit_spectral(X_dimension_reduction, K),
            "GMM": fit_gmm(X_dimension_reduction, K),
        } 
        | 
        {
            f"Hierarchical | {link.capitalize()}": fit_hierarchical(X_dimension_reduction, K, linkage=link) 
                for link in ['ward','single','average','complete']
        }
        |
        {"DBSCAN (independent of K)": fit_dbscan(X_dimension_reduction,eps=0.35,min_samples=5)}
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

def predictive_ari_cv(X, cluster_method, k_values, n_splits=5, random_state=42, **cluster_kwargs):
    n_samples, p = X.shape
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    pred_ari_means, pred_ari_stds = [], []

    for k in k_values:
        aris = []
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            # --- clustering on train and test ---
            if cluster_method == "KMeans":
                model_train = KMeans(n_clusters=k, n_init=10, random_state=fold_idx)
                model_test  = KMeans(n_clusters=k, n_init=10, random_state=fold_idx + 1000)
            elif cluster_method == "GMM":
                model_train = GaussianMixture(n_components=k, random_state=fold_idx)
                model_test  = GaussianMixture(n_components=k, random_state=fold_idx + 1000)
            elif cluster_method == "Spectral":
                model_train = SpectralClustering(n_clusters=k, random_state=fold_idx)
                model_test  = SpectralClustering(n_clusters=k, random_state=fold_idx + 1000)
            elif cluster_method == "Hierarchical":
                linkage = cluster_kwargs.get("linkage", "ward")
                model_train = AgglomerativeClustering(n_clusters=k, linkage=linkage)
                model_test  = AgglomerativeClustering(n_clusters=k, linkage=linkage)
            else:
                raise ValueError("Unsupported method.")

            labels_train = model_train.fit_predict(X[train_idx])
            labels_test = model_test.fit_predict(X[test_idx])

            # --- train random forest to generalize ---
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=p,
                max_features=int(np.round(np.sqrt(p))),
                random_state=fold_idx
            )
            rf.fit(X[train_idx], labels_train)
            labels_test_pred = rf.predict(X[test_idx])

            # --- compute predictive ARI ---
            aris.append(adjusted_rand_score(labels_test, labels_test_pred))

        pred_ari_means.append(np.mean(aris))
        pred_ari_stds.append(np.std(aris, ddof=1))

    best_k = k_values[int(np.argmax(pred_ari_means))]

    # --- plotting ---
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        k_values, pred_ari_means, yerr=pred_ari_stds,
        fmt='-o', capsize=4, label='Predictive ARI (mean ± SD)', color='blue'
    )

    # red dashed line for best K with legend entry
    plt.axvline(
        best_k, linestyle='--', alpha=0.8, color='red',
        label=f'Best K = {best_k}'
    )

    # Proper title handling for hierarchical linkages
    title = f'Predictive ARI (5-fold CV): {cluster_method}'
    if cluster_method == "Hierarchical":
        title += f' (linkage="{cluster_kwargs.get("linkage", "ward")}")'

    plt.title(title)
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Mean ± SD ARI')
    plt.legend(frameon=True, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return best_k, pred_ari_means, pred_ari_stds

def predictive_ari_dbscan(X, eps_values, min_samples_values, n_splits=5, random_state=42):
    """
    Evaluate DBSCAN generalizability using Predictive ARI across eps and min_samples values.
    """
    n_samples, p = X.shape
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    ari_results = np.zeros((len(eps_values), len(min_samples_values)))

    for i, eps in enumerate(eps_values):
        for j, ms in enumerate(min_samples_values):
            aris = []
            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
                # --- Fit DBSCAN on train/test folds ---
                model_train = DBSCAN(eps=eps, min_samples=ms)
                model_test  = DBSCAN(eps=eps, min_samples=ms)
                labels_train = model_train.fit_predict(X[train_idx])
                labels_test  = model_test.fit_predict(X[test_idx])

                # Skip if too few clusters (e.g., all noise or single cluster)
                if len(set(labels_train)) <= 1 or len(set(labels_test)) <= 1:
                    continue

                # --- Train RF classifier on train clusters ---
                rf = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=p,
                    max_features=int(np.round(np.sqrt(p))),
                    random_state=fold_idx
                )
                rf.fit(X[train_idx], labels_train)

                # --- Predict on test fold and compute ARI ---
                labels_test_pred = rf.predict(X[test_idx])
                ari = adjusted_rand_score(labels_test, labels_test_pred)
                aris.append(ari)

            ari_results[i, j] = np.mean(aris) if len(aris) > 0 else np.nan
            print(f"eps={eps:.2f}, min_samples={ms:2d} | mean ARI={ari_results[i,j]:.3f}")

    # --- Plot results ---
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        ari_results, annot=True, fmt=".3f",
        xticklabels=min_samples_values, yticklabels=np.round(eps_values, 2),
        cmap="crest"
    )
    plt.xlabel("min_samples")
    plt.ylabel("eps")
    plt.title("DBSCAN Predictive ARI (5-fold CV)")
    plt.tight_layout()
    plt.show()

    # --- Identify best parameter combination ---
    best_idx = np.nanargmax(ari_results)
    best_eps_i, best_ms_j = np.unravel_index(best_idx, ari_results.shape)
    best_eps = eps_values[best_eps_i]
    best_ms  = min_samples_values[best_ms_j]
    best_ari = ari_results[best_eps_i, best_ms_j]

    print(f"\nBest Parameters: eps={best_eps:.3f}, min_samples={best_ms} (Mean ARI={best_ari:.3f})")

    return best_eps, best_ms, ari_results

def dbscan_silhouette(X, eps=0.35, min_samples=5, metric="euclidean"):
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit_predict(X)
    n_noise = np.sum(labels == -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # compute on clustered points only
    mask = labels != -1
    if n_clusters < 2 or mask.sum() == 0:
        return dict(score=np.nan, n_clusters=n_clusters, n_noise=n_noise, labels=labels)

    score = silhouette_score(X[mask], labels[mask], metric=metric)
    return dict(score=score, n_clusters=n_clusters, n_noise=n_noise, labels=labels)

def single_linkage_best_cut(X, metric="euclidean", n_grid=60, min_frac=0.05, max_frac=0.95):
    Z = linkage(X, method="single", metric=metric)
    d = Z[:, 2]  # merge heights

    # scan sensible range of heights (avoid extremes)
    lo, hi = np.quantile(d, min_frac), np.quantile(d, max_frac)
    ts = np.linspace(lo, hi, n_grid)

    best = (None, -1.0, None)  # (t, score, labels)
    scores = []

    for t in ts:
        labels = fcluster(Z, t=t, criterion="distance")
        k = len(np.unique(labels))
        if k < 2:      # silhouette requires ≥2 clusters
            scores.append(np.nan); continue
        try:
            s = silhouette_score(X, labels)
        except Exception:
            s = np.nan
        scores.append(s)
        if np.isfinite(s) and s > best[1]:
            best = (t, s, labels)

    return Z, ts, np.array(scores), best  # best=(t*, score*, labels*)


def dbscan_initial_tune(X):
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))

    #============================================
    # DBSCAN - DEFAULT
    #============================================
    model = DBSCAN() # do not specify leave DEFAULT
    labels = model.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()

    ax[0].scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=15)
    ax[0].set_title(f"DBSCAN DEFAULT FIT")
    ax[0].set_xticks([]); ax[0].set_yticks([])

    # legend
    info_text = f"|clusters| = {n_clusters}\n|noise points| = {n_noise}"
    text_only = [Line2D([], [], linestyle='none', marker='')]  # invisible handle
    ax[0].legend(text_only, [info_text], loc="best", frameon=False, handlelength=0, handletextpad=0)


    #============================================
    # DISTANCES TUNING PLOT
    #============================================
    min_samples = 5  # typically valid in 2D embedding

    neighbors = NearestNeighbors(n_neighbors=min_samples) # Compute distance to the k-th nearest neighbor for every point
    neighbors_fit = neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)
    distances = np.sort(distances[:, -1]) # Sort the distances (to find the “elbow”)

    ax[1].plot(distances)
    ax[1].set_title(f"k-distance Graph (min_samples={min_samples})")
    ax[1].set_xlabel("Points sorted by distance")
    ax[1].set_ylabel(f"Distance to {min_samples}-th Nearest Neighbor")
    ax[1].grid(True)

    eps_opt = 0.35  # Based on the elbow around 0.35
    ax[1].axhline(eps_opt, color='red', linestyle='--', label=f"Chosen eps = {eps_opt}")
    ax[1].legend()

    #============================================
    # DBSCAN - TUNED (ELBOW)
    #============================================
    model = DBSCAN(eps=eps_opt, min_samples=min_samples)
    labels = model.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()

    ax[2].scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=15)
    ax[2].set_title(f"DBSCAN (eps={eps_opt}, min_samples={min_samples})")
    ax[2].set_xticks([]); ax[2].set_yticks([])

    # legend
    info_text = f"|clusters| = {n_clusters}\n|noise points| = {n_noise}"
    text_only = [Line2D([], [], linestyle='none', marker='')]  # invisible handle
    ax[2].legend(text_only, [info_text], loc="best", frameon=False, handlelength=0, handletextpad=0)

    plt.tight_layout()
    plt.show()