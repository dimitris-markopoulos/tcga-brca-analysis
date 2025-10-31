import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

def plot_all_clustering_methods(X_embed, df_plot, col_label, n_clusters=5, random_state=42):
    """
    Plot 1-row grid comparing KMeans, Spectral, GMM, Hierarchical (Ward),
    and true Subtype on the same UMAP embedding.
    """
    X_embed = np.asarray(X_embed)
    methods = {
        "K-Means": KMeans(n_clusters=n_clusters, n_init='auto', random_state=random_state),
        "Spectral": SpectralClustering(n_clusters=n_clusters, assign_labels='kmeans',
                                       random_state=random_state),
        "GMM": GaussianMixture(n_components=n_clusters, random_state=random_state),
        "Hierarchical (Ward)": AgglomerativeClustering(n_clusters=n_clusters, linkage='ward'),
    }

    fig, axes = plt.subplots(1, len(methods) + 1, figsize=(20, 4))
    axes = axes.flatten()

    # ---- Loop over clustering methods ----
    for i, (name, model) in enumerate(methods.items()):
        labels = model.fit_predict(X_embed)
        centers = model.cluster_centers_ if name == "K-Means" else None

        sns.scatterplot(
            x=X_embed[:, 0], y=X_embed[:, 1],
            hue=labels, palette='tab10', s=22, edgecolor='none',
            ax=axes[i], legend=False
        )

        if centers is not None:
            axes[i].scatter(
                centers[:, 0], centers[:, 1],
                s=120, c='black', marker='X',
                edgecolor='white', linewidth=1.2
            )

        axes[i].set_title(f"{name} (K={n_clusters})", fontsize=12, pad=8)
        axes[i].set_xticks([]); axes[i].set_yticks([])
        axes[i].set_xlabel(''); axes[i].set_ylabel('')

    # ---- Final panel: True Subtype ----
    sns.scatterplot(
        data=df_plot, x='UMAP-1', y='UMAP-2',
        hue=col_label, palette='tab10', s=22,
        ax=axes[-1], edgecolor='none', legend=True
    )
    axes[-1].set_title("True Subtype", fontsize=12, pad=8)
    axes[-1].set_xticks([]); axes[-1].set_yticks([])
    axes[-1].set_xlabel(''); axes[-1].set_ylabel('')
    axes[-1].legend(loc='upper left', fontsize=6, markerscale=1, frameon=True)

    # Shared axis labels
    fig.text(0.5, 0.02, 'UMAP-1', ha='center', fontsize=13)
    fig.text(0.02, 0.5, 'UMAP-2', va='center', rotation='vertical', fontsize=13)

    plt.tight_layout(rect=[0.04, 0.04, 1, 1])
    plt.show()
