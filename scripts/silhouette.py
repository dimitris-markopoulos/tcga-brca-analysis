import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture

def silhouette_sweep(X, K_values=range(2, 11), linkages = ['ward'], random_state=42):
    results = []
    for K in K_values:
        # KMeans
        km = KMeans(n_clusters=K, n_init='auto', random_state=random_state)
        km_labels = km.fit_predict(X)
        sil = silhouette_score(X, km_labels)
        results.append({'method':'KMeans','param':K,'silhouette':sil})

        # Spectral
        sc = SpectralClustering(n_clusters=K, random_state=random_state, assign_labels='kmeans', affinity='nearest_neighbors')
        sc_labels = sc.fit_predict(X)
        sil = silhouette_score(X, sc_labels)
        results.append({'method':'Spectral','param':K,'silhouette':sil})

        # GMM
        gm = GaussianMixture(n_components=K, covariance_type='full', random_state=random_state)
        gm_labels = gm.fit_predict(X)
        sil = silhouette_score(X, gm_labels)
        results.append({'method':'GMM','param':K,'silhouette':sil})

        # Hierarchical
        for linkage in linkages:
            hc = AgglomerativeClustering(n_clusters=K, linkage=linkage)
            hc_labels = hc.fit_predict(X)
            sil = silhouette_score(X, hc_labels)
            results.append({'method':f'Hierarchical | {linkage.capitalize()}','param':K,'silhouette':sil})

    return pd.DataFrame(results)

def dbscan_silhouette_grid(X, eps_values, min_samples_values):
    records = []
    for eps in eps_values:
        for ms in min_samples_values:
            model = DBSCAN(eps=eps, min_samples=ms)
            labels = model.fit_predict(X)
            
            if len(set(labels)) <= 1:
                continue  # skip degenerate cases
            score = silhouette_score(X, labels)
            records.append({"eps": eps, "min_samples": ms, "silhouette": score})
    return pd.DataFrame(records)