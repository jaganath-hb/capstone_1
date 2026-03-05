from sklearn.cluster import KMeans
import numpy as np


def cluster_embeddings(embeddings: np.ndarray, n_clusters: int = 8):
    """Simple KMeans clustering placeholder. Replace with HDBSCAN/UMAP+HDBSCAN for better clusters."""
    if embeddings.shape[0] < n_clusters:
        n_clusters = max(1, embeddings.shape[0])
    km = KMeans(n_clusters=n_clusters, random_state=42)
    labels = km.fit_predict(embeddings)
    return labels
