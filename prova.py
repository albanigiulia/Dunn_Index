import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans


matrix1= [[0,0,0,0], [0,0,0,0], [1,1,1,1], [1,1,1,1]]
kmeans1 = KMeans(n_clusters=2)
kmeans1.fit(matrix1)
print("Centri dei cluster:", kmeans1.cluster_centers_)
print("Etichette assegnate:", kmeans1.labels_) 
centroidi1 = kmeans1.cluster_centers_
etichette1 = kmeans1.labels_




def _calculate_dunn_index(data: np.ndarray,
                          labels: np.ndarray,
                          centroids: np.ndarray) -> float:

    cluster_distances = []

    for cluster_label in np.unique(labels):

        cluster_points = data[labels == cluster_label]

        if len(cluster_points) > 1:
            intra_cluster_distances = pairwise_distances(
                cluster_points, metric='euclidean', n_jobs=-1)

            cluster_distances.append(np.mean(intra_cluster_distances))

    inter_cluster_distances = pairwise_distances(
        centroids, metric='euclidean', n_jobs=-1)

    min_inter_cluster_distance = np.min(
        inter_cluster_distances[inter_cluster_distances > 0])

    max_intra_cluster_distance = np.max(cluster_distances)

    dunn_index = min_inter_cluster_distance / max_intra_cluster_distance

    return dunn_index

_calculate_dunn_index(matrix1, etichette1, centroidi1)