import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
from scipy.spatial import distance


def get_distance(x1, x2, mode='euclidean'):
    """
    Returns distance between two vectors x1 and x2
    :param x1: Vector 1
    :param x2: Vector 2
    :param mode: Distance metric (optional). Default: Euclidean
    :return: Euclidean distance between x1 and x2
    """
    if mode == 'euclidean':
        return np.sqrt(np.sum((x1 - x2) ** 2))
    elif mode == 'manhattan':
        return distance.cityblock(x1.tolist(), x2.tolist())
    elif mode == 'chebyshev':
        return distance.chebyshev(x1.tolist(), x2.tolist())
    else:
        return get_distance(x1, x2, mode='euclidean')


class KMeans:

    def __init__(self, n=2, max_iterations=100):
        self.n = n
        self.max_iterations = max_iterations

        # Make empty list of points, n elements long
        self.clusters = [[] for i in range(self.n)]
        self.centroids = []

    def predict(self, x, distance_metric='euclidean'):
        """
        Implements the k-means clustering algorithm
        :param x: Data points to cluster
        :param distance_metric: Distance metric to use for distance calculations (optional)
            Default: Euclidean
            Options:
                euclidean
                manhattan
                chebyshev
        :return: Data points, clustered
        """
        self.x = x
        self.num_samples, self.num_features = x.shape
        self.distance_metric = distance_metric

        # Choose random centroids
        rands = np.random.choice(self.num_samples, self.n, replace=False)
        self.centroids = [self.x[i] for i in rands]

        for i in range(self.max_iterations):

            self.clusters = self._create_clusters(self.centroids)

            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            # Converged yet?
            if self._converged(centroids_old, self.centroids):
                break

        # return cluster labels
        return pd.DataFrame(self.x, index=self._get_cluster_labels(self.clusters).ravel())
        #return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.num_samples, dtype=np.int8)
        for i, cluster in enumerate(clusters):
            for j in cluster:
                labels[j] = i
        return labels

    def _create_clusters(self, centroids):
        """
        Assign classes based on centroids
        :param centroids: Centroids of the data set
        :return: clusters
        """
        clusters = [[] for i in range(self.n)]
        for i, point in enumerate(self.x):
            centroid_index = self._closest_centroid(point, centroids)
            clusters[centroid_index].append(i)
        return clusters

    def _closest_centroid(self, sample, centroids):
        """
        Find index of closest centroid to a sample
        :param sample: A sample to find closest cetroid to
        :param centroids: A list of centroids
        :return: Index of closest centroid
        """
        # Find the distance from this point to all of the centroids
        distances = [get_distance(sample, point, mode=self.distance_metric) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index

    def _get_centroids(self, clusters):
        """
        Get the centroids of a list of clusters
        :param clusters: List of clusters
        :return: List of centroids for each cluster
        """
        centroids = np.zeros((self.n, self.num_features))
        for i, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.x[cluster], axis=0)
            centroids[i] = cluster_mean
        return centroids

    def _converged(self, centroids_old, centroids_new):
        """
        Check if the current iteration has converged
        This is done by checking if any of the centroids have moved:
            If centroids have moved since last iteration, then we haven't converged yet
            If centroids have not moved since last iteration, then we have converged
        :param centroids_old: List of centroids from previous iteration
        :param centroids_new: List of centroids from current iteration
        :return: True if converged, False if not
        """
        distances = [get_distance(centroids_old[i], centroids_new[i], mode=self.distance_metric)
                     for i in range(self.n)]
        return sum(distances) == 0

    def get_centroids_public(self):
        """
        Publically accesible - get centroids of the clustering
        :return: List of centroids
        """
        return self.centroids


if __name__ == "__main__":
    mat = scipy.io.loadmat('heightWeight.mat')
    data = pd.DataFrame(mat['heightWeightData'][:, 1:3], index=mat['heightWeightData'][:, 0].ravel())

    raw_data = mat['heightWeightData'][:, 1:3]

    classes = 2

    # plot for class 1
    #plot_original = plt.figure(1)
    #plot1_original = plt.scatter(data[0].loc[1].to_numpy(), data[1].loc[1].to_numpy())
    #plot2_original = plt.scatter(data[0].loc[2].to_numpy(), data[1].loc[2].to_numpy())

    km = KMeans(n=classes, max_iterations=200)
    pred = km.predict(raw_data, distance_metric='chebyshev')
    centroids = km.get_centroids_public()

    plot_pred = plt.figure(1)

    pred_plots = list()
    for i in range(classes):
        pred_plots.append(plt.scatter(pred[0].loc[i].to_numpy(), pred[1].loc[i].to_numpy()))
    plot_centroids = plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=150, c='k')

    plt.show()

    x = 123
