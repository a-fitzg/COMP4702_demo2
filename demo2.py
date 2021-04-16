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

    def __init__(self, points, n=2, distance_metric='euclidean', max_iterations=100):
        """
        A k-means clustering algorithm
        :param points: List of points
        :param distance_metric: Distance metric to use (optional).
            Options:
                euclidean (default),
                manhattan,
                chebyshev.
        :param n: Number of clusters to create
        :param max_iterations: Max number of iterations the algorithm will do (default 100)
        """
        self.n = n
        self.max_iterations = max_iterations

        # Make empty list of list of points, n elements long
        self.clusters = [[] for i in range(self.n)]

        # Make empty list of centroids
        self.centroids = []

        # Our output, empty for now
        self.complete_clustering = list()

        self.points = points
        self.num_samples, self.num_features = points.shape
        self.distance_metric = distance_metric

        # Run the clustering algorithm
        self.km_cluster()

    def get_clustered_data(self):
        """
        Returns the clustered data
        :return: Clustered data (as pandas DataFrame)
        """
        return self.complete_clustering

    def km_cluster(self):
        """
        Implements the k-means clustering algorithm
        """
        # Choose random centroids, based on existing points in the dataset
        # i.e. choose n random points in the dataset, to be our initial centroids
        rands = np.random.choice(self.num_samples, self.n, replace=False)
        self.centroids = [self.points[i] for i in rands]

        for i in range(self.max_iterations):

            # Get clusters (indices), based on centroids
            self.clusters = self._create_clusters(self.centroids)

            # Get new and old positions of centroids, to check if they have moved
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            # Converged yet?
            if self._converged(centroids_old, self.centroids):
                break

        # Get cluster labels, then save a DataFrame with the clusters, and labels for each point in each cluster
        self.complete_clustering = pd.DataFrame(self.points, index=self._get_cluster_labels(self.clusters).ravel())

    def _get_cluster_labels(self, clusters):
        """
        Make labels for each of the points in each cluster
        :param clusters: Groups of points clustered together
        :return: list of labelled points
        """
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
        clusters = [[] for k in range(self.n)]
        # Iterate through all points
        for i, point in enumerate(self.points):
            # Find closest centroid to this point
            centroid_index = self._closest_centroid(point, centroids)
            # Assign an index to this point, based on whichever centroid it is closest to
            # Then, append this point to the cluster associated with that centroid
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
        # Empty list (size n) of lists (points - based on number of features)
        centroids = np.zeros((self.n, self.num_features))
        for i, cluster in enumerate(clusters):
            # Get the
            cluster_mean = np.mean(self.points[cluster], axis=0)
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

    km = KMeans(raw_data, classes, distance_metric='euclidean', max_iterations=200)
    pred = km.get_clustered_data()
    predicted_centroids = km.get_centroids_public()

    plot_pred = plt.figure(1)

    pred_plots = list()
    for plot in range(classes):
        pred_plots.append(plt.scatter(pred[0].loc[plot].to_numpy(), pred[1].loc[plot].to_numpy()))
    plot_centroids = plt.scatter(predicted_centroids[:, 0], predicted_centroids[:, 1], marker='X', s=150, c='k')

    plt.show()

    x = 123
