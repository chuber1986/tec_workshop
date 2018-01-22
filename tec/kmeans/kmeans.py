import numpy as np

from tec.utils.gen_data import k_means_ds
from tec.utils.plotting import KMeansPlot


def distance(x, mean):
    """
    Computes the distance between the data point and a given cluster mean.
    This implementation computes the Euclidean distance.

    :param x: samples of shape (sample, features)
    :param mean: cluster mean of shape (features)
    :return: Euclidean distances of shape (sample, )
    """
    return np.sqrt(np.sum(np.square(x - mean), axis=1))


def initialize(x, k):
    """
    K-means initialization. Computes k random point in the feature space which represent the cluster means.
    This implementation uses the coordinates of k random but distinct points from the input samples as cluster means.
    This guaranties that each initial cluster mean is in a reasonable region in the feature space.
    Further, it guaranties that each cluster contains at least one input sample point.

    :param x: samples of shape (sample, features)
    :param k: number of expected clusters
    :return: a random set of k cluster means of shape (k, features)
    """

    # TODO Generate k random cluster means (not just zeros)
    ...

    return np.zeros((k, x.shape[1]))


def update_means(clusters):
    """
    Computes for each cluster its new mean value.

    :param clusters: current clusters of shape (cluster, sample, features)
    :return: set of k cluster means of shape (k, features)
    """
    new = []
    # TODO Compute the means values of the current clusters
    ...

    return np.asarray(new)


def update_clusters(x, means):
    """
    Searches for each input sample the nearest cluster mean and assigns the point to the corresponding cluster.

    :param x: samples of shape (sample, features)
    :param means: cluster means of shape (k, features)
    :return: new clusters of shape (cluster, sample, features)
    """
    # Compute distances
    # TODO Find the minimum distance of each sample to each cluster mean
    ...

    # Find the nearest cluster for each sample
    ...

    # Assign the each sample to its cluster
    # TODO Assign the each sample to its cluster
    clusters = []
    ...

    return clusters


def cluster(x, k=2, iters=10):
    """
    Takes samples x of shape (sample, features) and performs a k-means clustering on these samples.
    The input will be divided in k clusters.

    :return: Clustered samples of shape (cluster, sample, features)
    """
    plt = KMeansPlot(t_mean_update=0.5, t_cluster_update=0.5)

    # Initialize
    # TODO Initilize clusters
    ...

    # Main loop
    # TODO Implement main loop
    for i in range(iters):
        # Get new cluster mean values
        ...

        # Reassign samples to clusters
        ...

    # plt.save('../../model', 'kmeans')
    return clusters, means


def main():
    max_iterations = 10
    n_clusters = 3

    # Load dataset (0-3)
    x = k_means_ds(0)

    # Start clustering
    cluster(x, n_clusters, max_iterations)


if __name__ == '__main__':
    main()
