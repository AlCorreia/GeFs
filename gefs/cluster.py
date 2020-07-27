from numba import njit, boolean, int64, float64, optional
from numba.experimental import jitclass
import numpy as np


@njit
def cluster(data, n_clusters, ncat, maxit=100):
    """
        Runs KMeans on data and returns the labels of each sample.

        Parameters
        ----------
        data: numpy array
            Rows are instances and columns variables.
        n_clusters: int
            Number of clusters
        ncat:
            The number of categories of each variable.
        maxit: int
            The maximum number of iterations of the KMeans algorithm.
    """
    kmeans = Kmeans(n_clusters, ncat, None, maxit).fit(data)
    res = kmeans.labels.ravel()
    assert res.shape[0] == data.shape[0]
    return res


@njit
def nan_to_num(data):
    """ Replaces NaNs with zeros. """
    shape = data.shape
    data = data.ravel()
    data[np.isnan(data)] = 0
    return data.reshape(shape)


@njit
def get_distance(a, b, ncat):
    distance = 0.
    for i in range(len(a)):
        if np.isnan(a[i]) or np.isnan(b[i]):
            distance += 1
        elif ncat[i] > 1:
            distance += min(abs(a[i]-b[i]), 1)
        else:
            distance += (a[i]-b[i])**2
    return distance


@njit
def assign(point, centroids, ncat):
    minDist = np.Inf
    for i in range(centroids.shape[0]):
        dist = get_distance(point, centroids[i, :], ncat)
        if dist < minDist:
            minDist = dist
            label = i
    return label, minDist


@njit
def assign_clusters(data, centroids, ncat):
    error = 0
    labels = np.zeros(data.shape[0], dtype=np.int64)
    for i in range(data.shape[0]):
        labels[i], dist = assign(data[i, :], centroids, ncat)
        error += dist
    return labels, error


@njit
def get_error(centroid, data, ncat):
    error = 0
    for j in range(data.shape[0]):
        error += get_distance(centroid, data[j, :], ncat)
    return error


@jitclass([
    ('k', int64),
    ('ncat', int64[:]),
    ('data', float64[:,:]),
    ('nvars', int64),
    ('centroids', float64[:,:]),
    ('labels', int64[:]),
    ('error', float64),
    ('error1', float64),
    ('error2', float64),
    ('it', int64),
    ('maxit', int64),
    ('thr', optional(float64)),
])
class Kmeans:
    def __init__(self, k, ncat, thr=None, maxit=100):
        """
            Minimal KMeans implementation in numba.

            Parameters
            ----------
            k: int
                Number of clusters.
            data: numpy array
                Data with instances as rows and variables as columns.
            ncat: numpy array
                Number of categories of each variable.
            thr:
                Threshold at which the algorithm is considered to have converged.
            maxit: int
                Maximum number of iterations.
        """
        self.k = k
        self.nvars = 0
        self.ncat = ncat
        self.data = np.empty((0, 0), dtype=np.float64)
        self.centroids = np.empty((0, 0), dtype=np.float64)
        self.labels = np.empty((0), dtype=np.int64)
        self.thr = thr
        self.error = 0
        self.error1 = 0
        self.error2 = 0
        self.it = 0
        self.maxit = maxit

    def init_centroids(self):
        seed_idx = np.random.choice(self.data.shape[0], self.k, replace=False)
        self.centroids = self.data[seed_idx, :]
        self.error1 = 0

    def update_centroids(self):
        error = 0
        for i in range(self.k):
            cluster_data = self.data[self.labels == i, :]
            if cluster_data.shape[0] > 0:
                for j in range(self.nvars):
                    if np.all(np.isnan(cluster_data[:, j])):
                        self.centroids[i, j] = 0.
                    else:
                        self.centroids[i, j] = np.nanmean(cluster_data[:, j])
            error += get_error(self.centroids[i, :], cluster_data, self.ncat)
        self.error1 = error

    def assign_clusters(self):
        self.labels, self.error2 = assign_clusters(self.data, self.centroids, self.ncat)

    def fit(self, data):
        assert data.shape[0] >= self.k, "Too few data points."
        self.data = data
        self.nvars = data.shape[1]
        if self.thr is None:
            self.thr = data.size * 1e-6 # weird heuristic
        self.init_centroids()
        self.assign_clusters()
        self.it = 0
        while (abs(self.error1-self.error2) > self.thr) and (self.it < self.maxit):
            self.it += 1
            self.update_centroids()
            self.assign_clusters()
        return self


def makeRandomPoint(n, lower, upper, missing_perc=0.0):
    points = np.random.normal(loc=upper, size=[lower, n])
    missing_mask = np.full(points.size, False)
    missing_mask[:int(missing_perc * points.size)] = True
    np.random.shuffle(missing_mask)
    missing_mask = missing_mask.astype(bool)
    points.ravel()[missing_mask] = np.nan
    return points
