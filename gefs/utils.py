from math import erf
import numba as nb
from numba import njit
import numpy as np
import scipy.stats as stats
from .statsutils import chi_test, kruskal, kendalltau


@njit
def resample_strat(x, y, n_classes):
    """
    Resampling (bootstrapping) of x stratified according to y.

    Parameters
    ----------
    x: numpy array
        Typically a matrix nxm with n samples and m variables.
    y: numpy array (1D)
        The class variable we are stratifying against.
    n_classes: int
        The number of classes in y.

    Returns
    -------
    x, y: numpy arrays
        In-bag samples.
    """
    idx = np.arange(x.shape[0], dtype=np.int64)
    counts = bincount(y, n_classes)
    selected_idx = np.empty(0, dtype=np.int64)
    for i in range(n_classes):
        s = np.random.choice(idx[y==i], counts[i], replace=True)
        selected_idx = np.concatenate((selected_idx, s))
    return x[selected_idx, :], y[selected_idx], selected_idx


# numba implementation of numpy.isin
# https://stackoverflow.com/a/53083946/8107620
@njit
def in1d_vec_nb(matrix, index_to_remove):
    """
    Both matrix and index_to_remove have to be numpy arrays
    if index_to_remove is a list with different dtypes this
    function will fail.
    """
    out = np.empty(matrix.shape[0], dtype=nb.boolean)
    index_to_remove_set = set(index_to_remove)

    for i in nb.prange(matrix.shape[0]):
        if matrix[i] in index_to_remove_set:
            out[i] = True
        else:
            out[i] = False
    return out


@njit
def in1d_scal_nb(matrix, index_to_remove):
    """
    Both matrix and index_to_remove have to be numpy arrays
    if index_to_remove is a list with different dtypes this
    function will fail
    """
    out = np.empty(matrix.shape[0], dtype=nb.boolean)
    for i in nb.prange(matrix.shape[0]):
        if (matrix[i] == index_to_remove):
            out[i] = True
        else:
            out[i] = False
        return out


@njit
def isin_nb(matrix_in, index_to_remove):
    """
    Both matrix_in and index_to_remove have to be numpy arrays
    even if index_to_remove is actually a single number.
    """
    shape = matrix_in.shape
    if index_to_remove.shape == ():
        res = in1d_scal_nb(np.ravel(matrix_in), index_to_remove.take(0))
    else:
        res = in1d_vec_nb(np.ravel(matrix_in), index_to_remove)

    return res.reshape(shape)


@njit(fastmath=True)
def bincount(data, n):
    counts = np.zeros(n, dtype=np.int64)
    for j in range(n):
        counts[j] = np.sum(data==j)
    return counts


@njit
def Phi(x, mean, std):
    """ Cumulative distribution for normal distribution defined by mean and std. """
    return .5*(1.0 + erf((x-mean) / (std*np.sqrt(2.0))))


@njit
def phi(x, mean, std):
    """ Cumulative distribution for normal distribution defined by mean and std. """
    denom = np.sqrt(2*np.pi)*std
    num = np.exp(-.5*((x - mean) / std)**2)
    return num/denom


@njit
def logtrunc_phi(x, loc, scale, a, b):
    """
        Computes the log-density at `x` of a normal distribution with mean `loc` and
        variance `scale` truncated at `a` and `b`.
    """
    res = np.ones(x.shape[0], dtype=np.float64)
    denom = (Phi(b, loc, scale) - Phi(a, loc, scale))
    for i in range(x.shape[0]):
        if (x[i] < a) or (x[i] > b):
            res[i] = -np.Inf
        elif not np.isnan(x[i]):
            res[i] = np.log(phi(x[i], loc, scale)/denom)
    return res


@njit(fastmath=True)
def isin(a, b):
    """ Returns True if a in b. """
    for bi in b:
        if (bi == a):
            return True
    return False


@njit(fastmath=True)
def isin_arr(arr, b):
    res = np.empty(arr.shape[0], dtype=nb.boolean)
    for i in nb.prange(arr.shape[0]):
        res[i] = isin(arr[i], b)
    return res


@njit
def lse(a):
    """ Computes logsumexp in a 1D array. """
    result = 0.0
    largest_in_a = a[0]
    for i in range(a.shape[0]):
        if (not np.isnan(a[i])) and ((a[i] > largest_in_a) or np.isnan(largest_in_a)):
            largest_in_a = a[i]
    if largest_in_a == -np.inf:
        return a[0]
    for i in range(a.shape[0]):
        if not np.isnan(a[i]):
            result += np.exp(a[i] - largest_in_a)
    return np.log(result) + largest_in_a


@njit
def logsumexp2(a, axis):
    """ Computes logsumexp in a 2D array. """
    assert a.ndim == 2, 'Wrong logsumexp method.'
    if axis == 0:
        res = np.zeros(a.shape[1])
        for i in range(a.shape[1]):
            res[i] = lse(a[:, i].ravel())
    elif axis == 1:
        res = np.zeros(a.shape[0])
        for i in range(a.shape[0]):
            res[i] = lse(a[i, :].ravel())
    return res


@njit
def logsumexp3(a, axis):
    """ Computes logsumexp in a 3D array. """
    assert a.ndim == 3, 'Wrong logsumexp method.'
    if axis == 0:
        res = np.zeros((a.shape[1], a.shape[2]))
        for i in range(a.shape[1]):
            for j in range(a.shape[2]):
                res[i, j] = lse(a[:, i, j].ravel())
    elif axis == 1:
        res = np.zeros((a.shape[0], a.shape[2]))
        for i in range(a.shape[0]):
            for j in range(a.shape[2]):
                res[i, j] = lse(a[i, :, j].ravel())
    elif axis == 2:
        res = np.zeros((a.shape[0], a.shape[1]))
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                res[i, j] = lse(a[i, j, :].ravel())
    return res


@njit
def depfunc(i, deplist):
    """ Auxiliary function to assign clusters. See get_indep_clusters. """
    if deplist[i] == i:
        return i
    else:
        return deplist[i]


@njit
def get_indep_clusters(data, scope, ncat, thr):
    """
        Cluster the variables in data into independent clusters.

        Parameters
        ----------
        data: numpy n x m
            Numpy array comprising n realisations (instances) of m variables.
        scope: list
            The column indices of the variables to consider.
        ncat: numpy m
            The number of categories of each variable. One if its continuous.
        thr: float
            The threshold (p-value) below which we reject the hypothesis of
            independence. In that case, they are considered dependent and
            assigned to the same cluster.

        Returns
        -------
        clu: numpy m
            The cluster assigned to each variable.
            clu[m] is the cluster to which the mth variable is assigned.

    """
    n = len(scope)  # number of variables in the scope
    # Dependence list assuming all variables are independent.
    deplist = [i for i in range(n)]

    # Loop through all variables computing pairwise independence tests
    for i in range(0, (n-1)):
        for j in range(i+1, n):
            fatheri = depfunc(i, deplist)
            deplist[i] = fatheri
            fatherj = depfunc(j, deplist)
            deplist[j] = fatherj
            if fatheri != fatherj:
                v = 1
                unii = len(np.unique(data[~np.isnan(data[:, scope[i]]), scope[i]]))
                unij = len(np.unique(data[~np.isnan(data[:, scope[j]]), scope[j]]))
                mask = (~np.isnan(data[:, scope[j]]) * ~np.isnan(data[:, scope[i]]))
                m = np.sum(mask)
                if unii > 1 and unij > 1:
                    vari, varj = data[mask, scope[i]], data[mask , scope[j]]
                    if m > 4 and ncat[scope[i]] == 1 and ncat[scope[j]] == 1:
                        # both continuous
                        _, v = kendalltau(vari, varj)
                    elif m > 4*unij and ncat[scope[i]] == 1 and ncat[scope[j]] > 1:
                        # i continuous, j discrete
                        _, v = kruskal(vari, varj)
                    elif m > 4*unii and ncat[scope[i]] > 1 and ncat[scope[j]] == 0:
                        # i discrete, j continuous
                        _, v = kruskal(vari, varj)
                    elif m > unii*unij*2 and ncat[scope[i]] > 1 and ncat[scope[j]] > 1:
                        ## both discrete
                        v = chi_test(vari, varj)
                    if (v < thr) or np.isnan(v):  # not independent -> same cluster
                        deplist[fatherj] = fatheri

    clu = np.zeros(n)
    for i in range(n):
        clu[i] = depfunc(i, deplist)
    unique_clu = np.unique(clu)
    for i in range(n):
        clu[i] = np.min(np.where(clu[i] == unique_clu)[0])
    return clu


def scope2string(scope):
    """
        Auxiliary function that converts the scope (list of ints) into a string
        for printing.
    """
    if len(scope) <= 5:
        return scope
    res = ''
    first = scope[0]
    last = None
    for i in range(1, len(scope)-1):
        if scope[i-1] != scope[i]-1:
            first = scope[i]
        if scope[i]+1 != scope[i+1]:
            last = scope[i]
        if first is not None and last is not None:
            res += str(first) + ':' + str(last) + ' - '
            first = None
            last = None
    res += str(first) + ':' + str(scope[-1])
    return res


def get_counts(node_data, n_classes):
    """
        Returns the class counts in the data.

        Parameters
        ----------
        node_data: numpy array
            The data reaching the node.
        n_classes: int
            Number of classes in the data. This is needed as not
            every class is necessarily observed in node_data.

        Returns
        -------
        counts: numpy array (same dimension as all_classes)
            The class counts in node_data.
    """
    return bincount(node_data, n_classes)


class Dist:
    """
        Class that defines a (truncated) Gaussian density function.

        Attributes
        ----------
        scope: list of ints
            The variables over which the Gaussian is defined.
        lower: float
            The minimum value the variable might assume.
            Currently, only applicable for univariate Gaussians.
        upper: float
            The maximum value the variable might assume.
            Currently, only applicable for univariate Gaussians.
        n: int
            The number of data points used to fit the Gaussian.
        mean: float
            The empirical mean.
        cov: float
            The empirical variance, covariance
    """
    def __init__(self, scope, data=None, n=None, lower=-np.Inf, upper=np.Inf):
        if not isinstance(scope, list):
            scope = [scope]
        self.scope = scope
        self.lower = lower
        self.upper = upper
        if data is not None:
            self.n = data.shape[0]
            self.fit(data)
        else:
            self.n = n
            self.mean = None
            self.cov = None

    def fit(self, data):
        """
            Fits mean and convariance using data. Also normalizes the upper and
            lower thresholds to define where to truncate the Gaussian.

            Parameters
            ----------
            data: numpy n x m
                Numpy array comprising n realisations (instances) of m variables.
        """
        self.n = data.shape[0]
        assert self.n > 0, "Empty data"
        self.mean = np.nanmean(data[:, self.scope], axis=0)
        # assert ~np.isnan(self.mean), "Error computing the mean."
        if data[:, self.scope].shape[0] > 1:  # Avoid runtimewarning
            self.cov = np.cov(data[:, self.scope], rowvar=False)
            self.cov = np.where(self.cov > .1, self.cov, .1)
        else:
            self.cov = np.array(.1)  # Probably not the best solution here
        self.std = np.sqrt(self.cov)

        # Compute the tresholds to truncate the Gaussian.
        # The Gaussian has support [a, b]
        self.a = (self.lower - self.mean) / self.std
        self.b = (self.upper - self.mean) / self.std
        self.params = {'loc': self.mean, 'scale':self.std, 'a': self.a, 'b': self.b}
        return self

    def logpdf(self, data):
        """
            Computes the log-density at data.

            Parameters
            ----------
            data: numpy n x m
                Numpy array comprising n realisations (instances) of m variables.
        """
        complete_ind = np.where(~np.isnan(data[:, self.scope]).any(axis=1))[0]
        # Initialize log-density to zero (default value for non-observed variables)
        logprs = np.zeros(data.shape[0])
        logprs[complete_ind] = stats.truncnorm.logpdf(data[complete_ind, self.scope], **self.params).reshape(-1)
        return logprs.reshape(data.shape[0], 1)
