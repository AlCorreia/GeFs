from collections import OrderedDict
from math import erf, floor
from numba import njit, int64, float64, optional, prange, deferred_type, types, boolean
from numba.experimental import jitclass
import numpy as np

from .utils import isin_nb

FEATURE_THRESHOLD = 1e-7


@njit
def random_split(x):
    """ Produces a random split of array x. """
    left_ids = np.random.choice(np.array([0, 1]), len(x)) > 0
    return x[left_ids], x[~left_ids]


@njit('int64[:](int64[:],int64)', fastmath=True, inline='always', boundscheck=False)
def bincount(data, n):
    counts = np.zeros(n, dtype=np.int64)
    for j in prange(data.size):
        counts[data[j]] += 1
    return counts


#############################
##### IMPURITY MEASURES #####
#############################

@njit("float64(int64[:])")
def gini(counts):
    """
        Computes the gini score in the distribution of classes.
        The higher the gini score, the 'purer' the distribution.
    """
    p = counts/np.sum(counts)
    return np.sum(p*(p-1))


@njit("float64(int64[:])")
def entropy(counts):
    """
        Computes the entropy in the distribution of classes.
        Returns -1*entropy as we want to maximize instead of minimize the score.
    """
    p = counts/np.sum(counts) + 1e-12
    return np.sum(p*np.log2(p))


@njit("float64(int64[:])")
def purity(counts):
    """
        Purity is a measure defined by the ratio between the number of counts
        of the majority class over the total number of instances.
        The larger the purity, the better the cluster.
    """
    if counts.sum() == 0:
        return 1
    return np.max(counts)/np.sum(counts)


@njit("float64(int64[:],int64[:],types.string)", fastmath=True)
def gain(left_counts, right_counts, imp_measure):
    """
        Computes the gain of a split.

        Parameters
        ----------
        left_counts, right_counts: numpy
            The counts in each side of the split.
        imp_measure: string
            The type of impurity of measure to use.
    """
    ratio = np.sum(left_counts)/np.sum(left_counts + right_counts)
    if imp_measure == 'gini':
        return ratio*gini(left_counts) + (1-ratio)*gini(right_counts)
    elif imp_measure == 'purity':
        return ratio*purity(left_counts) + (1-ratio)*purity(right_counts)
    elif imp_measure == 'entropy':
        return ratio*entropy(left_counts) + (1-ratio)*entropy(right_counts)
    else:
        return -np.Inf


@njit("f8(i8[:],i8[:],i8)", fastmath=True, inline='always')
def gini_gain(left_counts, right_counts, n):
    total_left = 0
    total_right = 0
    gini_left = 0
    gini_right = 0
    for i in range(n):
        gini_left -= left_counts[i]**2
        gini_right -= right_counts[i]**2
        total_left += left_counts[i]
        total_right += right_counts[i]

    gini_left = total_left - gini_left/total_left
    gini_right = total_right - gini_right/total_right

    return gini_left + gini_right


@jitclass([
    ('score', float64),
    ('var', int64),
    ('threshold', float64[:]),
    ('surr_var', int64[:]),
    ('surr_thr', float64[:]),
    ('surr_go_left', boolean[:]),
    ('surr_blind', boolean),
    ('pos', int64),
    ('left_ids', int64[:]),
    ('right_ids', int64[:]),
    ('left_counts', int64[:]),
    ('right_counts', int64[:]),
    ('type', types.string)
])
class Split:
    def __init__(self):
        """
            Abstract class for storing the relevant information of a split.
        """
        self.score = -np.Inf
        self.var = -1
        self.threshold = np.zeros(1, dtype=np.float64) -np.Inf
        self.surr_var = np.empty(0, dtype=np.int64)
        self.surr_go_left = np.empty(0, dtype=boolean)
        self.surr_thr = np.empty(0, dtype=np.float64)
        self.surr_blind = True
        self.left_ids = np.empty(0, dtype=np.int64)
        self.right_ids = np.empty(0, dtype=np.int64)
        self.left_counts = np.empty(0, dtype=np.int64)
        self.right_counts = np.empty(0, dtype=np.int64)
        self.pos = -1
        self.type = 'num'


@njit(parallel=False)
def find_best_split(node, tree):
    """
        Looks for the best split among for the data reaching node.

        Parameters
        ----------
        node: Node object (nodes.py)
        tree: Tree object (trees.py)

        Returns
        -------
        best_split: Split object containing the info of the best split.
    """
    if (node.idx.shape[0] < tree.min_samples_split) or (purity(node.counts) >= 0.99) or (node.depth >= tree.max_depth):
        return None
    best_score = -1e6
    best_split = None
    vars = np.random.choice(np.arange(tree.X.shape[1]), tree.max_features, replace=False)
    X = tree.X[node.idx, :]
    y = tree.y[node.idx]
    for i in range(tree.max_features):
        var = vars[i]
        x = X[:, var]
        nan_mask = np.isnan(x)
        counts = node.counts - bincount(y[nan_mask], tree.n_classes)
        if tree.ncat[var] > 1:
            split = categorical_split(x[~nan_mask], y[~nan_mask], node.idx[~nan_mask], var, counts, tree)
        else:
            split = numerical_split(x[~nan_mask], y[~nan_mask], node.idx[~nan_mask], counts, tree)
        if split.score > best_score:
            if np.sum(nan_mask) > 0:
                nan_left, nan_right = random_split(node.idx[nan_mask])
                split.left_ids = np.concatenate((split.left_ids, nan_left))
                split.right_ids = np.concatenate((split.right_ids, nan_right))
                nan_counts_left = bincount(tree.y[nan_left], tree.n_classes)
                nan_counts_right = bincount(tree.y[nan_right], tree.n_classes)
                split.left_counts += nan_counts_left
                split.right_counts += nan_counts_right
            best_score = split.score
            best_split = split
            best_split.var = var
    if best_split is not None:
        if tree.surrogate:
            remaining_vars = np.array([v for v in vars if v != best_split.var], dtype=np.int64)
            x = X[:, best_split.var]
            x = x[~np.isnan(x)]
            y = y[~np.isnan(x)]
            surr_mask = x > best_split.threshold[0]
            surr_y = np.zeros_like(y)
            surr_y[surr_mask] = 1
            counts = bincount(surr_y, 2)
            best_split.surr_blind = np.argmax(counts) == 0 # If the majority is in zero, go left
            min_score = np.mean(surr_y == np.ones_like(surr_y)*np.argmax(counts))
            surr_var, surr_thr, surr_go_left, n = surrogate_split(X[~np.isnan(x), :], surr_y, remaining_vars, tree, min_score)
            if n > 0:
                best_split.surr_var, best_split.surr_thr, best_split.surr_go_left = surr_var[:n], surr_thr[:n], surr_go_left[:n]
        else:
            counts = np.array([np.sum(best_split.left_counts), np.sum(best_split.right_counts)], dtype=np.int64)
            best_split.surr_blind = np.argmax(counts) == 0 # If the majority is in zero, go left
    return best_split


@njit(parallel=False)
def numerical_split(x, y, idx, total_counts, tree):
    """
        Looks for the best split for numerical variables.

        Parameters
        ----------
        x: numpy
            One dimensional array containing the realisations of a categorical
            variable we want to split on. Assumed without missing values.
        y: numpy
            The class variable corresponding to the observations in x.
        idx: numpy
            The indices of the examples in x and y w.r.t. to the entire dataset.
        total_counts: array of ints
            The counts of the target variable over all samples.
        tree: Tree object
            Only for assessing tree parameters, namely min_samples_leaf and
            number of classes.

        Returns
        -------
        best_split: Split object
    """
    best_split = Split()
    best_split.type = 'num'
    order = np.argsort(x)
    x_ord = x[order]
    y_ord = y[order]
    idx_ord = idx[order]
    start = tree.min_samples_leaf
    end = y_ord.size - tree.min_samples_leaf
    left_counts = bincount(y_ord[:start], tree.n_classes)
    right_counts = total_counts - left_counts
    for i in range(start, end):
        left_counts[y_ord[i]] += 1
        if (x_ord[i+1] <= x_ord[i] + FEATURE_THRESHOLD):
            continue
        right_counts = total_counts - left_counts
        score = gini_gain(left_counts, right_counts, tree.n_classes)
        if score > best_split.score:
            best_split.score = score
            best_split.threshold[0] = x_ord[i]
            best_split.left_ids = idx_ord[:i+1]
            best_split.right_ids = idx_ord[i+1:]
            best_split.left_counts = left_counts.copy()
            best_split.right_counts = right_counts.copy()
    return best_split


@njit(parallel=False)
def surrogate_split(X, y, vars, tree, min_score):
    """
        Looks for surrogate splits.

        Parameters
        ----------
        x: numpy
            Multi-dimensional array containing the realisations of all variables
            (categorical and numerical) other than the split variable.
            May contain missing values.
        y: numpy
            1 if the split varible is above the threshold and 0 otherwise.
        vars: array of ints
            The index (column) of the all but the split variable.
        tree: Tree object
            Only for assessing tree parameters, namely min_samples_leaf and
            number of classes.

        Returns
        -------
        ord_vars: array of ints
            Variable indices ordered according to impurity of their best splits.
            The first variable produces the closest split to the split variable.
        thrs: array of floats
            The correspoding split threshold for each variable in ord_vars.
        go_lefts: numpy array (boolean)
            All splits are evaluated as if x_i <= thr_i, go left. A surrogate split
            needs to match the best split, and it might happen that the best way
            to do so is to define a split of the form if x_j <= thr_j, go right.
            For simplicity, instead of changing the inequality at each if statement,
            we create a boolean array which is true if the surrogate split defines
            a `go left` rule.
        n: int
            Number of surrogate splits found.
    """
    scores = np.zeros(len(vars), dtype=np.int64)-1e6
    thrs = np.zeros(len(vars), dtype=np.float64)
    go_lefts = np.zeros(len(vars), dtype=boolean)
    for j in range(len(vars)):
        # Filter missing values
        x_var = X[:, vars[j]]
        nan_mask = ~np.isnan(x_var)
        x_var = x_var[nan_mask]
        y_var = y[nan_mask]
        # Order the variables
        order = np.argsort(x_var)
        x_ord = x_var[order]
        y_ord = y_var[order]

        start = 2
        end = y_ord.size - 2
        best_score = -np.Inf
        best_thr = 0
        go_left = True

        score_same = np.sum(y_ord[start:]) + np.sum(y_ord[:start]==0)
        for i in range(start, end):
            if y_ord[i] == 0:
                score_same += 1
            else:
                score_same -= 1
            if (x_ord[i+1] <= x_ord[i] + FEATURE_THRESHOLD):
                continue
            score_inv = y_ord.size - score_same
            score = max(score_inv, score_same)/y_ord.size
            if score > best_score:
                best_score = score
                best_thr = (x_ord[i]+x_ord[i+1])/2
                if score_same >= score_inv:
                    go_left = True
                else:
                    go_left = False
        scores[j] = best_score
        thrs[j] = best_thr
        go_lefts[j] = go_left
    order = np.argsort(-scores)
    n = len(scores[scores > min_score])
    ord_vars = vars[order]
    thrs = thrs[order]
    go_lefts = go_lefts[order]
    return ord_vars, thrs, go_lefts, n


@njit
def categorical_split(x, y, idx, var, total_counts, tree):
    """
        Looks for the best split for categorical variables. It differs from
        a numerical split in that it is not sufficient to divide values above
        and below are threshold because the variable is not necessarily ordinal.
        Because of that all possible subsets of the categories need to be
        considered for a split.

        If the target variable is binary, we can do better by ordering the
        categories in `x` according to the proportion falling in class 1.
        Then we split `x` as if it were an ordered predictor.
        - from Elements of Statistical Learning p.310

        Parameters
        ----------
        x: numpy
            One dimensional array containing the realisations of a categorical
            variable we want to split on. Assumed without missing values.
        y: numpy
            The class variable corresponding to the observations in x.
        idx: numpy
            The indices of the examples in x and y w.r.t. to the entire dataset.
        var: int
            The index (column) of the variable.
        total_counts: array of ints
            The counts of the target variable over all samples.
        tree: Tree object
            Only for assessing tree parameters, namely min_samples_leaf and
            number of classes.

        Returns
        -------
        best_split: Split object
    """
    best_split = Split()
    best_split.type = 'cat'
    # Create an array with counts per category
    counts = np.zeros((tree.ncat[var], tree.n_classes), dtype=np.int64)
    x_int = np.asarray(x, dtype=np.int64)
    for i in range(len(x)):
        counts[x_int[i], y[i]] += 1

    if tree.n_classes == 2:
        categories = np.argsort(counts[:, 1].ravel())
        n_splits = tree.ncat[var]
    else:
        categories = np.arange(tree.ncat[var])
        n_splits = int(2**(categories.shape[0]-1))

    for i in range(1, n_splits):
        if tree.n_classes == 2:
            split = np.array([categories[j] for j in range(categories.shape[0]) if j < i])
        else:
            split = np.array([categories[j] for j in range(categories.shape[0]) if (i & (1 << j))])
        left_size = counts[split].sum()
        right_size = y.size-left_size
        if (left_size < tree.min_samples_leaf) or (right_size < tree.min_samples_leaf):
            score = -np.Inf
        else:
            left_counts = counts[split].sum(axis=0)
            right_counts = total_counts - left_counts
            score = gini_gain(left_counts, right_counts, tree.n_classes)
        if score > best_split.score:
            left_mask = isin_nb(x, split)
            best_split.score = score
            best_split.threshold = split.astype(np.float64)
            best_split.left_ids = idx[left_mask]
            best_split.right_ids = idx[~left_mask]
            best_split.left_counts = left_counts
            best_split.right_counts = right_counts
    return best_split
