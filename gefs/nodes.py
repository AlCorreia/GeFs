from collections import OrderedDict
from numba import njit, boolean, int64, float64, deferred_type, optional, types, prange
from numba.experimental import jitclass
import numba as nb
import numpy as np


from .signed import (signed, signed_max, signed_min, signed_max_vec, signed_min_vec,
                    signed_prod, signed_sum, signed_sum_vec, signed_econtaminate, signed_join)
from .utils import bincount, logtrunc_phi, isin, isin_arr, lse, logsumexp2, logsumexp3


node_type = deferred_type()

spec = OrderedDict()
spec['id'] = int64
spec['parent'] = optional(node_type)
spec['left_child'] = optional(node_type)  # first child
spec['right_child'] = optional(node_type) # last child
spec['sibling'] = optional(node_type)  # next sibling

spec['scope'] = int64[:]
spec['type'] = types.unicode_type
spec['n'] = float64
spec['w'] = optional(float64[:])
spec['logw'] = optional(float64[:])
spec['comparison'] = int64
spec['value'] = float64[:]
spec['mean'] = float64
spec['std'] = float64
spec['a'] = float64
spec['b'] = float64
spec['logcounts'] = optional(float64[:])
spec['p'] = optional(float64[:])
spec['logp'] = optional(float64[:])
spec['upper'] = float64[:]
spec['lower'] = float64[:]


@jitclass(spec)
class Node:
    def __init__(self, parent, scope, type, n):
        self.id = np.random.randint(0, 10000000) # Random identification number
        self.parent = parent
        # initialize parent and left right children as None
        self.left_child = None
        self.right_child = None
        self.sibling = None
        self.scope = scope
        self.type = type
        self.n = n
        if parent is not None:
            parent.add_child(self)
        # Sum params
        self.w = None
        self.logw = None
        # Leaf params
        self.value = np.zeros(1, dtype=np.float64)
        self.comparison = -1
        # Gaussian leaf params
        self.mean = 0.
        self.std = 1.
        self.a = -np.Inf
        self.b = np.Inf
        self.p = None
        self.logp = None
        self.logcounts = None
        self.upper = np.ones(len(scope))*(np.Inf)
        self.lower = np.ones(len(scope))*(-np.Inf)

    @property
    def children(self):
        """ A list with all children. """
        children = []
        child = self.left_child
        while child is not None:
            children.append(child)
            child = child.sibling
        return children

    @property
    def nchildren(self):
        return len(self.children)

    def add_sibling(self, sibling):
        self.sibling = sibling

    def add_child(self, child):
        # if parent has no children
        if self.left_child is None:
            # this node is it first child
            self.left_child = child
            self.right_child = child
        else:
            # the last (now right) child will have this node as sibling
            self.right_child.add_sibling(child)
            self.right_child = child
        if self.type == 'S':
            self.reweight()

    def reweight(self):
        children_n = np.array([c.n for c in self.children])
        n = np.sum(children_n)
        if n > 0:
            self.n = n
            self.w = np.divide(children_n.ravel(), self.n)
            self.logw = np.log(self.w.ravel())


node_type.define(Node.class_type.instance_type)


###########################
### INTERFACE FUNCTIONS ###
###########################

@njit
def ProdNode(parent, scope, n):
    return Node(parent, scope, 'P', n)

@njit
def SumNode(parent, scope, n):
    return Node(parent, scope, 'S', n)

@njit
def Leaf(parent, scope, n, value, comparison):
    node = Node(parent, scope, 'L', n)
    fit_indicator(node, value, comparison)
    return node

@njit
def GaussianLeaf(parent, scope, n):
    return Node(parent, scope, 'G', n)

@njit
def MultinomialLeaf(parent, scope, n):
    return Node(parent, scope, 'M', n)

@njit
def UniformLeaf(parent, scope, n, value):
    node = Node(parent, scope, 'U', n)
    node.value = value
    return node


#################################
###    AUXILIARY FUNCTIONS    ###
#################################


def n_nodes(node):
    if node.type in ['L', 'G', 'M']:
        return 1
    if node.type in ['S', 'P']:
        n = 1 + np.sum([n_nodes(c) for c in node.children])
    return n


def delete(node):
    for c in node.children:
        delete(c)
    node.parent = None
    node.left_child = None
    node.right_child = None
    node.sibling = None
    node = None


###########################
###    FIT FUNCTIONS    ###
###########################

@njit
def fit_gaussian(node, data, upper, lower):
    assert node.type == 'G', "Only gaussian leaves fit data."
    node.n = data.shape[0]
    m = np.nanmean(data[:, node.scope])
    if np.isnan(m):
        node.mean = 0.  # Assuming the data has been standardized
        node.std = np.sqrt(1.)
    else:
        node.mean = m
        if node.n > 1:  # Avoid runtimewarning
            node.std = np.std(data[:, node.scope])
        else:
            node.std = np.sqrt(1.)  # Probably not the best solution here
        node.std = max(np.sqrt(1.), node.std)
        # Compute the tresholds to truncate the Gaussian.
        # The Gaussian has support [a, b]
    node.a = lower
    node.b = upper


@njit
def fit_multinomial(node, data, k):
    assert node.type == 'M', "Node is not a multinomial leaf."
    d = data[~np.isnan(data[:, node.scope].ravel()), :]  # Filter missing
    d = data[:, node.scope].ravel()  # Project to scope
    d = np.asarray(d, np.int64)
    if d.shape[0] > 0:
        counts = bincount(d, k) + 1e-6
        node.logcounts = np.log(counts)
        node.p = counts/(d.shape[0] + k*1e-6)
    else:
        node.p = np.ones(k) * (1/k)
    node.logp = np.log(np.asarray(node.p))


@njit
def fit_multinomial_with_counts(node, counts):
    assert node.type == 'M', "Node is not a multinomial leaf."
    node.logcounts = np.log(np.asarray(counts))
    node.p = counts/np.sum(counts)
    node.logp = np.log(np.asarray(node.p))


@njit
def fit_indicator(node, value, comparison):
    node.value = value
    node.comparison = comparison


###########################
### EVALUATE FUNCTIONS  ###
###########################

@njit
def eval_eq(scope, value, evi):
    """
        Evaluates an indicator of the type 'equal'.
        True if evi[scope] == value, False otherwise.
    """
    s, v = scope[0], value[0]
    res = np.zeros(evi.shape[0], dtype=np.float64)-np.Inf
    for i in range(evi.shape[0]):
        if (evi[i, s] == v) or np.isnan(evi[i, s]):
            res[i] = 0
    return res


@njit
def eval_leq(scope, value, evi):
    """
        Evaluates an indicator of the type 'less or equal'.
        True if evi[scope] <= value, False otherwise.
    """
    s, v = scope[0], value[0]
    res = np.zeros(evi.shape[0], dtype=np.float64)
    for i in range(evi.shape[0]):
        if evi[i, s] > v:
            res[i] = -np.Inf
    return res


@njit
def eval_g(scope, value, evi):
    """
        Evaluates an indicator of the type 'greater'.
        True if evi[scope] > value, False otherwise.
    """
    s, v = scope[0], value[0]
    res = np.zeros(evi.shape[0], dtype=np.float64)
    for i in range(evi.shape[0]):
        if evi[i, s] <= v:
            res[i] = -np.Inf
    return res


@njit
def eval_in(scope, value, evi):
    """
        Evaluates an indicator of the type 'in':
        True if evi[scope] in value, False otherwise.
    """
    s, v = scope[0], value[0]
    res = np.zeros(evi.shape[0], dtype=np.float64)
    for i in range(evi.shape[0]):
        if not isin(evi[i, s], value):
            res[i] = -np.Inf
    return res


@njit
def eval_gaussian(node, evi):
    """ Evaluates a Gaussian leaf. """
    return logtrunc_phi(evi[:, node.scope].ravel(), node.mean, node.std, node.a, node.b).reshape(-1)


@njit
def eval_m(node, evi):
    """ Evaluates a multinomial leaf. """
    s = node.scope[0]
    res = np.zeros(evi.shape[0], dtype=np.float64)
    for i in range(evi.shape[0]):
        if not np.isnan(evi[i, s]):
            res[i] = node.logp[int(evi[i, s])]
    return res


@njit
def compute_batch_size(n_points, n_features):
    maxmem = max(n_points * n_features + (n_points)/10, 10 * 2 ** 17)
    batch_size = (-n_features + np.sqrt(n_features ** 2 + 4 * maxmem)) / 2
    return int(batch_size)


@njit(parallel=True)
def eval_root(node, evi):
    """
        Evaluates the PC rooted at `node` at evidence `evi`.
        This function only applies to the root of a PC.
    """
    if node.type == 'S':
        logprs = np.zeros((evi.shape[0], node.nchildren), dtype=np.float64)
        for i in nb.prange(node.nchildren):
            logprs[:, i] = evaluate(node.children[i], evi) + node.logw[i]
        res = logsumexp2(logprs, axis=1)
        return res
    elif node.type == 'P':
        logprs = np.zeros((evi.shape[0], node.nchildren), dtype=np.float64)
        logprs[:, 0] = evaluate(node.children[0], evi)
        nonzero = np.where(logprs[:, 0] != -np.Inf)[0]
        if len(nonzero) > 0:
            for i in nb.prange(1, node.nchildren):
                # Only evaluate nonzero examples to save computation
                logprs[nonzero, i] = evaluate(node.children[i], evi[nonzero, :])
        return np.sum(logprs, axis=1)
    else:
        return evaluate(node, evi)


@njit(parallel=True)
def eval_root_children(node, evi):
    """
        Evaluates the PC rooted at `node` at evidence `evi`.
        This function only applies to the root of a PC.
    """
    n_threads = nb.config.NUMBA_DEFAULT_NUM_THREADS
    sizes = np.full(n_threads, node.nchildren // n_threads, dtype=np.int32)
    sizes[:node.nchildren % n_threads] += 1
    offset_in_buffers = np.zeros(n_threads, dtype=np.int32)
    offset_in_buffers[1:] = np.cumsum(sizes[:-1])
    logprs = np.zeros((evi.shape[0], node.nchildren), dtype=np.float64)
    for thread_idx in nb.prange(n_threads):
        start = offset_in_buffers[thread_idx]
        stop = start + sizes[thread_idx]
        for i in range(start, stop):
            logprs[:, i] = evaluate(node.children[i], evi) + node.logw[i]
    return logprs


@njit
def evaluate(node, evi):
    """
        Evaluates the PC rooted at `node` at evidence `evi`.
        This function applies to any node in a PC and is called recursively.
    """
    if node.type == 'L':
        if node.comparison == 0:  # IN
            return eval_in(node.scope, node.value.astype(np.int64), evi)
        elif node.comparison == 1:  # EQ
            return eval_eq(node.scope, node.value.astype(np.float64), evi)
        elif node.comparison == 3:  # LEQ
            return eval_leq(node.scope, node.value, evi)
        elif node.comparison == 4:  # G
            return eval_g(node.scope, node.value, evi)
    elif node.type == 'M':
        return eval_m(node, evi)
    elif node.type == 'G':
        return eval_gaussian(node, evi)
    elif node.type == 'U':
        return np.ones(evi.shape[0]) * node.value
    elif node.type == 'P':
        logprs = np.zeros((evi.shape[0], node.nchildren), dtype=np.float64)
        logprs[:, 0] = evaluate(node.children[0], evi)
        nonzero = np.where(logprs[:, 0] != -np.Inf)[0]
        if len(nonzero) > 0:
            for i in nb.prange(1, node.nchildren):
                # Only evaluate nonzero examples to save computation
                logprs[nonzero, i] = evaluate(node.children[i], evi[nonzero, :])
        return np.sum(logprs, axis=1)
    elif node.type == 'S':
        logprs = np.zeros((evi.shape[0], node.nchildren), dtype=np.float64)
        for i in nb.prange(node.nchildren):
            logprs[:, i] = evaluate(node.children[i], evi) + node.logw[i]
        res = logsumexp2(logprs, axis=1)
        return res
    return np.zeros(evi.shape[0])


@njit(parallel=True)
def eval_root_class(node, evi, class_var, n_classes, naive):
    """
        Evaluates the PC rooted at `node` at evidence `evi`.
        Same as `eval_root` but evaluates all possibles instantiations of the
        class variable at once. For that three extra parameters are required.

        Parameters
        ----------
        class_var: int
            Index of the class variable
        n_classes: int
            Number of classes in the data
        naive: boolean
            Whether to simply propagate the counts (Friedman method).

        Returns
        -------
        logprs: numpy array (float) of shape (n_samples, n_classes, n_trees)
    """
    n_threads = nb.config.NUMBA_DEFAULT_NUM_THREADS
    sizes = np.full(n_threads, node.nchildren // n_threads, dtype=np.int32)
    sizes[:node.nchildren % n_threads] += 1
    offset_in_buffers = np.zeros(n_threads, dtype=np.int32)
    offset_in_buffers[1:] = np.cumsum(sizes[:-1])
    logprs = np.zeros((evi.shape[0], n_classes, node.nchildren), dtype=np.float64)
    for thread_idx in nb.prange(n_threads):
        start = offset_in_buffers[thread_idx]
        stop = start + sizes[thread_idx]
        for i in range(start, stop):
            if naive:
                logprs[:, :, i] = evaluate_class(node.children[i], evi, class_var, n_classes, naive)  # no weights here
            else:
                logprs[:, :, i] = evaluate_class(node.children[i], evi, class_var, n_classes, naive) + node.logw[i]
    return logprs


@njit
def evaluate_class(node, evi, class_var, n_classes, naive):
    """
        Evaluates the PC rooted at `node` at evidence `evi`.
        This function applies to any node in a PC and is called recursively.
        Same as `evaluate` but evaluates all possibles instantiations of the
        class variable at once. For that three extra parameters are required.

        Parameters
        ----------
        class_var: int
            Index of the class variable
        n_classes: int
            Number of classes in the data
        naive: boolean
            Whether to simply propagate the counts (Friedman method).
    """
    if node.type == 'L':
        res = np.zeros((evi.shape[0], 1))
        if node.comparison == 0:  # IN
            res[:, 0] = eval_in(node.scope, node.value.astype(np.int64), evi)
        elif node.comparison == 1:  # EQ
            res[:, 0] = eval_eq(node.scope, node.value.astype(np.float64), evi)
        elif node.comparison == 3:  # LEQ
            res[:, 0] = eval_leq(node.scope, node.value, evi)
        elif node.comparison == 4:  # G
            res[:, 0] = eval_g(node.scope, node.value, evi)
        return res
    elif node.type == 'M':
        if isin(class_var, node.scope):
            if naive:
                return np.zeros((evi.shape[0], n_classes)) + node.logcounts
            else:
                return np.zeros((evi.shape[0], n_classes)) + node.logp
        if naive:
            return np.zeros((evi.shape[0], 1))
        res = np.zeros((evi.shape[0], 1))
        res[:, 0] = eval_m(node, evi)
        return res
    elif node.type == 'G':
        res = np.zeros((evi.shape[0], 1))
        if naive:
            return res
        res[:, 0] = eval_gaussian(node, evi)
        return res
    elif node.type == 'U':
        if naive:
            return np.zeros((evi.shape[0], 1))
        return np.ones((evi.shape[0], 1)) * node.value
    elif node.type == 'P':
        logprs = np.zeros((evi.shape[0], n_classes, node.nchildren), dtype=np.float64)
        logprs[:, :, 0] = evaluate_class(node.children[0], evi, class_var, n_classes, naive)
        nonzero = ~np.isinf(logprs[:, 0, 0])
        if np.sum(nonzero) > 0:
            for i in range(1, node.nchildren):
                # Only evaluate nonzero examples to save computation
                logprs[nonzero, :, i] = evaluate_class(node.children[i], evi[nonzero, :], class_var, n_classes, naive)
        return np.sum(logprs, axis=2)
    elif node.type == 'S':
        logprs = np.zeros((evi.shape[0], n_classes, node.nchildren), dtype=np.float64)
        if naive:
            for i in range(node.nchildren):
                logprs[:, :, i] = evaluate_class(node.children[i], evi, class_var, n_classes, naive)  # no weights here
        else:
            for i in range(node.nchildren):
                logprs[:, :, i] = evaluate_class(node.children[i], evi, class_var, n_classes, naive) + node.logw[i]
        return logsumexp3(logprs, axis=2)
    return np.zeros((evi.shape[0], 1))


##################################
###    ROBUSTNESS FUNCTIONS    ###
##################################


@njit
def eval_m_rob(node, evi, n_classes, eps, ismax):
    s = node.scope[0]
    res = np.zeros(evi.shape[0], dtype=np.float64)
    for i in range(evi.shape[0]):
        if not np.isnan(evi[i, s]):
            logprs = np.zeros(node.p.shape)-np.Inf
            logprs[int(evi[i, s])] = 0.
            econt = econtaminate(node.p, logprs, eps, ismax)
            res[i] = lse(logprs + np.log(econt))
    return res


@njit
def econtaminate(vec, logprs, eps, ismax):
    """
        Returns a `eps`-contaminated version of `vec`.

        Parameters
        ----------
        vec: numpy array
            The original array of parameters.
        logprs: numpy array (same dimension as `vec`)
            The log-density associated to each parameter in vec.
            Typically, logprs[i] is the log-density (at a given evidence) of
            the PC rooted at the ith child of the sum node with parameters `vec`.
        eps: float between 0 and 1.
            The epsilon used to contaminate the parameters.
            See https://arxiv.org/abs/2007.05721
        ismax: boolean
            If True, the parameters in `vec` are perturbed so that the dot
            product between the `eps`-contaminated version of `vec` and `logprs`
            is maximised. If False, this dot product is minimised.
    """
    econt = np.asarray(vec) * (1-eps)
    room = 1 - np.sum(econt)
    if ismax:
        order = np.argsort(-1*logprs)
    else:
        order = np.argsort(logprs)
    for i in order:
        if room > eps:
            econt[i] = econt[i] + eps
            room -= eps
        else:
            econt[i] = econt[i] + room
            break
    return econt


@njit
def evaluate_rob_class(node, evi, class_var, n_classes, eps, maxclass):
    """
        Computes the expected value of min[P(evi, y') - P(evi, maxclass)] and
        min[P(evi, y') - P(evi, maxclass)], where maxclass is the predicted
        class and y' is any other possible class. Because this difference might
        be negative, we need to propagate negative values through the network
        which required signed values (see signed.py).

        Parameters
        ----------
        node: Node object (nodes.py)
            The root of the Probabilistic Circuit.
        evi: numpy array of size m
            Single realisation of m variables.
        class_var: int
            Index of the class variable
        n_classes: int
            Number of classes in the data
        eps: float between 0 and 1.
            The epsilon used to contaminate the parameters.
            See https://arxiv.org/abs/2007.05721
        maxclass: int
            Index of the predicted class.

        Returns
        -------
        res_min, res_max: signed arrays of size n_class
            The minimum and maximum values the density function assumes within
            the epsilon-contaminated set.
    """
    one = np.array([1.])
    if node.type == 'L':
        if node.comparison == 0:  # IN
            res = eval_in(node.scope, node.value.astype(np.int64), evi)
        elif node.comparison == 1:  # EQ
            res = eval_eq(node.scope, node.value.astype(np.float64), evi)
        elif node.comparison == 3:  # LEQ
            res = eval_leq(node.scope, node.value, evi)
        elif node.comparison == 4:  # G
            res = eval_g(node.scope, node.value, evi)
        res = signed(res, one)
        return res, res  # min, max are the same
    elif node.type == 'M':
        if isin(class_var, node.scope):
            indicators = np.ones(n_classes, dtype=boolean) # np.bool
            indicators[maxclass] = 0
            # Min
            econt_min = np.asarray(node.p)*(1-eps)
            econt_min[indicators] = econt_min[indicators] + eps
            econt_min = np.asarray(econt_min[maxclass]) - econt_min
            res_min = signed(econt_min, None)
            # Max
            econt_max = np.asarray(node.p)*(1-eps)
            econt_max[~indicators] = econt_max[~indicators] + eps
            econt_max = np.asarray(econt_max[maxclass]) - econt_max
            res_max = signed(econt_max, None)
        else:
            # Min
            res_min = eval_m_rob(node, evi, n_classes, eps, False)
            # Max
            res_max = eval_m_rob(node, evi, n_classes, eps, True)
            res_min, res_max = signed(res_min, one), signed(res_max, one)
        return res_min, res_max
    elif node.type == 'G':
        res = np.zeros((evi.shape[0], 1))
        delta = eps/2
        point = evi[:, node.scope].ravel()
        left = logtrunc_phi(point, node.mean-delta, node.std, node.a, node.b).reshape(-1)
        right = logtrunc_phi(point, node.mean+delta, node.std, node.a, node.b).reshape(-1)
        if left[0] >= right[0]:
            res_min = signed(right, one)
            res_max = signed(left, one)
        else:
            res_min = signed(left, one)
            res_max = signed(right, one)
        if (point[0] + delta >= node.mean) & (point[0] - delta <= node.mean):
            res_max = signed(np.asarray([node.mean]), one)
        return res_min, res_max
    elif node.type == 'P':
        res_min = signed(np.array([1]), None)
        res_max = signed(np.array([1]), None)
        res_min_cl, res_max_cl = None, None
        for i in range(node.nchildren):
            # Only evaluate nonzero examples to save computation
            child_min, child_max = evaluate_rob_class(node.children[i], evi, class_var, n_classes, eps, maxclass)
            if (i == 0) and (np.all(child_max.nonzero()) == False) and (np.all(child_min.nonzero()) == False):
                return child_max, child_max
            if isin(class_var, node.children[i].scope):
                res_min_cl, res_max_cl = child_min, child_max
            else:
                res_min = signed_prod(res_min, child_min)
                res_max = signed_prod(res_max, child_max)
        new_res_min = None
        new_res_max = None
        if res_min_cl != None and res_max_cl != None:
            for j in range(n_classes):
                if res_min_cl.sign[j] < 0:
                    res_min_j = signed_prod(res_max, res_min_cl.get(j))
                else:
                    res_min_j = signed_prod(res_min, res_min_cl.get(j))
                new_res_min = signed_join(new_res_min, res_min_j)
                if res_max_cl.sign[j] < 0:
                    res_max_j = signed_prod(res_min, res_max_cl.get(j))
                else:
                    res_max_j = signed_prod(res_max, res_max_cl.get(j))
                new_res_max = signed_join(new_res_max, res_max_j)
            return new_res_min, new_res_max
        return res_min, res_max
    elif node.type == 'S':
        values_min, values_max = np.zeros((node.nchildren, n_classes)), np.zeros((node.nchildren, n_classes))
        signs_min, signs_max = np.zeros((node.nchildren, n_classes)), np.zeros((node.nchildren, n_classes))
        for i in range(node.nchildren):
            res_min, res_max = evaluate_rob_class(node.children[i], evi, class_var, n_classes, eps, maxclass)
            values_min[i, :], signs_min[i, :] = res_min.value, res_min.sign
            values_max[i, :], signs_max[i, :] = res_max.value, res_max.sign
        res_min, res_max = signed(np.zeros(n_classes), None), signed(np.zeros(n_classes), None)
        for j in range(n_classes):
            min_j = signed(values_min[:, j], signs_min[:, j])
            econt_min = signed_econtaminate(node.w, min_j, eps, False)
            res_min_j = signed_prod(min_j, econt_min)
            res_min.insert(res_min_j.reduce(), j)
            max_j = signed(values_max[:, j], signs_max[:, j])
            econt_max = signed_econtaminate(node.w, max_j, eps, True)
            res_max_j = signed_prod(max_j, econt_max)
            res_max.insert(res_max_j.reduce(), j)
        return res_min, res_max
    res = signed(np.ones(evi.shape[0]), None)  # Just so numba compiles
    return res, res


@njit(parallel=True)
def compute_rob_class(node, evi, class_var, n_classes):
    """
        Compute the robustness of the PC rooted at `node` for each instance in `evi`.

        Parameters
        ----------
        node: Node object (nodes.py)
            The root of the Probabilistic Circuit.
        evi: numpy array n x m
            Data with n samples and m variables.
        class_var: int
            Index of the class variable
        n_classes: int
            Number of classes in the data
    """
    rob = np.zeros(evi.shape[0])
    logprobs = evaluate_class(node, evi, class_var, n_classes, False)
    maxclass = nb_argmax(logprobs, axis=1)
    for i in prange(evi.shape[0]):
        rob[i] = rob_loop_class(node, evi[i:i+1, :], class_var, n_classes, maxclass[i])
    return maxclass, rob


@njit
def rob_loop_class(node, evi, class_var, n_classes, maxclass):
    """
        Same as `compute_rob_class` but for a single instance.

        Performs a binary search, increasing the value of eps until
        min[P(evi, y') - P(evi, maxclass)] becomes negative. A negative value
        here indicates that not every PC in the class of PCs defined by a
        contamination of eps yields the same classification, and hence the
        robustness value should be less than eps.
        See https://arxiv.org/abs/2007.05721 for details.
    """
    lower = 0
    upper = 1
    it = 0
    while (lower < upper - .005) & (it <= 200):
        ok = True
        rob = (lower + upper)/2
        min_values, max_values = evaluate_rob_class(node, evi, class_var, n_classes, rob, maxclass)
        for j in range(n_classes):
            if j != maxclass:
                if min_values.get(j).sign[0] <= 0:
                    ok = False
                    break
        if ok:
            lower = rob
        else:
            upper = rob
        it += 1
    return rob


@njit
def nb_argmax(x, axis):
    """ Implementation of numpy.argmax in numba. """
    assert (axis==0) | (axis==1), "axis must be set to either 0 or 1."
    if axis == 0:
        res = np.zeros(x.shape[1], dtype=np.int64)
        for i in range(x.shape[1]):
            res[i] = np.argmax(x[:, i])
    elif axis == 1:
        res = np.zeros(x.shape[0], dtype=np.int64)
        for i in range(x.shape[0]):
            res[i] = np.argmax(x[i, :])
    return res


@njit
def nb_argsort(x, axis):
    """ Implementation of numpy.argsort in numba. """
    assert (axis==0) | (axis==1), "axis must be set to either 0 or 1."
    ordered = np.zeros_like(x)
    if axis == 0:
        for i in range(x.shape[1]):
            ordered[:, i] = np.argsort(x[:, i])
        return ordered
    elif axis == 1:
        for i in range(x.shape[0]):
            ordered[i, :] = np.argsort(x[i, :])
        return ordered
