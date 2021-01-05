from math import floor
import numba as nb
from numba import njit, int16, int64, float64, optional, prange, deferred_type, types, boolean
from numba.typed import List
from numba.experimental import jitclass
import numpy as np
import operator
import random
from scipy import stats
from tqdm import tqdm

from .learning import LearnSPN, fit
from .nodes import SumNode, ProdNode, Leaf, GaussianLeaf, MultinomialLeaf, fit_multinomial, fit_gaussian
from .pc import PC
from .utils import bincount, isin_nb, resample_strat
from .split import find_best_split, Split


@njit(parallel=True)
def evaluate(node, X, n_classes=2):
    """
        Returns the corresponding class counts for each instance in X.
        Missing variables are handled via surrogate splits. If no surrogate splits
        are available for a given decision node, an instance missing the corresponding
        decision variable is sent to one of that node's children at random.

        Parameters
        ----------
        node: TreeNode object
            Typically the root node of the tree.
        X: numpy n x m
            Numpy array comprising n realisations (instances) of m variables.
        n_classes: int
            Number of classes in the data.
    """
    n_samples = X.shape[0]
    n_threads = nb.config.NUMBA_DEFAULT_NUM_THREADS
    res = np.empty((X.shape[0], n_classes), dtype=np.float64)

    if n_samples < n_threads:
        for i in range(n_samples):
            res[i, :] = evaluate_instance(node, X[i, :])
        return res

    sizes = np.full(n_threads, n_samples // n_threads, dtype=np.int64)
    sizes[:n_samples % n_threads] += 1
    offset_in_buffers = np.zeros(n_threads, dtype=np.int64)
    offset_in_buffers[1:] = np.cumsum(sizes[:-1])

    for thread_idx in prange(n_threads):
        start = offset_in_buffers[thread_idx]
        stop = start + sizes[thread_idx]
        for i in range(start, stop):
            res[i, :] = evaluate_instance(node, X[i, :])
    return res


@njit
def evaluate_instance(node, x):
    """
        Routes a single instance x and returns the corresponding class counts.
        Missing variables are handled via surrogate splits. If no surrogate splits
        are available for a given decision node and x is missing the corresponding
        decision variable, then x is sent to one of that node's children at random.

        Parameters
        ----------
        node: TreeNode object
            Typically the root node of the tree.
        x: numpy array of size m
            Numpy array comprising a single realisation of m variables.
        n_classes: int
            Number of classes in the data.
    """
    s = node.split
    if s is None:
        # Leaf Node
        return node.counts
    if not np.isnan(x[s.var]):
        # Observed split variable
        if s.type == 'num':
            go_left = x[s.var] <= s.threshold[0]
        else:
            go_left = np.any(isin_nb(np.array(x[s.var]), s.threshold))
    else:
        # Missing split variable; look for surrogate split
        var = -1
        for i in range(len(s.surr_var)):
            if not np.isnan(x[s.surr_var[i]]):
                var = s.surr_var[i]
                thr = s.surr_thr[i]
                left = s.surr_go_left[i]
                break
        if var == -1:
            # If all missing, take a side at random
            go_left = s.surr_blind
        elif x[var] <= thr:
            go_left = left
        else:
            go_left = ~left
    if go_left:
        return evaluate_instance(node.left_child, x)
    else:
        return evaluate_instance(node.right_child, x)


@njit
def build_tree(tree, parent, counts, ordered_ids):
    """
        Recursively splits the data, which contained in the tree object itself
        and is indexed by ordered_ids.

        Parameters
        ----------
        tree: Tree object
        parent: TreeNode object
            The last node added to the tree, which will be the parent of the
            two nodes resulting from the split (if any) of this function call.
        counts: numpy array (int)
            The class counts of the samples reaching the parent node.
        ordered_ids: numpy array (int)
            The ids of the samples reaching the parent node.
    """
    root = TreeNode(0, counts, parent, ordered_ids, False)
    queue = List()
    queue.append(root)
    n_nodes = 1
    while len(queue) > 0:
        node = queue.pop(0)
        split = find_best_split(node, tree)
        if split is not None:
            node.split = split
            left_child = TreeNode(n_nodes, split.left_counts, node,
                                  split.left_ids, False)
            node.left_child = left_child
            queue.append(left_child)
            n_nodes += 1
            right_child = TreeNode(n_nodes, split.right_counts, node,
                                   split.right_ids, False)
            node.right_child = right_child
            queue.append(right_child)
            n_nodes += 1
        else:
            node.isleaf = True
        tree.depth = max(tree.depth, node.depth)
    return root, n_nodes


@njit(parallel=True)
def build_forest(X, y, n_estimators, bootstrap, ncat, imp_measure,
                 min_samples_split, min_samples_leaf, max_features, max_depth,
                 surrogate):
    """
        Fits a Random Forest to (X, y).
        For the most part, the parameters match those of the Random Forest
        implementation in scikit-learn.
    """
    n_samples = X.shape[0]
    n_classes = np.max(y)+1  # We assume ordinals from 0, 1, 2, ..., max(y)
    n_threads = nb.config.NUMBA_DEFAULT_NUM_THREADS
    estimators = [Tree(ncat, imp_measure, min_samples_split, min_samples_leaf,
                       max_features, max_depth, surrogate)
                  for i in range(n_estimators)]

    sizes = np.full(n_threads, n_estimators // n_threads, dtype=np.int64)
    sizes[:n_estimators % n_threads] += 1
    offset_in_buffers = np.zeros(n_threads, dtype=np.int64)
    offset_in_buffers[1:] = np.cumsum(sizes[:-1])

    for thread_idx in prange(n_threads):
        start = offset_in_buffers[thread_idx]
        stop = start + sizes[thread_idx]
        for i in range(start, stop):
            Xtree_, ytree_, _ = resample_strat(X, y, n_classes)
            estimators[i].fit(Xtree_, ytree_)
    return estimators


@njit
def add_split(tree_node, pc_node, ncat, root=False):
    """
        Creates a new split in the PC. Each split consists of a Sum node with
        two Product nodes as children. Each Product node has an indicator
        variable mimicking the hard split in a Decision Tree. For instance,
        if the split in the Decision Tree is X_i <= 5, one Product node will
        have an indicator that evaluates to one only if X_i <= 5 and to zero
        otherwise.

        This does not yield a decomposable PC, but facilitates computation.
        See the supp. material at https://arxiv.org/abs/2006.14937 for more details.
    """
    split_col = tree_node.split.var
    split_value = tree_node.split.threshold
    n_points_left = len(tree_node.split.left_ids)
    n_points_right = len(tree_node.split.right_ids)
    lp = np.sum(np.where(ncat==1, 0, ncat)) * 1e-6 # LaPlace counts
    scope = np.arange(len(ncat), dtype=np.int64)
    if root:
        sumnode = pc_node
    else:
        sumnode = SumNode(scope=scope, parent=pc_node,
                          n=n_points_left+n_points_right+lp)
    upper1 = pc_node.upper.copy()
    lower1 = pc_node.lower.copy()
    upper2 = pc_node.upper.copy()
    lower2 = pc_node.lower.copy()
    if ncat[split_col] > 1:
        cat = np.arange(ncat[split_col], dtype=np.float64)
        mask = ~isin_nb(cat, split_value)
        out_split_value = cat[mask]

        upper1[split_col] = len(split_value)  # upper: number of variables in
        lower1[split_col] = len(out_split_value)  # lower: number of variables out
        p1 = ProdNode(scope=scope, parent=sumnode, n=n_points_left+lp)
        p1.upper, p1.lower = upper1, lower1
        ind1 = Leaf(scope=np.array([split_col]), parent=p1, n=n_points_left+lp,
                    value=split_value, comparison=0)  # Comparison IN

        upper2[split_col] = len(out_split_value)  # upper: number of variables in
        lower2[split_col] = len(split_value)  # lower: number of variables out
        p2 = ProdNode(scope=scope, parent=sumnode, n=n_points_right+lp)
        p2.upper, p2.lower = upper2, lower2
        ind2 = Leaf(scope=np.array([split_col]), parent=p2, n=n_points_right+lp,
                    value=out_split_value, comparison=0)  # Comparison IN
    else:
        upper1[split_col] = min(split_value[0], upper1[split_col])
        p1 = ProdNode(scope=scope, parent=sumnode, n=n_points_left+lp)
        p1.upper, p1.lower = upper1, lower1
        ind1 = Leaf(scope=np.array([split_col]), parent=p1, n=n_points_left+lp,
                    value=split_value, comparison=3)  # Comparison <=

        lower2[split_col] = max(split_value[0], lower2[split_col])
        p2 = ProdNode(scope=scope, parent=sumnode, n=n_points_right+lp)
        p2.upper, p2.lower = upper2, lower2
        ind2 = Leaf(scope=np.array([split_col]), parent=p2, n=n_points_right+lp,
                    value=split_value, comparison=4)  # Comparison >
    return p1, p2


@njit
def add_dist(tree_node, pc_node, data, ncat, learnspn, max_height, thr):
    """
        Fits a density estimator at a given leaf.

        Parameters
        ----------
        tree_node: TreeNode object
            The corresponding node at the original Decision Tree.
        pc_node: Node object (nodes.py)
            The corresponding node at the PC.
        data: numpy n (instances) x m (variables)
            The data reaching this leaf.
        ncat: numpy array (int)
            The number of categories of each variable.
        learnspn: int
            The number of samples (at a given leaf) required to run LearnSPN.
        max_height: int
            The maximum height (eq. depth) of the LearnSPN structure.
        thr: float
            The threshold (p-value) below which we reject the hypothesis of
            independence. In that case, they are considered dependent and
            assigned to the same cluster.
    """
    counts = tree_node.counts
    n_points = len(tree_node.idx)
    data_leaf = data[tree_node.idx, :]
    scope = np.arange(data.shape[1], dtype=np.int64)
    lp = np.sum(np.where(ncat==1, 0, ncat)) * 1e-6 # LaPlace counts
    upper, lower = pc_node.upper, pc_node.lower

    if n_points >= learnspn:
        # The class variable is modeled independently as multinomial data
        learner = LearnSPN(ncat, thr, 2, max_height, None)
        fit(learner, data_leaf, pc_node)
    else:
        for var in scope:
            if ncat[var] > 1:
                leaf = MultinomialLeaf(pc_node, np.array([var]), n_points+lp)
                fit_multinomial(leaf, data_leaf, int(ncat[var]))
            else:
                leaf = GaussianLeaf(pc_node, np.array([var]), n_points+lp)
                fit_gaussian(leaf, data_leaf, upper[var], lower[var])
        return None


def tree2pc(tree, learnspn=np.Inf, max_height=1000000, thr=0.01):
    """
        Converts a Decision Tree into a Probabilistic Circuit.

        Parameters
        ----------
        tree: Tree object
            The original Decision Tree.
        learnsnp: int
            The number of samples (at a given leaf) required to run LearnSPN.
            Set to infinity by default, so as not to run LearnSPN anywhere.
        max_height: int
            The maximum height (eq. depth) of the LearnSPN structure.
        thr: float
            The threshold (p-value) below which we reject the hypothesis of
            independence. In that case, they are considered dependent and
            assigned to the same cluster.
    """
    scope = np.array([i for i in range(tree.X.shape[1]+1)], dtype=np.int64)
    data = np.concatenate([tree.X, np.expand_dims(tree.y, axis=1)], axis=1)
    ncat = np.array(tree.ncat)
    lp = np.sum(np.where(ncat==1, 0, ncat)) * 1e-6 # LaPlace counts
    upper = ncat.copy().astype(np.float64)
    upper[upper == 1] = np.Inf
    lower = ncat.copy().astype(np.float64)
    lower[ncat == 1] = -np.Inf
    lower = np.ones(data.shape[1])*(-np.Inf)
    classcol = len(ncat)-1
    # Create a new PC with a Sum node as root.
    pc = PC()
    pc_node = SumNode(scope=scope, parent=None, n=data.shape[0]+lp)
    pc.root = pc_node
    pc.ncat = ncat
    tree_queue = [tree.root]
    pc_queue = [pc.root]
    root = True
    while len(tree_queue) > 0:
        tree_node = tree_queue.pop(0)
        pc_node = pc_queue.pop(0)
        # If node is leaf, fit a density estimator
        if tree_node.isleaf:
            add_dist(tree_node, pc_node, data, ncat, learnspn, max_height, thr)
        # If node is a decision node, add another split in the PC.
        else:
            p_left, p_right = add_split(tree_node, pc_node, ncat, root)
            tree_queue.extend([tree_node.left_child, tree_node.right_child])
            pc_queue.extend([p_left, p_right])
        root = False
    return pc


def delete(node):
    if node.left_child is not None:
        delete(node.left_child)
        node.left_child = None
    if node.right_child is not None:
        delete(node.right_child)
        node.right_child = None
    node.parent = None
    node = None


def delete_tree(tree):
    delete(tree.root)


node_type = deferred_type()

@jitclass([
    ('id', int64),  # Unique (random) id
    ('counts', int64[:]),  # Class counts of the data reaching the node.
    ('idx', int64[:]),  # Indices of the samples reaching the node.
    ('split', optional(Split.class_type.instance_type)),  # Split object
    ('parent', optional(node_type)),
    ('left_child', optional(node_type)),
    ('right_child', optional(node_type)),
    ('isleaf', optional(nb.boolean)),
    ('depth', int16),
])
class TreeNode:
    """
        Class defining each node in a Decision Tree.
    """
    def __init__(self, id, counts, parent, idx, isleaf):
        self.id = id
        self.counts = counts
        self.parent = parent
        self.idx = idx
        self.isleaf = isleaf
        self.split = None
        self.left_child = None
        self.right_child = None
        if parent is not None:
            self.depth = parent.depth + 1
        else:
            self.depth = 0

node_type.define(TreeNode.class_type.instance_type)


@jitclass([
    ('X', optional(float64[:,:])),  # Explanatory variables
    ('y', optional(int64[:])),  # Target variable
    ('ncat', optional(int64[:])),  # Number of categories of each variable
    ('scope', optional(int64[:])),  # Index the pertinent variables at each node.
    ('imp_measure', types.string),  # Type of impurity measure used to define splits.
    ('min_samples_leaf', int64),  # The minimum number of samples at each leaf.
    ('min_samples_split', int64),  # The minimum number of samples required to define a split.
    ('n_classes', int64),  # Number of classes in the data.
    ('max_features', optional(int64)),  # Maximum number of features to consider at each split.
    ('n_nodes', int64),  # Total number of nodes in the tree.
    ('root', TreeNode.class_type.instance_type),  # The root node of the tree (TreeNode object).
    ('depth', int16),  # The depth of the tree.
    ('max_depth', int64),  # The maximum depth of the tree.
    ('surrogate', boolean),  # Whether to learn surrogate splits at each decision node.
])
class Tree:
    """
        Decision Tree implementation in numba. Most parameters (see list above)
        match the Decision Tree implementation in scikit-learn.
    """
    def __init__(self, ncat=None, imp_measure='gini', min_samples_split=2, min_samples_leaf=1, max_features=None, max_depth=1e6, surrogate=False):
        self.X = None
        self.y = None
        self.ncat = ncat
        self.scope = None
        self.imp_measure = imp_measure
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_nodes = 0
        self.n_classes = 0
        self.depth = 0
        self.max_depth = max_depth
        self.root = TreeNode(0, np.empty(0, dtype=np.int64), None, np.empty(0, dtype=np.int64), False)
        self.surrogate = surrogate

    def fit(self, X, y):
        """ Fits the Random Forest to (X, Y). """
        self.X = X
        self.y = y
        self.n_classes = np.max(y)+1
        self.n_nodes = 0
        if self.max_features is None:
            self.max_features = X.shape[1]

        counts = bincount(y, self.ncat[-1])
        ordered_ids = np.arange(X.shape[0], dtype=np.int64)
        self.root, self.n_nodes = build_tree(self, None, counts, ordered_ids)

    def get_node(self, id):
        """ Fetchs node by its id. """
        queue = [self.root]
        while queue != []:
            node = queue.pop(0)
            if node.id == id:
                return node
            if node.split is not None:
                queue.extend([node.left_child, node.right_child])
        print("Node %d is not part of the network.", id)

    def predict(self, X):
        """
            Returns the most likely class (highest number of counts) for
            each instance in X.
        """
        root = self.root
        counts = evaluate(root, X, self.n_classes)
        res = np.empty(counts.shape[0], dtype=np.float64)
        for i in range(counts.shape[0]):
            res[i] = np.argmax(counts[i, :])
        return res

    def predict_proba(self, X):
        """
            Returns the conditional distribution over the class variable
            (normalised class counts) for each instance in X.
        """
        counts = evaluate(self.root, X, self.n_classes)
        res = np.empty(counts.shape)
        for i in range(counts.shape[0]):
            res[i, :] = (counts[i, :])/np.sum(counts[i, :])
        return res


class RandomForest:
    """
        Minimal Random Forest implementation as an ensemble of instances of the
        Tree class (see above).
        For the most part, the class attributes match those of the Random Forest
        implementation in scikit-learn.
    """
    def __init__(self, n_estimators=100, imp_measure='gini', min_samples_split=2,
                 min_samples_leaf=1, max_features=None, bootstrap=True,
                 ncat=None, max_depth=1e6, surrogate=False):

        self.n_estimators = n_estimators
        self.imp_measure = imp_measure
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.ncat = ncat.astype(np.int64)
        self.max_features = max_features
        self.max_depth = max_depth
        self.surrogate = surrogate

    def fit(self, X, y):
        """ Fits the Random Forest to (X, Y). """
        y = y.astype(np.int64)
        n = X.shape[0]
        if self.max_features is None:
            self.max_features = min(floor(X.shape[1]/3), X.shape[1])
        self.scope = np.array([i for i in range(X.shape[1]+1)])
        self.estimators = build_forest(X, y, self.n_estimators, self.bootstrap,
                                       self.ncat, self.imp_measure,
                                       self.min_samples_split,
                                       self.min_samples_leaf,
                                       self.max_features, self.max_depth,
                                       self.surrogate)

    def topc(self, learnspn=np.Inf, max_height=1000000, thr=0.01):
        """
            Returns a Probabilistic Circuit matching the tree structure.

            Parameters
            ----------
            learnspn: int
                The number of instances (at the leaves) required to run LearnSPN.
            max_height: int
                The maximum height (eq. depth) of the LearnSPN structure.
            thr: float
                The threshold (p-value) below which we reject the hypothesis of
                independence. In that case, they are considered dependent and
                assigned to the same cluster.
        """
        pc = PC()
        pc.root = SumNode(scope=self.scope, parent=None, n=1)
        for estimator in tqdm(self.estimators):
            tree_pc = tree2pc(estimator, learnspn=learnspn, max_height=max_height, thr=thr)
            pc.root.add_child(tree_pc.root)
        pc.ncat = tree_pc.ncat
        return pc

    def predict(self, X, vote=True):
        """
            Classifies samples X either by voting or averaging the conditional
            distribution of each tree.
        """
        if vote:
            votes = np.zeros(shape=(X.shape[0], self.n_estimators))
            for i, estimator in enumerate(self.estimators):
                votes[:, i] = estimator.predict(X)
            return stats.mode(votes, axis=1)[0].reshape(-1)
        else:
            probas = np.zeros(shape=(X.shape[0], self.n_estimators, self.ncat[-1]))
            for i, estimator in enumerate(self.estimators):
                probas[:, i] =  estimator.predict_proba(X)
            return np.mean(probas, axis=1)

    def delete(self):
        for est in self.estimators:
            delete_tree(est)
        self.estimators = None
