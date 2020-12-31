from numba import njit, float64, int64, optional
from numba.experimental import jitclass
import numpy as np

from .cluster import cluster
from .nodes import SumNode, ProdNode, Leaf, GaussianLeaf, MultinomialLeaf, fit_gaussian, fit_multinomial
from .utils import get_indep_clusters, isin


@jitclass([
    ('thr', float64),
    ('nclusters', int64),
    ('max_height', int64),
    ('ncat', int64[:]),
    ('classcol', optional(int64)),
])
class LearnSPN:
    """
        Learning method based on Gens and Domingos' LearnSPN.

        Attributes
        ----------
        thr: float
            p-value threshold for independence tests in product nodes.
        nclustes: int
            Number of clusters in sum nodes.
        max_height: int
            Maximum height (depth) of the network.
        ncat: numpy array
            Number of categories of each variable in the data. It should be
            one-dimensional and of length equal to the number of variables.
        classcol: int
            The index of the column containing class variable.
    """

    def __init__(self, ncat, thr=0.001, nclusters=2, max_height=1000000,
        classcol=None):
        """ Set the hyperparameters of the learning algorithm. """
        self.thr = thr
        self.nclusters = nclusters
        self.max_height = max_height
        self.ncat = ncat
        self.classcol = classcol


@njit
def fit(learnspn, data, last_node=None):
    """
        Learn an SPN given some data.

        Parameters
        ----------

        data: numpy array
            Rows are instances and columns variables.
        last_node: Node object
            The parent of the structure to be learned (if any).

        Returns
        -------

        root: SPN object
            The SPN learned from data.
    """
    scope = np.arange(data.shape[1], dtype=np.int64)  # Range over all variables (data cols)
    root = add_node(learnspn, data, scope, learnspn.max_height, last_node=last_node)
    return root


@njit
def add_node(learnspn, data, scope, max_height, last_node=None):
    """
        Add a new node to the network. This function is called recursively
        until max_height is reached or the data is fully partioned into
        leaf nodes.

        Parameters
        ----------

        data: numpy array
            Rows are instances and columns variables.
        scope: list
            Indices of the columns of the variables in the scope of a node.
        max_height: int
            Maximum height (depth) of the network.
        last_node: Node object
            The last node that was added to the network.

        Returns
        -------

        node: Node object
            The node just added to the network.
    """

#         assert len(scope) > 0, "Empty scope " + str(scope)
    n = len(scope)  # number of variables
    m = data.shape[0]  # number of instances
    if last_node is None:
        last_type = 'P'
    else:
        last_type = last_node.type
    node = None
    if n > 1:
        if (max_height <= 0) or (last_type != 'P'):
            node = add_prodnode(learnspn, data, scope, max_height, last_node)
    else:  # Single variable in the scope
        node = add_leaf(learnspn, data, scope, last_node)
    if node is None:
        # We were not able to cut vertically, so we run clustering of the
        # data (each row is a point), and then we use the result of
        # clustering to create the children of a sum node.
        node = add_sumnode(learnspn, data, scope, max_height, last_node)
    if node is None:
        # If no split found, assume fully factorised model.
        node = ProdNode(last_node, scope, data.shape[0])
        for var in scope:
            add_leaf(learnspn, data, [var], node)
    return node


@njit
def add_prodnode(learnspn, data, scope, max_height, last_node):
    """
        Add a product node to the network. We run pairwise independence
        tests in all variables so as to find clusters of independent
        variables. Those clusters are the children of a product node.

        Parameters
        ----------

        learnspn: LearnSPN object
            Holds the hyperparameters for LearnSPN
        data: numpy array
            Rows are instances and columns variables.
        scope: list
            Indices of the columns of the variables in the scope of a node.
        max_height: int
            Maximum height (depth) of the network.
        last_node: Node object
            The last node that was added to the network.

        Returns
        -------

        node: ProdNode object
            If an independence relation was found, and the data can be
            partioned into clusters of independent variables.
        None
            Otherwise.
    """
    n = len(scope)  # number of variables in the scope
    m = data.shape[0]  # number of instances
    if max_height > 0:
        clu = get_indep_clusters(data, scope, learnspn.ncat, learnspn.thr)
    else:
        clu = np.array([i for i in range(n)], dtype=np.float64)
    nclu = len(np.unique(clu))

    if learnspn.classcol is not None:
        class_in_scope = isin(int(learnspn.classcol), scope)
        classes_obs = len(np.unique(data[:, int(learnspn.classcol)]))
    else:
        classes_obs = 0
        class_in_scope = False
    class_split = (class_in_scope) and (classes_obs < 2)
    if (nclu > 1) or class_split:  # If independent clusters were found or split on class var.
        prodnode = ProdNode(last_node, scope, m)
        # If there is only one class in the data, add indicator node
        # Class-selective
        if class_split:
            # Add an indicator node for the given value of the class variable
            leaf = Leaf(prodnode, np.array([int(learnspn.classcol)]), data.shape[0],
                        np.unique(data[:, int(learnspn.classcol)]), 1)
            # Remove class variable from scope
            scope = scope.copy()
            scope = np.array([v for v in scope if v != learnspn.classcol], dtype=np.int64)
            clu = np.delete(clu, int(learnspn.classcol))
        for c in np.unique(clu):
            add_node(learnspn, data,
                     scope=np.array([scope[i] for i in np.where(clu == c)[0]], dtype=np.int64),
                     max_height=max_height-1,
                     last_node=prodnode)
        return prodnode
    else:  # No independent clusters were found.
        return None


@njit
def add_sumnode(learnspn, data, scope, max_height, last_node):
    """
        Add a sum node to the network. We split the data into nclusters
        clusters and create a child for each of them. The corresponding
        weights of each child is given by the relative size of each cluster.

        Parameters
        ----------

        learnspn: LearnSPN object
            Holds the hyperparameters for LearnSPN
        data: numpy array
            Rows are instances and columns variables.
        scope: list
            Indices of the columns of the variables in the scope of a node.
        max_height: int
            Maximum height (depth) of the network.
        last_node: Node object
            The last node that was added to the network.

        Returns
        -------

        node: SumNode object or None
            The sum node learned from the given data.
            None if no split was found.
    """
    n = len(scope)  # number of variables in the scope
    m = data.shape[0]  # number of instances
    class_in_scope = False
    if learnspn.classcol is not None:
        class_in_scope = isin(int(learnspn.classcol), scope)
    if class_in_scope:
        # Create sum node with class as cluster
        clu_ind = np.asarray(data[:, int(learnspn.classcol)].ravel(), dtype=np.int64)
    else:
        if learnspn.nclusters >= data.shape[0]:
            clu_ind = np.arange(data.shape[0], dtype=np.int64)
        else:
            scope_clu = scope
            clu_ind = cluster(data[:, scope_clu],
                              learnspn.nclusters, learnspn.ncat)
    j = 0
    # Once the clusters are defined, build the children using that
    # partition of the data.
    if len(np.unique(clu_ind)) == 1:
        return None
    else:
        sumnode = SumNode(last_node, scope, m)
        for i in np.unique(clu_ind):
            members = np.where(clu_ind == i)[0]
            if len(members) > 0:
                add_node(learnspn, data[members, :],
                              scope=scope,
                              max_height=max_height-1,
                              last_node=sumnode)
    return sumnode


@njit
def add_leaf(learnspn, data, scope, last_node):
    """
        Add a leaf node to the network.

        Parameters
        ----------

        learnspn: LearnSPN object
            Holds the hyperparameters for LearnSPN
        data: numpy array
            Rows are instances and columns variables.
        scope: list
            Indices of the columns of the variables in the scope of a node.
        last_node: Node object
            The last node that was added to the network.

        Returns
        -------

        node: SumNode object
            If the data is categorical, returns a sum node with the
            corresponding leaf nodes (indicators) as children.
        node: GaussianLeaf object
            If the data is continuous, returns a gaussian node with mean
            and variance parameters learned from the given data.
    """
    m = data.shape[0]
    assert len(scope) == 1, "Univariate leaf should not have more than one variable in its scope."
    if learnspn.ncat[scope[0]] > 1:  # Categorical variable
        leaf = MultinomialLeaf(last_node, np.asarray(scope), data.shape[0])
        fit_multinomial(leaf, data, int(learnspn.ncat[scope[0]]))
        return leaf
    else:  # Continuous variable (assumed normally distributed).
        leaf = GaussianLeaf(last_node, np.asarray(scope), m)
        fit_gaussian(leaf, data, np.inf, -np.inf)
        return leaf
