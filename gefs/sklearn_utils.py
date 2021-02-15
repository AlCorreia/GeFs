import numpy as np
from sklearn.tree import _tree
from sklearn.ensemble._forest import (_generate_sample_indices,
                                      _generate_unsampled_indices,
                                      _get_n_samples_bootstrap)

from .learning import LearnSPN, fit
from .nodes import (SumNode, ProdNode, Leaf, GaussianLeaf, MultinomialLeaf,
                   UniformLeaf, fit_multinomial, fit_multinomial_with_counts,
                   fit_gaussian)
from .pc import PC
from .utils import bincount


def calc_inbag(n_samples, rf):
    """
        Recovers samples used to create trees in scikit-learn RandomForest objects.

        See https://github.com/scikit-learn-contrib/forest-confidence-interval

        Parameters
        ----------
        n_samples : int
            The number of samples used to fit the scikit-learn RandomForest object.
        forest : RandomForest
            Regressor or Classifier object that is already fit by scikit-learn.

        Returns
        -------
        sample_idx: list
            The indices of the samples used to train each tree.
    """

    assert rf.bootstrap == True, "Forest was not trained with bootstrapping."

    n_trees = rf.n_estimators
    sample_idx = []
    n_samples_bootstrap = _get_n_samples_bootstrap(
        n_samples, rf.max_samples
    )

    for t_idx in range(n_trees):
        sample_idx.append(
            _generate_sample_indices(rf.estimators_[t_idx].random_state,
                                     n_samples, n_samples_bootstrap))
    return sample_idx


def calc_outofbag(n_samples, rf):
    """
        Recovers samples used to create trees in scikit-learn RandomForest objects.

        See https://github.com/scikit-learn-contrib/forest-confidence-interval

        Parameters
        ----------
        n_samples : int
            The number of samples used to fit the scikit-learn RandomForest object.
        forest : RandomForest
            Regressor or Classifier object that is already fit by scikit-learn.

        Returns
        -------
        sample_idx: list
            The indices of the samples used to train each tree.
    """

    assert rf.bootstrap == True, "Forest was not trained with bootstrapping."

    n_trees = rf.n_estimators
    sample_idx = []
    n_samples_bootstrap = _get_n_samples_bootstrap(
        n_samples, rf.max_samples
    )

    for t_idx in range(n_trees):
        sample_idx.append(
            _generate_unsampled_indices(rf.estimators_[t_idx].random_state,
                                        n_samples, n_samples_bootstrap))
    return sample_idx


def tree2pc_sklearn(tree, X, y, ncat, learnspn, max_height=100000,
                    thr=0.01, minstd=1, smoothing=1e-6, return_pc=False):
    """
        Parses a sklearn DecisionTreeClassifier to a Generative Decision Tree.
        Note that X, y do not need to match the data used to train the
        decision tree exactly. However, if they do not match you might get
        branches of the tree with no data, and hence poor models of the
        distribution at the leaves.

        Parameters
        ----------
        tree: DecisionTreeClassifier
        X: numpy array
            Explanatory variables.
        y: numpy array
            Target variable.
        ncat: numpy array (int64)
            The number of categories for each variable. 1 for continuous variables.
        learnsnp: int
            The number of samples (at a given leaf) required to run LearnSPN.
            Set to infinity by default, so as not to run LearnSPN anywhere.
        max_height: int
            Maximum height (depth) of the LearnSPN models at the leaves.
        thr: float
            p-value threshold for independence tests in product nodes.
        return_pc: boolean
            If True returns a PC object, if False returns a Node object (root).
        minstd: float
            The minimum standard deviation of gaussian leaves.
        smoothing: float
            Additive smoothing (Laplace smoothing) for categorical data.

    https://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree
    """

    scope = np.array([i for i in range(X.shape[1]+1)]).astype(int)
    data = np.concatenate([X, np.expand_dims(y, axis=1)], axis=1)
    lp = np.sum(np.where(ncat==1, 0, ncat)) * smoothing # LaPlace counts
    classcol = len(ncat)-1

    # Recursively parse decision tree nodes to PC nodes.
    def recurse(node, node_ind, depth, data, upper, lower):
        value = tree_.value[node_ind][0]
        counts = np.bincount(data[:, -1].astype(int), minlength=int(ncat[-1]))
        # If split node
        if tree_.feature[node_ind] != _tree.TREE_UNDEFINED:
            split_var = feature_name[node_ind]
            split_value = np.array([tree_.threshold[node_ind]], dtype=np.float64)
            sumnode = SumNode(scope=scope, n=data.shape[0]+lp)
            if node is not None:
                node.add_child(sumnode)
            # Parse left node <=
            upper1 = upper.copy()
            lower1 = lower.copy()
            upper1[split_var] = min(split_value, upper1[split_var])
            split1 = data[np.where(data[:, split_var] <= split_value)]
            p1 = ProdNode(scope=scope, n=split1.shape[0]+lp)
            sumnode.add_child(p1)
            ind1 = Leaf(scope=np.array([split_var]), n=split1.shape[0]+lp, value=split_value, comparison=3)  # Comparison <=
            p1.add_child(ind1)
            recurse(p1, tree_.children_left[node_ind], depth + 1, split1.copy(), upper1, lower1)
            # Parse right node >
            upper2 = upper.copy()
            lower2 = lower.copy()
            lower2[split_var] = max(split_value, lower2[split_var])
            split2 = data[np.where(data[:, split_var] > split_value)]
            p2 = ProdNode(scope=scope, n=split2.shape[0]+lp)
            sumnode.add_child(p2)
            ind2 = Leaf(scope=np.array([split_var]), n=split2.shape[0]+lp, value=split_value, comparison=4)  # Comparison >
            p2.add_child(ind2)
            recurse(p2, tree_.children_right[node_ind], depth + 1, split2.copy(), upper2, lower2)
            return sumnode
        # Leaf node
        else:
            assert node is not None, "Tree has no splits."
            if data.shape[0] >= learnspn:
                learner = LearnSPN(ncat, thr, 2, max_height, None)
                fit(learner, data, node)
            else:
                for var in scope:
                    if ncat[var] > 1:  # Categorical variable
                        leaf = MultinomialLeaf(scope=np.array([var]), n=data.shape[0]+lp)
                        node.add_child(leaf)
                        fit_multinomial(leaf, data, int(ncat[var]), smoothing)
                    else:  # Continuous variable
                        leaf = GaussianLeaf(scope=np.array([var]), n=data.shape[0]+lp)
                        node.add_child(leaf)
                        fit_gaussian(leaf, data, upper[var], lower[var], minstd)
                return None

    upper = ncat.copy().astype(float)
    upper[upper == 1] = np.Inf
    lower = ncat.copy().astype(float)
    lower[ncat == 1] = -np.Inf

    feature_names = [i for i in range(X.shape[1])]
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    root = recurse(None, 0, 1, data, upper, lower)
    if return_pc:
        pc = PC(ncat)
        pc.root = root
        return pc
    return root


def rf2pc(rf, X_train, y_train, ncat, learnspn=np.Inf, max_height=10000,
          thr=0.01, minstd=1, smoothing=1e-6):
    """
        Parses a sklearn RandomForestClassifier to a Generative Forest.
        Note that X, y do not need to match the data used to train the
        decision tree exactly. However, if they do not match you might get
        branches of the tree with no data, and hence poor models of the
        distribution at the leaves.

        Parameters
        ----------
        tree: DecisionTreeClassifier
        X: numpy array
            Explanatory variables.
        y: numpy array
            Target variable.
        ncat: numpy array (int64)
            The number of categories for each variable. 1 for continuous variables.
        learnsnp: int
            The number of samples (at a given leaf) required to run LearnSPN.
            Set to infinity by default, so as not to run LearnSPN anywhere.
        max_height: int
            Maximum height (depth) of the LearnSPN models at the leaves.
        thr: float
            p-value threshold for independence tests in product nodes.
        minstd: float
            The minimum standard deviation of gaussian leaves.
        smoothing: float
            Additive smoothing (Laplace smoothing) for categorical data.
    """
    scope = np.arange(len(ncat)).astype(int)
    lp = np.sum(np.where(ncat==1, 0, ncat)) * smoothing # LaPlace counts
    classcol = len(ncat)-1
    data = np.concatenate([X_train, np.expand_dims(y_train, axis=1)], axis=1)
    sample_idx = calc_inbag(X_train.shape[0], rf)
    pc = PC(ncat)
    pc.root = SumNode(scope=scope, n=1)
    for i, tree in enumerate(rf.estimators_):
        X_tree = X_train[sample_idx[i], :]
        y_tree = y_train[sample_idx[i]]
        si = tree2pc_sklearn(tree, X_tree, y_tree, ncat, learnspn, max_height,
                             thr, minstd, smoothing, return_pc=False)
        pc.root.add_child(si)
    return pc
