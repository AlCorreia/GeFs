import numpy as np

from .learning import LearnSPN, fit
from .nodes import SumNode, ProdNode, Leaf, GaussianLeaf, eval_root, eval_root_children, eval_root_class, delete
from .utils import logsumexp3


class PC:
    """
        Class that defines and evaluates an PC.

        Attributes
        ----------

        root: Node object
            The root node of the PC.
        ncat: numpy
            The number of categories of each variable. One for continuous variables.
        learner: object
            Defines the learning method of the PC.
            Currently, only LearnSPN (Gens and Domingos, 2013).
    """

    def __init__(self, ncat=None):
        self.ncat = ncat
        self.root = None
        self.maxv = None
        self.minv = None
        self.n_nodes = 0


    def learnspn(self, data, ncat=None, thr=0.001, nclusters=2, max_height=1000000, classcol=None):
        if ncat is not None:
            self.ncat = ncat
        assert self.ncat is not None, "You must provide `ncat`, the number of categories of each class."
        learner = LearnSPN(self.ncat, thr, nclusters, max_height, classcol)
        self.root = fit(learner, data)


    def set_topological_order(self):
        """
            Updates the ids of the nodes so that they match their topological
            order.
        """
        def get_topological_order(node, order=[]):
            if order == []:
                node.id = len(order)
                order.append(node)
            for child in node.children:
                child.id = len(order)
                order.append(child)
            for child in node.children:
                get_topological_order(child, order)
            return order
        self.order = get_topological_order(self.root, [])
        self.n_nodes = len(self.order)

    def log_likelihood(self, data, avg=False):
        """
            Computes the log-likelihood of data.

            Parameters
            ----------

            data: numpy array
                Input data including the class variable.
                Missing values should be set to numpy.nan
        """
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)
        if avg:
            nchildren = self.root.nchildren
            logs_avg = np.empty(shape=(data.shape[0], nchildren))
            for i in range(nchildren):
                logs_avg[:, i] = eval_root(self.root.children[i], data)
            logs_avg = np.mean(logs_avg, axis=1)
            return logs_avg
        return eval_root(self.root, data)

    def likelihood(self, data):
        """
            Computes the likelihood of data.

            Parameters
            ----------

            data: numpy array
                Input data including the class variable.
                Missing values should be set to numpy.nan
        """
        ll = self.log_likelihood(data)
        return np.exp(ll)

    def classify(self, X, classcol, return_prob=False):
        """
            Classifies instances running proper PC inference, that is,
            argmax_y P(X, Y=y).

            Parameters
            ----------

            X: numpy array
                Input data not including the class variable.
                Missing values should be set to numpy.nan
            classcol: int
                The original index of the class variable.
            return_prob: boolean
                Whether to return the conditional probability of each class.
        """
        eps = 1e-6
        if X.ndim == 1:
            X = np.expand_dims(X, axis=0)
        nclass = int(self.ncat[classcol])
        joints = eval_root_class(self.root, X, classcol, nclass, naive=False)
        joints = logsumexp3(joints, axis=2)
        joints_minus_max = joints - np.max(joints, axis=1, keepdims=True)
        probs = np.where(np.exp(joints_minus_max) >= (np.log(eps) - np.log(nclass)), np.exp(joints_minus_max), 0)
        probs = probs/probs.sum(axis=1, keepdims=True)
        if return_prob:
            return np.argmax(probs, axis=1), probs
        return np.argmax(probs, axis=1)

    def classify_avg(self, X, classcol, return_prob=False, naive=False):
        """
            Classifies instances by taking the average of the conditional
            probabilities defined by each PC, that is,
            argmax_y sum_n P_n(Y=y|X)/n

            This is only makes sense if the PC was learned as an ensemble, where
            each model is the child of the root.

            Parameters
            ----------

            X: numpy array
                Input data not including the class variable.
                Missing values should be set to numpy.nan
            classcol: int
                The original index of the class variable.
            return_prob: boolean
                Whether to return the conditional probability of each class.
            naive: boolean
                Whether to treat missing values as suggested by  Friedman in 1975,
                that is, by taking the argmax over the counts of all pertinent
                cells.
            """
        eps = 1e-6
        if X.ndim == 1:
            X = np.expand_dims(X, axis=0)
        nclass = int(self.ncat[classcol])
        # joints = np.zeros((X.shape[0], nclass, self.root.nchildren))
        joints = eval_root_class(self.root, X, classcol, nclass, naive)
        if naive:
            counts = np.exp(joints).astype(int)  # int to filter out the smoothing
            conditional = counts/np.sum(counts, axis=1, keepdims=True)
        else:
            # Convert from log to probability space
            joints_minus_max = joints - np.max(joints, axis=1, keepdims=True)
            probs = np.where(np.exp(joints_minus_max) >= (np.log(eps) - np.log(nclass)), np.exp(joints_minus_max), 0)
            # Normalize to sum out X: we get P(Y|X) by dividing by P(X)
            conditional = probs/np.sum(probs, axis=1, keepdims=True)
        # Average over the trees
        agg = np.mean(conditional, axis=2)
        maxclass = np.argmax(agg, axis=1)
        if return_prob:
            return maxclass, agg
        return maxclass

    def classify_lspn(self, X, classcol, return_prob=False):
        """
            Classifies instances running proper PC inference, that is,
            argmax_y P(X, Y=y).

            Parameters
            ----------

            X: numpy array
                Input data not including the class variable.
                Missing values should be set to numpy.nan
            classcol: int
                The original index of the class variable.
            return_prob: boolean
                Whether to return the conditional probability of each class.
            naive: boolean
                Whether to treat missing values as suggested in Friedman1975,
                that is, by taking the argmax over the counts of all pertinent
                cells.
        """
        eps = 1e-6
        if X.ndim == 1:
            X = np.expand_dims(X, axis=0)
        nclass = int(self.ncat[classcol])
        maxclass = np.zeros(X.shape[0])-1
        maxlogpr = np.zeros(X.shape[0])-np.Inf
        joints = np.zeros((X.shape[0], nclass))
        for i in range(nclass):
            iclass = np.zeros((X.shape[0], 1)) + i
            Xi = np.append(X, iclass, axis=1)
            joints[:, i] = np.squeeze(eval_root(self.root, Xi))
        joints_minus_max = joints - np.max(joints, axis=1, keepdims=True)
        probs = np.where(np.exp(joints_minus_max) >= (np.log(eps) - np.log(nclass)), np.exp(joints_minus_max), 0)
        probs = probs/probs.sum(axis=1, keepdims=True)
        if return_prob:
            return np.argmax(probs, axis=1), probs
        return np.argmax(probs, axis=1)

    def classify_avg_lspn(self, X, classcol, return_prob=False):
        """
            Classifies instances by taking the average of the conditional
            probabilities defined by each tree, that is,
            argmax_y sum_n P_n(Y=y|X)/n
            This is only makes sense if the PC was learned as an ensemble, where
            each model is the child of the root.

            Parameters
            ----------

            X: numpy array
                Input data not including the class variable.
                Missing values should be set to numpy.nan
            classcol: int
                The original index of the class variable.
            return_prob: boolean
                Whether to return the conditional probability of each class.
        """
        eps = 1e-6
        nclass = int(self.ncat[classcol])
        joints = np.zeros((X.shape[0], self.root.nchildren, nclass))
        for i in range(nclass):
            iclass = np.zeros((X.shape[0], 1)) + i
            Xi = np.append(X, iclass, axis=1)
            joints[:, :, i] = eval_root_children(self.root, Xi)
        joints_minus_max = joints - np.max(joints, axis=2, keepdims=True)
        probs = np.where(np.exp(joints_minus_max) >= (np.log(eps) - np.log(nclass)), np.exp(joints_minus_max), 0)
        normalized = probs/np.sum(probs, axis=2, keepdims=True)
        agg = np.mean(normalized, axis=1)
        maxclass = np.argmax(agg, axis=1)
        if return_prob:
            return maxclass, agg
        return maxclass

    def clear(self):
        """ Deletes the structure of the PC. """
        if self.root is not None:
            self.root.remove_children(*self.root.children)
            self.root = None

    def get_node(self, id):
        """ Fetchs node by its id. """
        queue = [self.root]
        while queue != []:
            node = queue.pop(0)
            if node.id == id:
                return node
            if node.type not in ['L', 'G', 'U']:
                queue.extend(node.children)
        print("Node %d is not part of the network.", id)

    def get_node_of_type(self, type):
        """ Fetchs all nodes of a given type. """
        queue = [self.root]
        res = []
        while queue != []:
            node = queue.pop(0)
            if node.type == type:
                res.append(node)
            if node.type not in ['L', 'G', 'U']:
                queue.extend(node.children)
        return res

    def delete(self):
        """
            Calls the delete function of the root node, which in turn deletes
            the rest of the nodes in the PC. Given that nodes in an PC point
            to each other, they are always referenced and never automatically
            deleted by the Python interpreter.
        """
        delete(self.root)
