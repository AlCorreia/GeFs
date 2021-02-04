# GeFs - Generative Forests in Python

[Generative Forests](https://arxiv.org/abs/2006.14937) are a class of Probabilistic Circuits (PCs) that subsumes Random Forests. They maintain the discriminative structure learning and overall predictive performance of Random Forests, while extending them to a full generative model over the joint p(X, y). This enhances Random Forests with pricipled methods for

- Outlier detection
- Robust classification
- Inference with missing values

This repository reproduces the experiments provided in the papers [Joints in Random Forests](https://arxiv.org/abs/2006.14937) and [Towards Robust Classification with Deep Generative Forests](https://arxiv.org/abs/2007.05721). See the `experiments` folder for the experimental set-up.

## Installation

To install GeFs it suffices to run `pip install .` at the root directory of this repository. This project was developed for Python 3 and mostly likely will not run in Python 2.

## Usage

We learn the structure of a GeF as in a regular Random Forest. For ease of use, we keep similar signatures to the scikit-learn implementation. Once the structure is learned, we convert it to a GeF with the `topc()` method, as in the following snippet.

```
from gefs import RandomForest
from prep import get_data, train_test_split

data, ncat = get_data(name)  # Preprocess the data
# ncat is the number of categories of each variable in the data
X_train, X_test, y_train, y_test, data_train, data_test = train_test_split(data, ncat)
rf = RandomForest(n_estimators=30, ncat=ncat)  # Train a Random Forest
rf.fit(X_train, y_train)
gef = rf.topc()  # Convert to a GeF
```

Currently `topc()` fits a GeF by extending the leaves either with a fully-factorised distribution (default) or with another PC via LearnSPN. This behaviour is defined by the `learnspn` parameter that gives the minimum number of samples to run LearnSPN. For instance, `rf.topc(learnspn=30)` would run LearnSPN for every leaf in the Random Forest with more than 30 samples.

Classification is performed either by averaging the prediction of each tree (`classify_avg` method) or by defining a mixture over them (`classify` method). 

```
y_pred_avg = gef.classify_avg(X_test, classcol=data.shape[1]-1)
y_pred_mixture = gef.classify(X_test, classcol=data.shape[1]-1)
```

Note that given GeFs are generative models, we could predict any categorical variable in the data, not just the class variable. Therefore, we need to pass the index of the variable we want to predict to the `classcol` parameter. In the datasets provided here, the class variable is always the last one, hence `data.shape[1]-1`.

### Computing Robustness Values
Robustness values can be computed with the `compute_rob_class` function.
```
from gefs import compute_rob_class
pred, rob = compute_rob_class(gef.root, X_test, data.shape[1]-1, int(ncat[-1]))
```
The function returns the prediction and the robustness value of each instance in `X_test`. Note that `compute_rob_class` requires the index and the number of categories of the target variable as third and fourth parameters. 

### Computing log-densities
The log-density of each sample can be computed with the `log_likelihood` function.
```
logs = gef.log_likelihood(data_test)
```
Here if `data_test` is a matrix of n observations and m variables, `logs` will be an array of size n, containing `log(p(x))` for each observarion `x` in `data_test`.

## References

If you find GeFs useful please consider citing us in your work

```
@article{correia2020joints,
  title={Joints in Random Forests},
  author={Correia, A. H. C. and Peharz, Robert and de Campos, C. P.},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}

@article{correia2020towards,
  title={Towards Robust Classification with Deep Generative Forests},
  author={Correia, A. H. C. and Peharz, R. and de Campos, C. P.},
  journal={ICML 2020 Workshop on Uncertainty and Robustness in Deep Learning},
  year={2020}
}
```
