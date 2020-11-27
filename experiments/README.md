# Experiments

This is the code that reproduces the experiments found in the papers [Joints in Random Forests](https://arxiv.org/abs/2006.14937) and [Towards Robust Classification with Deep Generative Forests](https://arxiv.org/abs/2007.05721).


## Datasets and Requirements

The data is assumed to be available in `csv` format at the `data` folder. These csv files can be directly obtained from the [OpenML-CC18 web-page](https://www.openml.org/s/99/data), but for ease of reproduction most datasets (excluding Mnist and Fashion-Mnist) used in the papers are already included in this repository.

These experiments require some extra packages, namely `jupyter`, `pandas`, `matplotlib` and `seaborn`. You can check a list at `requirements.txt` and install them all with `pip install -r requirements.txt`.

## Missing Variable Experiments

`run_missing.py` is the script that runs all the missing values experiments in the 'Joints in Random Forests' paper. For each dataset, every method is run at different percentages of missing values at test time (from 0 to 90%).

As an example, the following command will reproduce the results on the Authentication dataset.

`python3 run_missing.py -d authent`

The script will save two csv files to a folder called `missing`, one includes the mean accuracy values for each model and each percent of missing value, and the other the corresponding confidence intervals (95%).

### Parameters

* `--dataset` or `-d`: Name of the dataset to be run (str);
* `--runs` or `-r`: Runs (a list, default to 0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
* `--n_folds` or `-f`: Number of folds (int, default 5);
* `--lspn` or `-l`: Whether to learn GeF(LearnSPN) by fitting learnSPNs at the leaves (boolean, default True).

These parameters above are the only ones needed to reproduce the results, as the others kept to their default values. Set the dataset name accordingly, and for datasets larger than 10K samples, `-r 0, -f 10`. If you have installed rvlib, and want to run experiments with GeF(LSPN), then set `--lspn` to True.

* `--n_estimators` or `-e`: Number of estimators (int, default 100);
* `--msl` or `-m`: Minimum number of samples at the leaves (int, default 1);

Running LearnSPN requires installing the RMath package from R, see above. Note that any value in 'yes', 'true', 't', 'y', '1' evaluates to True, any value in 'no', 'false', 'f', 'n', '0' evaluates to False).

## Robustness and Outlier Detection Experiments
The experiments in the 'Towards Robust Classification with Deep Generative Forests' paper can be reproduced in the `Robustness-Experiments` and `Outlier-Detection-Experiments` jupyter notebooks.

## Refereces

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
