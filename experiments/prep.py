import numpy as np
import pandas as pd


# Auxiliary functions
def get_dummies(data):
    data = data.copy()
    if isinstance(data, pd.Series):
        data = pd.factorize(data)[0]
        return data
    for col in data.columns:
        data.loc[:, col] = pd.factorize(data[col])[0]
    return data


def learncats(data, classcol=None, continuous_ids=[]):
    """
        Learns the number of categories in each variable and standardizes the data.

        Parameters
        ----------
        data: numpy n x m
            Numpy array comprising n realisations (instances) of m variables.
        classcol: int
            The column index of the class variables (if any).
        continuous_ids: list of ints
            List containing the indices of known continuous variables. Useful for
            discrete data like age, which is better modeled as continuous.

        Returns
        -------
        ncat: numpy m
            The number of categories of each variable. One if the variable is
            continuous.
    """
    data = data.copy()
    ncat = np.ones(data.shape[1])
    if not classcol:
        classcol = data.shape[1]-1
    for i in range(data.shape[1]):
        if i != classcol and (i in continuous_ids or is_continuous(data[:, i])):
            continue
        else:
            data[:, i] = data[:, i].astype(int)
            ncat[i] = max(data[:, i]) + 1
    return ncat


def get_stats(data, ncat=None):
    """
        Compute univariate statistics for continuous variables.

        Parameters
        ----------
        data: numpy n x m
            Numpy array comprising n realisations (instances) of m variables.

        Returns
        -------
        data: numpy n x m
            The normalized data.
        maxv, minv: numpy m
            The maximum and minimum values of each variable. One and zero, resp.
            if the variable is categorical.
        mean, std: numpy m
            The mean and standard deviation of the variable. Zero and one, resp.
            if the variable is categorical.

    """
    data = data.copy()
    maxv = np.ones(data.shape[1])
    minv = np.zeros(data.shape[1])
    mean = np.zeros(data.shape[1])
    std = np.zeros(data.shape[1])
    if ncat is not None:
        for i in range(data.shape[1]):
            if ncat[i] == 1:
                maxv[i] = np.max(data[:, i])
                minv[i] = np.min(data[:, i])
                mean[i] = np.mean(data[:, i])
                std[i] = np.std(data[:, i])
                assert maxv[i] != minv[i], 'Cannot have constant continuous variable in the data'
                data[:, i] = (data[:, i] - minv[i])/(maxv[i] - minv[i])
    else:
        for i in range(data.shape[1]):
            if is_continuous(data[:, i]):
                maxv[i] = np.max(data[:, i])
                minv[i] = np.min(data[:, i])
                mean[i] = np.mean(data[:, i])
                std[i] = np.std(data[:, i])
                assert maxv[i] != minv[i], 'Cannot have constant continuous variable in the data'
                data[:, i] = (data[:, i] - minv[i])/(maxv[i] - minv[i])
    return data, maxv, minv, mean, std


def normalize_data(data, maxv, minv):
    """
        Normalizes the data given the maximum and minimum values of each variable.

        Parameters
        ----------
        data: numpy n x m
            Numpy array comprising n realisations (instances) of m variables.
        maxv, minv: numpy m
            The maximum and minimum values of each variable. One and zero, resp.
            if the variable is categorical.

        Returns
        -------
        data: numpy n x m
            The normalized data.
    """
    data = data.copy()
    for v in range(data.shape[1]):
        if maxv[v] != minv[v]:
            data[:, v] = (data[:, v] - minv[v])/(maxv[v] - minv[v])
    return data


def standardize_data(data, mean, std):
    """
        Standardizes the data given the mean and standard deviations values of
        each variable.

        Parameters
        ----------
        data: numpy n x m
            Numpy array comprising n realisations (instances) of m variables.
        mean, std: numpy m
            The mean and standard deviation of the variable. Zero and one, resp.
            if the variable is categorical.

        Returns
        -------
        data: numpy n x m
            The standardized data.
    """
    data = data.copy()
    for v in range(data.shape[1]):
        if std[v] > 0:
            data[:, v] = (data[:, v] - mean[v])/(std[v])
            #  Clip values more than 6 standard deviations from the mean
            data[:, v] = np.clip(data[:, v], -6, 6)
    return data


def is_continuous(data):
    """
        Returns true if data was sampled from a continuous variables, and false
        Otherwise.

        Parameters
        ----------
        data: numpy
            One dimensional array containing the values of one variable.
    """
    observed = data[~np.isnan(data)]  # not consider missing values for this.
    rules = [np.min(observed) < 0,
             np.sum((observed) != np.round(observed)) > 0,
             len(np.unique(observed)) > min(30, len(observed)/3)]
    if any(rules):
        return True
    else:
        return False


def train_test_split(data, ncat, train_ratio=0.7, prep='std'):
    assert train_ratio >= 0
    assert train_ratio <= 1
    shuffle = np.random.choice(range(data.shape[0]), data.shape[0], replace=False)
    data_train = data[shuffle[:int(train_ratio*data.shape[0])], :]
    data_test = data[shuffle[int(train_ratio*data.shape[0]):], :]
    if prep=='norm':
        data_train, maxv, minv, _, _, = get_stats(data_train, ncat)
        data_test = normalize_data(data_test, maxv, minv)
    elif prep=='std':
        _, maxv, minv, mean, std = get_stats(data_train, ncat)
        data_train = standardize_data(data_train, mean, std)
        data_test = standardize_data(data_test, mean, std)

    X_train, y_train = data_train[:, :-1], data_train[:, -1]
    X_test, y_test = data_test[:, :-1], data_test[:, -1]

    return X_train, X_test, y_train, y_test, data_train, data_test


# Preprocessing functions
def adult(data):
    cat_cols = ['workclass', 'education', 'education-num', 'marital-status', 'occupation',
               'relationship', 'race', 'sex', 'native-country', 'y']
    cont_cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'capital-gain',
                'capital-loss', 'hours-per-week']
    data.loc[:, cat_cols] = get_dummies(data[cat_cols])
    ncat = learncats(data.values, classcol=-1, continuous_ids=[data.columns.get_loc(c) for c in cont_cols])
    return data.values.astype(float), ncat


def australia(data):
    cat_cols = ['A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13', 'class']
    cont_cols = ['A2', 'A3', 'A8', 'A11', 'A14', 'A15']
    data.loc[:, cat_cols] = get_dummies(data[cat_cols])
    data = data.replace('?', np.nan)
    ncat = learncats(data.values.astype(float), classcol=-1, continuous_ids=[data.columns.get_loc(c) for c in cont_cols])
    return data.values.astype(float), ncat


def bank(data):
    cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan',
       'contact', 'month', 'day_of_week', 'poutcome', 'y']
    cont_cols = ['age', 'duration', 'campaign', 'previous', 'emp.var.rate',
                'cons.price.idx','cons.conf.idx', 'euribor3m', 'nr.employed']
    data.loc[:, cat_cols] = get_dummies(data[cat_cols])
    data.loc[:, 'pdays'] = np.where(data['pdays']==999, 0, 1)
    ncat = learncats(data.values, classcol=-1, continuous_ids=[data.columns.get_loc(c) for c in cont_cols])
    return data.values.astype(float), ncat


def credit(data):
    cat_cols = ['SEX', 'EDUCATION', 'MARRIAGE', 'default payment next month']
    cont_cols = ['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2',
       'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    data.loc[:, cat_cols] = get_dummies(data[cat_cols])
    ncat = learncats(data.values, classcol=-1, continuous_ids=[data.columns.get_loc(c) for c in cont_cols])
    return data.values.astype(float), ncat


def electricity(data):
    cat_cols = ['day', 'class']
    cont_cols = ['date', 'period', 'nswprice', 'nswdemand', 'vicprice',
       'vicdemand', 'transfer']
    data.loc[:, cat_cols] = get_dummies(data[cat_cols])
    ncat = learncats(data.values, classcol=-1, continuous_ids=[data.columns.get_loc(c) for c in cont_cols])
    return data.values.astype(float), ncat


def segment(data):
    data = data.drop(columns=['region.centroid.col', 'region.pixel.count'])
    cat_cols = ['short.line.density.5', 'short.line.density.2', 'class']
    cont_cols = ['region.centroid.row', 'vedge.mean', 'vegde.sd', 'hedge.mean', 'hedge.sd',
                 'intensity.mean', 'rawred.mean', 'rawblue.mean', 'rawgreen.mean', 'exred.mean', 'exblue.mean' ,
                 'exgreen.mean', 'value.mean', 'saturation.mean', 'hue.mean']
    data.loc[:, cat_cols] = get_dummies(data[cat_cols])
    ncat = learncats(data.values, classcol=-1, continuous_ids=[data.columns.get_loc(c) for c in cont_cols])
    return data.values.astype(float), ncat


def german(data):
    cat_cols = [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19, 20]
    cont_cols = [1, 4, 7, 10, 12, 15, 17]
    data.iloc[:, cat_cols] = get_dummies(data[cat_cols])
    ncat = learncats(data.values, classcol=-1, continuous_ids=cont_cols)
    return data.values.astype(float), ncat


def vowel(data):
    cat_cols = ['Speaker_Number', 'Sex', 'Class']
    data.loc[:, cat_cols] = get_dummies(data[cat_cols])
    ncat = learncats(data.values, classcol=data.shape[1]-1)
    return data.values.astype(float), ncat


def cmc(data):
    cat_cols = ['Wifes_education', 'Husbands_education', 'Wifes_religion', 'Wifes_now_working%3F',
            'Husbands_occupation', 'Standard-of-living_index', 'Media_exposure', 'Contraceptive_method_used']
    cont_cols = ['Wifes_age', 'Number_of_children_ever_born']
    data.loc[:, cat_cols] = get_dummies(data[cat_cols])
    ncat = learncats(data.values, classcol=data.shape[1]-1)
    return data.values.astype(float), ncat


def get_data(name):
    if 'wine' in name:
        data_red = pd.read_csv('../data/winequality_red.csv')
        data_white = pd.read_csv('../data/winequality_white.csv')
        data = pd.concat([data_red, data_white]).values
        data[:, -1] = np.where(data[:, -1] <= 6, 0, 1)
        ncat = learncats(data, classcol=data.shape[1]-1)
    elif 'bank' in name:
        data = pd.read_csv('../data/bank-additional-full.csv', sep=';')
        data, ncat = bank(data)
    elif 'segment' in name:
        data = pd.read_csv('../data/segment.csv')
        data, ncat = segment(data)
    elif 'german' in name:
        data = pd.read_csv('../data/german.csv', sep=' ', header=None)
        data, ncat = german(data)
    elif 'vehicle' in name:
        data = pd.read_csv('../data/vehicle.csv')
        data['Class'] = get_dummies(data['Class'])
        ncat = np.ones(data.shape[1])
        ncat[-1] = len(np.unique(data['Class']))
        data = data.values.astype(float)
    elif 'vowel' in name:
        data = pd.read_csv('../data/vowel.csv')
        data, ncat = vowel(data)
    elif 'authent' in name:
        data = pd.read_csv('../data/authent.csv')
        data['Class'] = get_dummies(data['Class'])
        ncat = learncats(data.values).astype(int)
        data = data.values.astype(float)
    elif 'diabetes' in name:
        data = pd.read_csv('../data/diabetes.csv')
        data['class'] = get_dummies(data['class'])
        ncat = learncats(data.values,
                         continuous_ids=[0] # Force first variable to be continuous
                         ).astype(int)
        data = data.values.astype(float)
    elif 'cmc' in name:
        data = pd.read_csv('../data/cmc.csv')
        data, ncat = cmc(data)
    elif 'electricity' in name:
        data = pd.read_csv('../data/electricity.csv')
        data, ncat = electricity(data)
    elif 'gesture' in name:
        data = pd.read_csv('../data/gesture.csv')
        data['Phase'] = get_dummies(data['Phase'])
        data = data.values.astype(float)
        ncat = np.ones(data.shape[1])
        ncat[-1] = 5
    elif 'breast' in name:
        data = pd.read_csv('../data/wdbc.csv')
        data['Class'] = get_dummies(data['Class'])
        data = data.values.astype(float)
        ncat = np.ones(data.shape[1])
        ncat[-1] = 2
    elif 'krvskp' in name:
        data = pd.read_csv('../data/kr-vs-kp.csv')
        data = get_dummies(data)
        ncat = learncats(data.values)
        data = data.values.astype(float)
    elif 'dna' in name:
        data = pd.read_csv('../data/dna.csv')
        data = get_dummies(data).values.astype(float)
        ncat = learncats(data)
    elif 'robot' in name:
        data = pd.read_csv('../data/robot.csv')
        data['Class'] = get_dummies(data['Class'])
        data = data.values.astype(float)
        ncat = learncats(data)
    elif 'mice' in name:
        data = pd.read_csv('../data/miceprotein.csv')
        data['class'] = get_dummies(data['class'])
        data = data.replace('?', np.nan)
        data = data.drop(['MouseID', 'Genotype', 'Treatment', 'Behavior'], axis=1)
        data = data.values.astype(float)
        ncat = learncats(data)
    elif 'dresses' in name:
        data = pd.read_csv('../data/dresses.csv')
        data = data.replace('?', np.nan)
        data = get_dummies(data)
        data = data.values.astype(float)
        data[data < 0] = np.nan
        ncat = learncats(data)
    elif 'texture' in name:
        data = pd.read_csv('../data/texture.csv')
        data['Class'] = get_dummies(data['Class'])
        data = data.values.astype(float)
        ncat = np.ones(data.shape[1])
        ncat[-1] = 11
    elif 'splice' in name:
        data = pd.read_csv('../data/splice.csv')
        data = data.drop('Instance_name', axis=1)
        data = get_dummies(data).values.astype(float)
        ncat = learncats(data)
    elif 'jungle' in name:
        data = pd.read_csv('../data/jungle.csv')
        data = get_dummies(data)
        data = data.values.astype(float)
        ncat = learncats(data)
    elif 'phishing' in name:
        data = pd.read_csv('../data/phishing.csv')
        data = get_dummies(data)
        data = data.values.astype(float)
        ncat = learncats(data)
    elif 'fashion' in name:
        data = pd.read_csv('../data/fashion.csv')
        data = data.values.astype(np.float64)
        ncat = np.ones(data.shape[1]).astype(np.int64)
        ncat[-1] = 10
    elif 'mnist' in name:
        data = pd.read_csv('../data/mnist.csv')
        data = data.values.astype(np.float64)
        ncat = np.ones(data.shape[1]).astype(np.int64)
        ncat[-1] = 10
    else:
        print("Sorry, dataset {} is not available.".format(name))
        print("You have to provide the data and run the appropriate pre-processing steps yourself.")
        raise ValueError

    return data, ncat
