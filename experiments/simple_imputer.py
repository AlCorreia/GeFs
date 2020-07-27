import numpy as np
import scipy.stats as stats


class SimpleImputer:
    """ Simple mean/most frequent imputation. """
    def __init__(self, ncat, method='mean'):
        self.ncat = ncat
        assert method in ['mean', 'mode'], "%s is not supported as imputation method." %method
        self.method = method

    def fit(self, data):
        assert data.shape[1] == len(self.ncat), "Data does not match the predefined number of variables."
        self.data = data
        self.values = np.zeros(data.shape[1])
        for j in range(data.shape[1]):
            # Filter missing values first
            ref_data = self.data[~np.isnan(self.data[:, j]), j]
            if self.ncat[j] == 1:
                if self.method == 'mode':
                    self.values[j] = stats.mode(ref_data)[0]
                elif self.method == 'mean':
                    self.values[j] = np.mean(ref_data)
            else:
                self.values[j] = stats.mode(ref_data)[0]
        return self

    def transform(self, data):
        data = data.copy()
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)
        missing_coordinates = np.where(np.isnan(data))
        for j in range(data.shape[1]):
            ind = missing_coordinates[0][missing_coordinates[1]==j]
            data[ind, j] = self.values[j]
        return data
