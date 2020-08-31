
# standard
import os

# packages
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class DataSet():
    def __init__(self, dataset_str=''):
        cwd = os.getcwd()

        # emg dataset is .npy file format, data is already standardized
        self.input = np.load(f'{cwd}/dataset/extracted/extracted_features.npy')

        # features are in every column except the last one
        self.data = self.input[:, :-1]

        # labels are in the last column
        self.target = self.input[:, -1]      
