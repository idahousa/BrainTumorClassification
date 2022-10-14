import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
class ZNorm:
    def __init__(self):
        self.scaler = StandardScaler()
    def train(self,i_features=None):
        assert isinstance(i_features,np.ndarray)
        assert len(i_features.shape)==2 #m-by-n matrix
        """fit() method is to compute the mean and std of input features and stores them in the scaler object"""
        self.scaler.fit(i_features)
    def predict(self,i_features=None):
        assert isinstance(i_features, np.ndarray)
        assert len(i_features.shape) == 2  # m-by-n matrix
        return self.scaler.transform(i_features)
    """Utilities for saving and loading model"""
    def save(self, i_file_path=None):
        assert isinstance(i_file_path, str)
        with open(i_file_path, 'wb') as file:
            pickle.dump(self, file)
    @staticmethod
    def load(i_file_path=None):
        assert isinstance(i_file_path, str)
        assert os.path.exists(i_file_path)
        with open(i_file_path, 'rb') as file:
            return pickle.load(file)
if __name__ == '__main__':
    print('This module is to implement the Z-Score normaliztion')
    print('Reference: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html')
    data = np.asarray([[0, 0], [0, 0], [1, 1], [1, 1]])
    scaler = ZNorm()
    scaler.train(i_features=data)
    preds = scaler.predict(i_features=data)
    print(preds)
"""=================================================================================================================="""