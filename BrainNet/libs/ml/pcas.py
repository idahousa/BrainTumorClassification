import os
import pickle
import numpy as np
from BrainNet.libs.ml.znorm import ZNorm
from sklearn.decomposition import PCA
class PrincialComponentAnalysis:
    def __init__(self,i_num_comps=8,i_znorm_flag=False):
        assert isinstance(i_num_comps,int) #Number of components to be used (dimension of the low-dimension space)
        assert isinstance(i_znorm_flag,bool)#Whether we use z-score normalization on the input features or not.
        assert i_num_comps>0
        self.model = None
        self.dim   = i_num_comps
        if i_znorm_flag:
            self.znorm = ZNorm()
        else:
            self.znorm = None
    def train(self,i_features):
        assert isinstance(i_features, np.ndarray)
        assert len(i_features.shape) == 2  # m-by-n matrix
        if self.model is None:
            self.dim   = min(self.dim,i_features.shape[-1])
            self.model = PCA(n_components=self.dim)
            if self.znorm is None:
                self.model.fit(i_features)
            else:
                assert isinstance(self.znorm,ZNorm)
                self.znorm.train(i_features=i_features)
                self.model.fit(self.znorm.predict(i_features=i_features))
        else:
            print('Model already trained! It now can only used to transform features to low dimensional space!')
            return False
    """Get eigence values in each axis (projection direction)"""
    def get_eigence_values(self):
        if self.model is not None:
            assert isinstance(self.model,PCA)
            return self.model.singular_values_
        else:
            return False
    """Get ratio of variance in each project axis"""
    def get_variance_ratio(self):
        if self.model is not None:
            assert isinstance(self.model,PCA)
            return self.model.explained_variance_ratio_
        else:
            return False
    """Main function to transform data"""
    def predict(self,i_features):
        assert isinstance(i_features, np.ndarray)
        assert len(i_features.shape) == 2  # m-by-n matrix
        if self.znorm is None:
            return self.model.transform(i_features)
        else:
            return self.model.transform(self.znorm.predict(i_features=i_features))
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
    print('This module is to implement the pricipal component analysis')
    print('Reference: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA')
    data  = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    model = PrincialComponentAnalysis(i_num_comps=1,i_znorm_flag=True)
    model.train(i_features=data)
    print(model.get_eigence_values())
    print(model.get_variance_ratio())
    print(model.predict(i_features=data))
"""=================================================================================================================="""