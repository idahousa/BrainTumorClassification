import pickle
import os.path
import numpy as np
from sklearn import svm
from BrainNet.libs.logs import Logs
from BrainNet.libs.ml.znorm import ZNorm
from sklearn.model_selection import GridSearchCV
from BrainNet.libs.ml.pcas import PrincialComponentAnalysis
class SupportVectorMachines:
    def __init__(self,
                 i_kernel_name = 'linear',
                 i_znorm_flag  = False,
                 i_pca_flag    = False,
                 i_pca_comps   = 10,
                 i_num_classes = 2):
        assert isinstance(i_kernel_name,str)
        assert i_kernel_name in ('linear','rbf','poly','sigmoid')
        assert isinstance(i_znorm_flag,bool)
        assert isinstance(i_pca_flag,bool)
        assert isinstance(i_pca_comps,int)
        assert isinstance(i_num_classes,int)
        assert i_num_classes>1
        if i_pca_flag:
            self.pca = PrincialComponentAnalysis(i_znorm_flag=i_znorm_flag,i_num_comps=i_pca_comps)
        else:
            self.pca = None
            if i_znorm_flag:
                self.znorm = ZNorm()
            else:
                self.znorm = None
        self.svm = None
        """=============================================================================================================
        - C: float, default=1.0. Regularization parameter. The strength of the regularization is inversely proportional to C. 
          Must be strictly positive. The penalty is a squared l2 penalty.
          C is high it will classify all the data points correctly, also there is a chance to overfit.
        - kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} or callable, default=’rbf’
          Specifies the kernel type to be used in the algorithm. If none is given, ‘rbf’ will be used.
          If a callable is given it is used to pre-compute the kernel matrix from data matrices;
          that matrix should be an array of shape (n_samples, n_samples).
        - degreeint, default=3
          Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
        - gamma{‘scale’, ‘auto’} or float, default=’scale’
          Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
          when gamma is higher, nearby points will have high influence; 
          low gamma means far away points also be considered to get the decision boundary.
        ============================================================================================================="""
        self.num_classes = i_num_classes
        self.kernel_name = i_kernel_name
        if i_kernel_name == 'linear':
            self.svm_params = {'kernel' : ['linear'],
                               'C'      : [0.1,1.0,10.0,100.0,1000.0]}
        elif i_kernel_name == 'rbf':
            self.svm_params = {'kernel' : ['rbf'],
                               'C'      : [0.1,1.0,10.0,100.0,1000.0],
                               'gamma'  : [1.0,0.1,0.01,0.001,0.0001]}
        elif i_kernel_name == 'poly':
            self.svm_params = {'kernel' : ['poly'],
                               'C'      : [0.1, 1.0, 10.0, 100.0, 1000.0],
                               'gamma'  : [1.0, 0.1, 0.01, 0.001, 0.0001],
                               'degree' : [1,3,5,10]}
        else:
            self.svm_params = {'kernel' : ['sigmoid'],
                               'C'      : [0.1, 1.0, 10.0, 100.0, 1000.0],
                               'gamma'  : [1.0, 0.1, 0.01, 0.001, 0.0001]}
        self.grid_search = GridSearchCV(estimator=svm.SVC( decision_function_shape='ovr'),param_grid=self.svm_params)
    """Train the grid-search to find the optimal kernel with parameters"""
    def train(self,i_features=None,i_labels=None):
        assert isinstance(i_features,np.ndarray)
        assert len(i_features.shape)==2
        assert isinstance(i_labels,np.ndarray)
        assert len(i_labels.shape)==1
        """Start training"""
        if self.svm is None:#Traind for the first time
            if self.pca is None:
                if self.znorm is None:
                    self.grid_search.fit(i_features,i_labels)
                else:
                    self.znorm.train(i_features=i_features)
                    self.grid_search.fit(self.znorm.predict(i_features=i_features),i_labels)
            else:
                """PCA feature reduction"""
                assert isinstance(self.pca,PrincialComponentAnalysis)
                self.pca.train(i_features=i_features)
                """Train the grid-search"""
                self.grid_search.fit(self.pca.predict(i_features=i_features),i_labels)
            self.svm = self.grid_search.best_estimator_
        else:
            print('SVM model already existed. Not train anymore!')
        return self
    """Predict the class label of each sample: Return [num_sample,1] matrix"""
    def predict(self,i_features=None):
        assert isinstance(i_features,(np.ndarray,list,tuple))
        if self.svm is None:
            raise Exception('Please train the model first!')
        else:
            pass
        if self.pca is None:
            if self.znorm is None:
                return self.svm.predict(i_features)
            else:
                return  self.svm.predict(self.znorm.predict(i_features=i_features))
        else:
            return self.svm.predict(self.pca.predict(i_features=i_features))
    """Return the matrix of prediction score of each sample for each class: Return [num_samples, num_classes] matrix"""
    def scoring(self,i_features=None):
        assert isinstance(i_features, (np.ndarray, list, tuple))
        if self.svm is None:
            raise Exception('Please train the model first!')
        else:
            #print(self.svm)
            pass
        if self.pca is None:
            if self.znorm is None:
                return self.svm.decision_function(i_features)
            else:
                return self.svm.decision_function(self.znorm.predict(i_features=i_features))
        else:
            return self.svm.decision_function(self.pca.predict(i_features=i_features))
    """Utilities for saving and loading model"""
    def save(self,i_file_path=None):
        assert isinstance(i_file_path,str)
        with open(i_file_path,'wb') as file:
            pickle.dump(self,file)
    @staticmethod
    def load(i_file_path=None):
        assert isinstance(i_file_path,str)
        assert os.path.exists(i_file_path)
        with open(i_file_path,'rb') as file:
             return pickle.load(file)
    def evaluation(self,i_db=None):
        assert isinstance(i_db, (list, tuple)) #In format( feature,label)
        confusion_matrix = np.zeros(shape=(self.num_classes, self.num_classes))
        for index, item in enumerate(i_db):
            features, label = item
            prediction = self.predict(i_features=features.reshape((1,-1)))[0]
            confusion_matrix[label, prediction] += 1
        Logs.log(i_str='- SVM kernel: {}'.format(self.kernel_name))
        Logs.log_matrix('- Confusion Matrix', i_matrix=confusion_matrix)
        Logs.log('- Accuracy: {:2.3f} (%)'.format(np.trace(confusion_matrix) * 100 / np.sum(confusion_matrix)))
        Logs.log('-' * 100)
if __name__ == '__main__':
    print('This module is to implement the support vector machine for classification problem')
    print('Reference: https://scikit-learn.org/stable/modules/svm.html')
    print('Reference: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html')
    print('Reference: https://towardsdatascience.com/gridsearchcv-for-beginners-db48a90114ee')
    from sklearn import datasets
    import scipy.special
    iris = datasets.load_iris()
    model = SupportVectorMachines(i_kernel_name='sigmoid',i_znorm_flag=True,i_pca_flag=True,i_pca_comps=3,i_num_classes=3)
    """
    x = [data for (index, data) in enumerate(iris.data) if iris.target[index]<=1]
    y = [data for data in iris.target if data<=1]
    x = np.array(x)
    y = np.array(y)
    print(x, y)
    model.train(i_features=x,i_labels=y)
    pred = model.scoring(x)
    print(pred)
    print(pred.shape,x.shape,y.shape)
    """
    model.train(i_features=iris.data,i_labels=iris.target)
    model.save('example.obj')
    x = model.load('example.obj')
    pred = model.scoring(iris.data)
    print(scipy.special.softmax(pred,axis=-1))
    print(x.predict(iris.data))
    print(iris.target)
    print(model.svm)
"""=================================================================================================================="""