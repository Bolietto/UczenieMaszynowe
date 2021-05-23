import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from RandomNumberGenerator import RandomNumberGenerator
from sklearn.base import clone

class RandomSubspace(BaseEstimator, ClassifierMixin):
    """
    Implementacja random subspace

    Parameters
    ----------
     base_estimator : base_estimator (Default = None)
         Deklaracja klasyfikatora bazowego

     n_estimators : int (Default = 10)
         Liczba klasyfikatorów do wygenerowania
     
     random_state : int (Default = None)
         Ziarno losowe
    
    """
    def __init__(self, base_estimator = None, n_estimators = 10, random_state = None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.X_, self.y_ = X, y
        if not hasattr(self, "ensemble_"):
            self.ensemble_ = []
        
        RandomGenerator = RandomNumberGenerator(self.random_state)
        
        for base_clf in range(self.n_estimators):
            feature_subspace_size = RandomGenerator.nextInt(1, self.X_.shape[1])
            Subspace = np.empty((self.X_.shape[0],0))
            
            for feature in range(feature_subspace_size):
                Subspace = np.append(Subspace, np.reshape(self.X_[:, RandomGenerator.nextInt(0, feature_subspace_size - 1) ], (self.X_.shape[0],1)), axis=1)
            
            
            model = clone(self.base_estimator).fit(Subspace, self.y_)
            self.ensemble_.append(model)
            
        return self
            
    def predict(self, X):
        y_pred = []      
        
        for x_query in X:
            votes = []
            
            for Clf in self.ensemble_:
                votes.append(Clf.predict(x_query.reshape(1, -1)))
                
            if np.sum(votes) / self.n_estimators > 0.5:
                y_pred.append(1)
                
            else:
                y_pred.append(0)
                
        
        return y_pred