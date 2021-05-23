import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import clone
from scipy.stats import mode

class implementedBoosting(BaseEstimator, ClassifierMixin):
    """
    Implementacja booatingu

    Parameters
    ----------
     base_estimator : base_estimator (Default = None)
         Deklaracja klasyfikatora bazowego

     n_estimators : int (Default = 10)
         
    
    """

    def __init__(self, base_estimator = None, n_estimators = 10):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        if not hasattr(self, "ensemble_"):
            self.ensemble_ = []
            self.alphas_ = []

        self.classes_ = np.unique(y)
        self.X_, self.y_ = X, y        

        weights = np.ones(len(self.X_))
        
        for base_clf in range(self.n_estimators):
            model = clone(self.base_estimator).fit(self.X_, self.y_, weights)
            self.ensemble_.append(model)
            y_pred = model.predict(self.X_)

            errorArray = np.abs(y_pred - self.y_)            

            finalError = np.sum(errorArray)
            self.alphas_.append(self.calculate_alpha(finalError, len(self.X_)))
            weights += errorArray
       
        return self

    def calculate_alpha(self, err, setsize):
        alpha = np.log((1 - (err/setsize)) / ((err/setsize) + 0.00001))
        return alpha

    def predict(self, X):        
        y_pred = []
        fx = np.zeros(len(X))

        for Clf, a in zip(self.ensemble_,self.alphas_):
            y_pred = Clf.predict(X)
            fx = fx + a * ((y_pred - 0.5) * 2)
            
        fx = fx + 0.00000001
        y_pred = (np.sign(fx) + 1) / 2

        return y_pred

