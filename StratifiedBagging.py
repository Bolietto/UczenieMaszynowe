import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import clone

class StratifiedBagging(BaseEstimator, ClassifierMixin):
    """
    Implementacja StratifiedBagging

    Parameters
    ----------
     base_estimator : base_estimator (Default = None)
         Deklaracja klasyfikatora bazowego

     n_estimators : int (Default = 10)
         Liczba klasyfikatorów
     
     random_state : int (Default = None)
         Ziarno losowości
     
     subset_size : int (Default = 50)
         Ilośc wzorców w pojedyńczym bagu
         
    
    """
    def __init__(self, base_estimator=None, n_estimators=10, random_state = None, subset_size = 50):

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.subset_size = subset_size
        
        
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.X_, self.y_ = X, y
        if not hasattr(self, "ensemble_"):
            self.ensemble_ = []
        
        balance_ratio = np.sum(self.y_)/len(self.y_)
        dataset = self.X_
        dataset = np.append(dataset, self.y_, axis=1)
        
        for base_clf in range(self.n_estimators):
            subset = np.empty((0,self.X_.shape[1]+1))
        
            for sample in range(subset_size):
                draw_index = np.random.randint(0, len(self.X_))
                random_sample = dataset[draw_index]
                
                #sprawdzanie zbalansowania 
                if sample == 0:
                    subset = np.append(subset, random_sample, axis = 0)
                    continue
                if np.sum(subset[:, -1])/len(subset) > balance_ratio:
                    if random_sample[-1] == 0:
                        subset = np.append(subset, random_sample, axis = 0)
                    else:
                        while(random_sample[-1] != 0):
                            draw_index = np.random.randint(0, len(self.X_))
                            random_sample = dataset[draw_index]
                        subset = np.append(subset, random_sample, axis = 0)    
                
                else:
                    if random_sample[-1] == 1:
                        subset = np.append(subset, random_sample, axis = 0)
                    else:
                        while(random_sample[-1] != 1):
                            draw_index = np.random.randint(0, len(self.X_))
                            random_sample = dataset[draw_index]
                        subset = np.append(subset, random_sample, axis = 0)
                        
            X = dataset[:, :-1]
            y = dataset[:, -1].astype(int)
                
            model = clone(self.base_estimator).fit(X, y)
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

    
        
     
