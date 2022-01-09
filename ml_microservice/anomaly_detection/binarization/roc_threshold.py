import numpy as np
from sklearn.metrics import roc_curve
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

class ROCthreshold(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None, **fit_params):
        """
        Params:
        - X : scores
        - y : true labels
        """
        if y is None:
            raise ValueError("y must be not None")
        fpr, tpr, thrs = roc_curve(y, X)
        J = tpr - fpr
        idx = np.argmax(J)
        self.t_ = thrs[idx]
        return self

    def transform(self, X):
        if not hasattr(self, "t_"):
            raise RuntimeError("Must be fitted first")
        return np.array(np.greater(X, self.t_), dtype=int)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)