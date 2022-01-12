from collections import defaultdict
import inspect

import numpy as np
import pandas as pd

from . import configuration as cfg

class BaseEstimator:
    """
    Sklearn inspired BaseEstimator
    """
    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("Estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    def __getstate__(self):
        try:
            state = super().__getstate__()
        except AttributeError:
            state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        try:
            super().__setstate__(state)
        except AttributeError:
            self.__dict__.update(state)

class Persistent:
    def save(self, path_dir):
        raise NotImplementedError()

class Forecaster:
    def forecast(self, ts):
        """
        Params:
        -------
        pd.DataFrame, ["timestamp", "value", \["outliers"]]

        Returns:
        --------
        pd.DataFrame, ["timestamp", "value", "forecast", "residual"]
        """
        raise NotImplementedError()

class AnomalyDetector(BaseEstimator, Persistent):
    @staticmethod
    def _is_valid(ts):
        return type(ts) is pd.DataFrame and \
            cfg.cols["X"] in ts.columns

    @staticmethod
    def _X(ts):
        """
        ts[X] as numpy
        """
        return ts[cfg.cols["X"]].to_numpy()
    
    @staticmethod
    def _timestamps(ts):
        """
        ts[timestamps]
        """
        if cfg.cols["timestamp"] not in ts.columns:
            return None
        return ts[cfg.cols["timestamp"]]
    
    @staticmethod
    def _y(ts):
        """
        ts[y] as numpy
        """
        if cfg.cols["y"] not in ts.columns:
            return None
        return ts[cfg.cols["y"]].to_numpy()
    

    def fit(self, ts):
        """
        Params
        ------
        ts : pd.DataFrame
            Must have columns ["Timestamp", "value"]. Optionally "outlier"
        Return
        ------
        self
        """
        raise NotImplementedError()

    def predict(self, ts):
        """
        Params
        ------
        ts : pd.DataFrame
            Must have columns ["Timestamp", "value"]
        Returns:
        --------
        pd.DataFrame, columns ["Timestamp", "value", "outlier"]
                        optionally also \["forecast"] and \["outlier_score"]]
        """
        if not hasattr(self, "predict_proba"):
            raise NotImplementedError()
        pred_prob = self.predict_proba(ts)
        pred_prob[cfg.cols["y"]] = np.array(
            np.greater(
                pred_prob[cfg.cols["pred_prob"]].to_numpy(), 
                getattr(self, "t", 0.5)
            ), 
            dtype = int
        )
        if cfg.cols["timestamp"] in ts.columns:
            pred_prob[cfg.cols["timestamp"]] = ts[cfg.cols["timestamp"]]
        return pred_prob
