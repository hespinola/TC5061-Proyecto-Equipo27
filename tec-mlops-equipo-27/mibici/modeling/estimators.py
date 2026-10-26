"""Estimadores desplegables; conservan el preprocesamiento y el umbral elegido."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


class LagRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, column="salidas_rezago_168"):
        self.column = column

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return X[self.column].to_numpy(dtype=float)


class FrequencyClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        data = X.assign(target=np.asarray(y, dtype=int))
        self.rates_ = data.groupby(["estacion_id", "hora_del_dia"]).target.mean()
        self.fallback_ = float(np.mean(y))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        keys = pd.MultiIndex.from_frame(X[["estacion_id", "hora_del_dia"]])
        p = self.rates_.reindex(keys).fillna(self.fallback_).to_numpy(dtype=float)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class ThresholdClassifier(ClassifierMixin, BaseEstimator):
    """Envuelve un estimador ya ajustado; predict aplica el umbral de validación."""

    def __init__(self, estimator, threshold):
        self.estimator = estimator
        self.threshold = threshold
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)
