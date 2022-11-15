from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class CustomStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_features):
        self.numerical_features = numerical_features

    def fit(self, X, y=None):
        self.means = np.mean(X[self.numerical_features], axis=0)
        self.stds = np.std(X[self.numerical_features], axis=0)
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        diff = X_[self.numerical_features] - self.means
        X_[self.numerical_features] = diff / self.stds
        return X_
