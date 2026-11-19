"""Modèles candidats et baselines."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from titanic.features import build_preprocessor


class GenderBaseline(BaseEstimator, ClassifierMixin):
    """Règle historique « les femmes survivent, les hommes non » (≈ 79 % sur Kaggle)."""

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = (X["Sex"] == "female").to_numpy(dtype=float)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (X["Sex"] == "female").to_numpy(dtype=int)


def build_candidates(seed: int) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline([
            ("prep", build_preprocessor(scale_numeric=True)),
            ("classifier", LogisticRegression(max_iter=2000, random_state=seed)),
        ]),
        "random_forest": Pipeline([
            ("prep", build_preprocessor(scale_numeric=False)),
            ("classifier", RandomForestClassifier(random_state=seed, n_jobs=-1)),
        ]),
        "gradient_boosting": Pipeline([
            ("prep", build_preprocessor(scale_numeric=False)),
            ("classifier", HistGradientBoostingClassifier(random_state=seed)),
        ]),
    }


def build_baselines() -> dict:
    return {
        "baseline_majority": DummyClassifier(strategy="most_frequent"),
        "baseline_gender": GenderBaseline(),
    }
