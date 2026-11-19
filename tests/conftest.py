import copy

import pytest

from titanic.config import load_params
from titanic.data import load_data, split_data


@pytest.fixture(scope="session")
def params():
    return load_params()


@pytest.fixture(scope="session")
def fast_params(params):
    """Grille et validation croisée réduites pour des tests rapides."""
    p = copy.deepcopy(params)
    p["cv"] = {"n_splits": 3, "n_repeats": 1}
    p["models"] = {
        "logistic_regression": {"classifier__C": [1.0]},
        "random_forest": {"classifier__n_estimators": [100], "classifier__max_depth": [6]},
        "gradient_boosting": {"classifier__max_iter": [100]},
    }
    return p


@pytest.fixture(scope="session")
def data(params):
    return load_data(params)


@pytest.fixture(scope="session")
def splits(data, params):
    return split_data(data, params)
