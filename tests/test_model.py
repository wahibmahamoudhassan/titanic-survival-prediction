import json

import pandas as pd
import pytest

from titanic.predict import predict
from titanic.train import run


@pytest.fixture(scope="module")
def trained(tmp_path_factory, fast_params):
    d = tmp_path_factory.mktemp("run")
    report = run(fast_params, models_dir=d / "models", figures_dir=d / "fig", make_figures=False)
    import joblib
    return report, joblib.load(d / "models" / "model.joblib"), d


def test_le_modele_bat_les_baselines(trained):
    report, *_ = trained
    best = report["cv"][report["model"]]["mean"]
    assert best > report["cv"]["baseline_gender"]["mean"]
    assert best > report["cv"]["baseline_majority"]["mean"] + 0.15


def test_metriques_enregistrees(trained):
    _, _, d = trained
    report = json.loads((d / "models" / "metrics.json").read_text(encoding="utf-8"))
    assert {"model", "cv", "test", "test_accuracy_ci95", "best_params"} <= set(report)
    lo, hi = report["test_accuracy_ci95"]
    assert lo <= report["test"][report["model"]]["accuracy"] <= hi


def test_predictions_format_kaggle(trained, splits):
    _, model, _ = trained
    X_test = splits[1]
    out = predict(X_test, model=model)
    assert list(out.columns) == ["PassengerId", "Survived", "probability"]
    assert out["Survived"].isin([0, 1]).all() and out["probability"].between(0, 1).all()


def test_coherence_metier(trained):
    """Une femme de 1re classe doit avoir une probabilité de survie bien supérieure à un homme de 3e classe."""
    _, model, _ = trained
    base = {"PassengerId": 0, "Name": "X, Mrs. Y", "Age": 30, "SibSp": 0, "Parch": 0,
            "Ticket": "A", "Fare": 80.0, "Cabin": "B5", "Embarked": "S"}
    woman = {**base, "Pclass": 1, "Sex": "female"}
    man = {**base, "Pclass": 3, "Sex": "male", "Name": "X, Mr. Y", "Fare": 8.0, "Cabin": None}
    p = predict(pd.DataFrame([woman, man]), model=model)["probability"].tolist()
    assert p[0] > 0.8 and p[1] < 0.3


def test_colonnes_manquantes_refusees(trained):
    _, model, _ = trained
    with pytest.raises(ValueError, match="Colonnes manquantes"):
        predict(pd.DataFrame([{"PassengerId": 1, "Sex": "male"}]), model=model)
