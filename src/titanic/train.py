"""Entraînement : baselines, optimisation des hyperparamètres, sélection, évaluation finale.

Exécution : python -m titanic.train
"""
import json
import platform

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay  # noqa: E402
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_val_score  # noqa: E402

from titanic import __version__  # noqa: E402
from titanic.config import FIGURES_DIR, MODELS_DIR, load_params  # noqa: E402
from titanic.data import load_data, split_data  # noqa: E402
from titanic.evaluate import bootstrap_ci, classification_metrics  # noqa: E402
from titanic.models import build_baselines, build_candidates  # noqa: E402

PRETTY = {
    "baseline_majority": "Baseline : classe majoritaire",
    "baseline_gender": "Baseline : « les femmes survivent »",
    "logistic_regression": "Régression logistique",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
}


def run(params: dict | None = None, models_dir=MODELS_DIR, figures_dir=FIGURES_DIR, make_figures=True) -> dict:
    params = params or load_params()
    seed = params["seed"]
    metric = params["selection_metric"]
    X_train, X_test, y_train, y_test = split_data(load_data(params), params)
    cv = RepeatedStratifiedKFold(n_splits=params["cv"]["n_splits"], n_repeats=params["cv"]["n_repeats"],
                                 random_state=seed)

    cv_scores, fitted, best_params = {}, {}, {}
    for name, est in build_baselines().items():
        cv_scores[name] = cross_val_score(est, X_train, y_train, cv=cv, scoring=metric)
        fitted[name] = est.fit(X_train, y_train)

    for name, pipe in build_candidates(seed).items():
        search = GridSearchCV(pipe, params["models"][name], cv=cv, scoring=metric, n_jobs=-1, refit=True)
        search.fit(X_train, y_train)
        i = search.best_index_
        n_folds = cv.get_n_splits()
        cv_scores[name] = np.array([search.cv_results_[f"split{k}_test_score"][i] for k in range(n_folds)])
        fitted[name] = search.best_estimator_
        best_params[name] = {k.replace("classifier__", ""): v for k, v in search.best_params_.items()}

    # Sélection sur la validation croisée du TRAIN uniquement (les baselines ne sont pas candidates)
    candidates = list(best_params)
    best = max(candidates, key=lambda n: cv_scores[n].mean())
    model = fitted[best]

    # Évaluation finale : une seule consultation du test
    test_results = {}
    for name, est in fitted.items():
        pred = est.predict(X_test)
        proba = est.predict_proba(X_test)[:, 1]
        test_results[name] = classification_metrics(y_test, pred, proba)
    best_pred = model.predict(X_test)
    ci = bootstrap_ci(y_test.to_numpy(), best_pred, seed=seed)

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / "model.joblib")
    report = {
        "model": best,
        "version": __version__,
        "selection": f"{metric} moyenne en validation croisée répétée ({cv.get_n_splits()} plis) sur le train",
        "best_params": best_params,
        "cv": {n: {"mean": float(s.mean()), "std": float(s.std())} for n, s in cv_scores.items()},
        "test": test_results,
        "test_accuracy_ci95": ci,
        "n_train": int(len(y_train)), "n_test": int(len(y_test)),
        "environment": {"python": platform.python_version(), "scikit-learn": sklearn.__version__},
    }
    (models_dir / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if make_figures:
        figures_dir.mkdir(parents=True, exist_ok=True)
        _figures(cv_scores, fitted, best, model, X_test, y_test, figures_dir, seed)
    return report


def _save(fig, path) -> None:
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _figures(cv_scores, fitted, best, model, X_test, y_test, out, seed):
    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
    order = sorted(cv_scores, key=lambda n: cv_scores[n].mean())
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.boxplot([cv_scores[n] for n in order], orientation="horizontal", widths=0.55)
    ax.set_yticks(range(1, len(order) + 1), [PRETTY[n] for n in order])
    ax.set_xlabel("Accuracy (validation croisée 5 × 5 sur le train)")
    ax.set_title("Comparaison des modèles")
    _save(fig, out / "cv_comparison.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    for name in ["baseline_gender", "logistic_regression", "random_forest", "gradient_boosting"]:
        RocCurveDisplay.from_predictions(y_test, fitted[name].predict_proba(X_test)[:, 1], name=PRETTY[name], ax=ax)
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs (rappel)")
    ax.set_title("Courbes ROC — jeu de test")
    _save(fig, out / "roc_curves.png")

    fig, ax = plt.subplots(figsize=(4.6, 4))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=["Décédé", "Survivant"],
                                          cmap="Blues", colorbar=False, ax=ax)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_title(f"Matrice de confusion — {PRETTY[best]}")
    _save(fig, out / "confusion_matrix.png")

    imp = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=seed, scoring="accuracy")
    s = pd.Series(imp.importances_mean, index=X_test.columns).drop(["PassengerId", "Ticket"]).sort_values()
    err = pd.Series(imp.importances_std, index=X_test.columns)[s.index]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(s.index.str.replace("Name", "Name (→ titre)"), s.values, xerr=err.values, color="#1f6feb")
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("Baisse d'accuracy quand la colonne est permutée (test, 30 répétitions)")
    ax.set_title("Importance par permutation des colonnes brutes")
    _save(fig, out / "permutation_importance.png")


if __name__ == "__main__":
    r = run()
    print(f"{'Modèle':38s} {'CV accuracy':>16s} {'Test accuracy':>14s} {'Test ROC-AUC':>13s}")
    for n in ["baseline_majority", "baseline_gender", "logistic_regression", "random_forest", "gradient_boosting"]:
        cv, t = r["cv"][n], r["test"][n]
        print(f"{PRETTY[n]:38s} {cv['mean']:.3f} ± {cv['std']:.3f} {t['accuracy']:14.3f} {t['roc_auc']:13.3f}")
    lo, hi = r["test_accuracy_ci95"]
    print(f"\nModèle retenu : {PRETTY[r['model']]} {r['best_params'][r['model']]}")
    print(f"Accuracy test : {r['test'][r['model']]['accuracy']:.3f} (IC 95 % bootstrap : [{lo:.3f} ; {hi:.3f}])")
