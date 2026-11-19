"""Prédictions sur un nouveau fichier (format Kaggle `test.csv`).

Exécution : python -m titanic.predict --input test.csv --output submission.csv
"""
import argparse

import joblib
import pandas as pd

from titanic.config import MODELS_DIR

REQUIRED = ["PassengerId", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]


def predict(df: pd.DataFrame, model=None) -> pd.DataFrame:
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {sorted(missing)}")
    model = model or joblib.load(MODELS_DIR / "model.joblib")
    proba = model.predict_proba(df[REQUIRED])[:, 1]
    return pd.DataFrame({"PassengerId": df["PassengerId"], "Survived": (proba >= 0.5).astype(int),
                         "probability": proba.round(4)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--kaggle", action="store_true", help="format de soumission (sans probabilité)")
    args = parser.parse_args()
    out = predict(pd.read_csv(args.input))
    if args.kaggle:
        out = out[["PassengerId", "Survived"]]
    out.to_csv(args.output, index=False)
    print(f"{len(out)} prédictions écrites dans {args.output}")


if __name__ == "__main__":
    main()
