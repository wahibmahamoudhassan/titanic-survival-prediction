"""Chargement, contrôle et partition des données."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from titanic.config import ROOT

RAW_COLUMNS = ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "SibSp",
               "Parch", "Ticket", "Fare", "Cabin", "Embarked"]


def load_data(params: dict) -> pd.DataFrame:
    path = Path(params["data"]["path"])
    df = pd.read_csv(path if path.is_absolute() else ROOT / path)
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {sorted(missing)}")
    if df["PassengerId"].duplicated().any():
        raise ValueError("Identifiants passagers dupliqués.")
    if not set(df["Survived"].unique()) <= {0, 1}:
        raise ValueError("La cible Survived doit valoir 0 ou 1.")
    return df


def split_data(df: pd.DataFrame, params: dict):
    """Hold-out stratifié. Aucun prétraitement n'est appris avant ce découpage."""
    target = params["data"]["target"]
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=params["data"]["test_size"],
                            stratify=y, random_state=params["seed"])
