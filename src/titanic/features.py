"""Ingénierie des variables, entièrement intégrée au Pipeline scikit-learn.

Tout ce qui est *appris* sur les données (âge médian par titre, médianes,
modalités) est estimé dans `fit`, donc uniquement sur les données
d'entraînement de chaque pli : aucune fuite vers la validation ou le test.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TITLE_MAP = {
    "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs", "Lady": "Rare", "Countess": "Rare",
    "the Countess": "Rare", "Dona": "Rare", "Don": "Rare", "Sir": "Rare", "Jonkheer": "Rare",
    "Capt": "Rare", "Col": "Rare", "Major": "Rare", "Dr": "Rare", "Rev": "Rare",
}

NUMERIC_FEATURES = ["Age", "LogFare", "FamilySize", "Pclass"]
BINARY_FEATURES = ["IsAlone", "HasCabin", "IsChild"]
CATEGORICAL_FEATURES = ["Sex", "Embarked", "Title", "Deck"]


def extract_title(name: pd.Series) -> pd.Series:
    """« Braund, Mr. Owen Harris » → « Mr » ; titres peu fréquents regroupés en « Rare »."""
    raw = name.str.extract(r",\s*([^.]+)\.", expand=False).str.strip()
    return raw.map(TITLE_MAP).fillna("Rare")


class TitanicFeatures(BaseEstimator, TransformerMixin):
    """Crée les variables métier à partir des colonnes brutes du Titanic."""

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        X["Title"] = extract_title(X["Name"])
        # Âge médian par titre (un « Master » est un garçon, une « Mrs » une adulte…)
        self.age_by_title_ = X.groupby("Title")["Age"].median().to_dict()
        self.age_global_ = float(X["Age"].median())
        self.fare_median_ = float(X["Fare"].median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        out = pd.DataFrame(index=X.index)
        out["Pclass"] = X["Pclass"].astype(float)
        out["Sex"] = X["Sex"]
        out["Embarked"] = X["Embarked"]
        out["Title"] = extract_title(X["Name"])
        age_fill = out["Title"].map(self.age_by_title_).fillna(self.age_global_)
        out["Age"] = X["Age"].fillna(age_fill)
        out["IsChild"] = (out["Age"] < 14).astype(int)
        fare = X["Fare"].fillna(self.fare_median_).clip(lower=0)
        out["LogFare"] = np.log1p(fare)
        out["FamilySize"] = X["SibSp"] + X["Parch"] + 1
        out["IsAlone"] = (out["FamilySize"] == 1).astype(int)
        out["HasCabin"] = X["Cabin"].notna().astype(int)
        deck = X["Cabin"].str[0].fillna("U")
        out["Deck"] = deck.replace({"T": "A", "G": "F"})   # ponts très rares regroupés
        return out


def build_preprocessor(scale_numeric: bool = True) -> Pipeline:
    numeric = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric.append(("scale", StandardScaler()))
    columns = ColumnTransformer([
        ("num", Pipeline(numeric), NUMERIC_FEATURES),
        ("bin", "passthrough", BINARY_FEATURES),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL_FEATURES),
    ], verbose_feature_names_out=False)
    return Pipeline([("features", TitanicFeatures()), ("columns", columns)])
