import numpy as np
import pandas as pd

from titanic.features import TitanicFeatures, build_preprocessor, extract_title


def test_extraction_des_titres():
    names = pd.Series(["Braund, Mr. Owen Harris", "Heikkinen, Miss. Laina", "Moran, Mme. X",
                       "Rothes, the Countess. of (Lucy)", "Palsson, Master. Gosta Leonard", "Smith, Dr. John"])
    assert extract_title(names).tolist() == ["Mr", "Miss", "Mrs", "Rare", "Master", "Rare"]


def test_aucune_valeur_manquante_apres_transformation(splits):
    X_train, X_test, *_ = splits
    prep = build_preprocessor().fit(X_train)
    out = prep.transform(X_test)
    assert not np.isnan(out).any()


def test_imputation_apprise_sur_le_train_uniquement(splits):
    """L'âge médian par titre ne doit dépendre que des données passées à fit()."""
    X_train, X_test, *_ = splits
    f1 = TitanicFeatures().fit(X_train)
    f2 = TitanicFeatures().fit(pd.concat([X_train, X_test]))
    assert f1.age_by_title_ != f2.age_by_title_ or f1.age_global_ != f2.age_global_


def test_variables_derivees():
    row = pd.DataFrame([{"PassengerId": 1, "Pclass": 3, "Name": "Doe, Master. Tom", "Sex": "male", "Age": np.nan,
                         "SibSp": 1, "Parch": 2, "Ticket": "X", "Fare": 20.0, "Cabin": "C85", "Embarked": "S"}])
    f = TitanicFeatures()
    f.age_by_title_, f.age_global_, f.fare_median_ = {"Master": 4.0}, 28.0, 14.0
    out = f.transform(row).iloc[0]
    assert out["Age"] == 4.0 and out["IsChild"] == 1
    assert out["FamilySize"] == 4 and out["IsAlone"] == 0
    assert out["HasCabin"] == 1 and out["Deck"] == "C"


def test_categorie_inconnue_toleree(splits):
    X_train, X_test, *_ = splits
    prep = build_preprocessor().fit(X_train)
    X_new = X_test.head(3).copy()
    X_new["Embarked"] = "Z"                     # port inconnu à l'entraînement
    assert prep.transform(X_new).shape[0] == 3
