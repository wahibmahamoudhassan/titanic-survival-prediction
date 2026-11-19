def test_dimensions(data):
    assert data.shape == (891, 12)
    assert 0.35 < data["Survived"].mean() < 0.42


def test_split_stratifie_et_disjoint(splits, data):
    X_train, X_test, y_train, y_test = splits
    assert set(X_train["PassengerId"]).isdisjoint(X_test["PassengerId"])
    assert len(X_train) + len(X_test) == len(data)
    assert abs(y_train.mean() - y_test.mean()) < 0.02
    assert "Survived" not in X_train.columns
