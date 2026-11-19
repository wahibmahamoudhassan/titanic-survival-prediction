PY := python
export PYTHONPATH := src

.PHONY: install train test lint predict notebook clean

install:   ## Installer les dépendances
	$(PY) -m pip install -r requirements-dev.txt

train:     ## Comparer les modèles, sélectionner, évaluer, produire figures et métriques (~1 min)
	$(PY) -m titanic.train

test:      ## Lancer les tests
	$(PY) -m pytest -v

lint:
	ruff check src tests

predict:   ## Prédire sur un fichier : make predict INPUT=test.csv
	$(PY) -m titanic.predict --input $(INPUT) --output submission.csv --kaggle

notebook:
	jupyter lab notebooks/

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
