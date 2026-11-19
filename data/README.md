# Données

`raw/titanic.csv` — 891 passagers, 12 colonnes. Il s'agit du fichier d'entraînement (`train.csv`) de la compétition Kaggle [Titanic — Machine Learning from Disaster](https://www.kaggle.com/c/titanic), tel que redistribué par [datasciencedojo/datasets](https://github.com/datasciencedojo/datasets/blob/master/titanic.csv).

SHA-256 : `4a437fde05fe5264e1701a7387ac6fb75393772ba38bb2c9c566405af5af4bd7`

| Colonne | Description |
|---|---|
| `PassengerId` | Identifiant |
| `Survived` | Cible : 1 = survivant, 0 = décédé |
| `Pclass` | Classe du billet (1, 2, 3) |
| `Name` | Nom, contenant le titre (Mr, Mrs, Miss, Master…) |
| `Sex` | Sexe |
| `Age` | Âge (20 % de valeurs manquantes) |
| `SibSp` | Nombre de frères, sœurs ou conjoints à bord |
| `Parch` | Nombre de parents ou enfants à bord |
| `Ticket` | Numéro de billet |
| `Fare` | Tarif (£) |
| `Cabin` | Numéro de cabine (77 % de valeurs manquantes) |
| `Embarked` | Port d'embarquement : C = Cherbourg, Q = Queenstown, S = Southampton |

Le fichier `test.csv` de Kaggle ne contient pas la cible : il sert uniquement aux soumissions (`make predict INPUT=test.csv`).
