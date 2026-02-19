# Fraud Detection MLOps

Plateforme MLOps complète pour la détection de fraude sur transactions bancaires (dataset Credit Card Fraud). Le projet couvre l'ensemble du cycle de vie ML : entraînement, serving, monitoring, drift detection et déploiement CI/CD sur GCP.

---

## Table des matières

- [Architecture](#architecture)
- [Stack technique](#stack-technique)
- [Prérequis](#prérequis)
- [Installation & démarrage rapide](#installation--démarrage-rapide)
- [Services](#services)
- [Entraînement du modèle](#entraînement-du-modèle)
- [API d'inférence](#api-dinférence)
- [Monitoring (Prometheus & Grafana)](#monitoring-prometheus--grafana)
- [Interface utilisateur (Streamlit)](#interface-utilisateur-streamlit)
- [Tests](#tests)
- [Validation des données](#validation-des-données)
- [Drift Detection](#drift-detection)
- [CI/CD & Déploiement GCP](#cicd--déploiement-gcp)
- [Structure du projet](#structure-du-projet)
- [Variables d'environnement](#variables-denvironnement)

---

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Streamlit   │────▶│  FastAPI     │────▶│   MLflow     │
│  UI (:8501)  │     │  API (:8001) │     │ Server(:5001)│
└──────────────┘     └──────┬───────┘     └──────────────┘
                            │
                    ┌───────▼───────┐
                    │  Prometheus   │
                    │   (:9092)     │
                    └───────┬───────┘
                            │
                    ┌───────▼───────┐
                    │   Grafana     │
                    │   (:3002)     │
                    └───────────────┘
```

**Flux principal :**
1. L'utilisateur soumet une transaction via l'UI Streamlit ou directement via l'API
2. L'API FastAPI charge le modèle depuis MLflow (ou en fallback local) et retourne une prédiction (probabilité de fraude)
3. Prometheus scrape les métriques de l'API (latence, RPS, nombre de fraudes, drift PSI)
4. Grafana visualise les métriques en temps réel

---

## Stack technique

| Composant         | Technologie                          |
|-------------------|--------------------------------------|
| Langage           | Python 3.10                          |
| ML Framework      | scikit-learn (Logistic Regression)   |
| API               | FastAPI + Uvicorn                    |
| Experiment Tracking | MLflow 3.9                         |
| Monitoring        | Prometheus 2.54 + Grafana 10.4       |
| UI                | Streamlit 1.37                       |
| Conteneurisation  | Docker + Docker Compose              |
| CI/CD             | GitHub Actions                       |
| Cloud             | Google Cloud Platform (Artifact Registry, GCS, Compute Engine) |
| Linting           | Ruff                                 |
| Tests             | Pytest                               |

---

## Prérequis

- **Docker** & **Docker Compose** (v2+)
- **Python 3.10** (pour le développement local sans Docker)
- **Git LFS** (le dataset `data/creditcard.csv` est tracké via LFS)

```bash
# Installer Git LFS si nécessaire
brew install git-lfs   # macOS
git lfs install
```

---

## Installation & démarrage rapide

### 1. Cloner le repo

```bash
git clone https://github.com/AxeRaoRt/housing_mlops_projects.git
cd housing_mlops_projects
git lfs pull   # Récupérer le dataset (~150 Mo)
```

### 2. Lancer l'ensemble des services

```bash
docker compose up -d
```

Cela démarre **5 services** :
| Service      | URL                          |
|-------------|------------------------------|
| API         | http://localhost:8001         |
| MLflow      | http://localhost:5001         |
| Prometheus  | http://localhost:9092         |
| Grafana     | http://localhost:3002         |
| UI Streamlit| http://localhost:8501         |

### 3. Vérifier que tout fonctionne

```bash
# Santé de l'API
curl http://localhost:8001/health

# Prédiction de test
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"Time":0,"Amount":100,"V1":0,"V2":0,"V3":0,"V4":0,"V5":0,"V6":0,"V7":0,"V8":0,"V9":0,"V10":0,"V11":0,"V12":0,"V13":0,"V14":0,"V15":0,"V16":0,"V17":0,"V18":0,"V19":0,"V20":0,"V21":0,"V22":0,"V23":0,"V24":0,"V25":0,"V26":0,"V27":0,"V28":0}'
```

### 4. Arrêter les services

```bash
docker compose down
```

---

## Services

### MLflow Tracking Server

- **Image** : `ghcr.io/mlflow/mlflow:v3.9.0`
- **Port** : `5001` → `5000` (container)
- **Backend** : SQLite (`/mlflow/mlflow.db`)
- **Artifacts** : `/mlflow/artifacts` (volume Docker partagé)
- **UI** : http://localhost:5001

Toutes les expériences, métriques et modèles sont enregistrés ici. Le modèle enregistré sous le nom `fraud-model` est chargé automatiquement par l'API au démarrage.

---

## Entraînement du modèle

Le script `src/train.py` entraîne un pipeline **StandardScaler + LogisticRegression** (class_weight balanced) sur le dataset Credit Card Fraud.

### Via Docker Compose (recommandé)

```bash
docker compose --profile train up train
```

Le service `train` :
- Monte `data/` et `models/` en volumes
- Se connecte à MLflow pour logger l'expérience
- Produit 4 artefacts dans `models/` :
  - `model_{version}.joblib` — modèle sérialisé
  - `model_{version}_metrics.json` — ROC-AUC, PR-AUC, matrice de confusion
  - `model_{version}_schema.json` — liste des features attendues
  - `model_{version}_baseline.json` — statistiques de référence pour la détection de drift

### En local (sans Docker)

```bash
pip install -r requirements.txt
python -m src.train --data-path data/creditcard.csv --model-version v1
```

Options :
- `--data-path` : chemin vers le CSV (défaut : `data/creditcard.csv`)
- `--model-version` : tag de version (défaut : `v1`)
- `--test-size` : proportion du test set (défaut : `0.2`)
- `--no-mlflow` : désactive le logging MLflow (mode local uniquement)

---

## API d'inférence

API FastAPI exposant 5 endpoints :

| Méthode | Endpoint        | Description                                     |
|---------|----------------|-------------------------------------------------|
| GET     | `/health`      | Statut du service + version du modèle chargé    |
| GET     | `/metrics`     | Métriques Prometheus (format texte)              |
| POST    | `/predict`     | Prédiction de fraude sur une transaction         |
| POST    | `/drift`       | Détection de drift sur un batch de transactions  |
| GET     | `/drift/latest`| Dernier rapport de drift                         |

### Chargement du modèle (dual-mode)

1. **MLflow Model Registry** (par défaut) : charge le modèle `fraud-model` depuis MLflow avec le stage/alias `Production`
2. **Fichier local** (fallback) : si MLflow est indisponible, charge `models/model_v1.joblib`

### Exemple de requête `/predict`

```json
// POST /predict
{
  "Time": 406.0,
  "Amount": 239.93,
  "V1": -1.359,  "V2": -0.072,  "V3": 2.536,
  "V4": 1.378,   "V5": -0.338,  "V6": 0.462,
  "V7": 0.239,   "V8": 0.098,   "V9": 0.363,
  "V10": 0.090,  "V11": -0.551, "V12": -0.617,
  "V13": -0.991, "V14": -0.311, "V15": 1.468,
  "V16": -0.470, "V17": 0.207,  "V18": 0.025,
  "V19": 0.403,  "V20": 0.251,  "V21": -0.018,
  "V22": 0.277,  "V23": -0.110, "V24": 0.066,
  "V25": 0.128,  "V26": -0.189, "V27": 0.133,
  "V28": -0.021
}
```

```json
// Réponse
{
  "model_version": "mlflow-v1",
  "fraud_probability": 0.0342,
  "is_fraud": false
}
```

---

## Monitoring (Prometheus & Grafana)

### Prometheus

- **Port** : `9092`
- **Config** : `monitoring/prometheus.yml`
- Scrape l'API FastAPI toutes les 5 secondes sur `api:8001/metrics`

**Métriques collectées :**
- `requests_total` — compteur total de requêtes (par endpoint, méthode, status)
- `request_latency_seconds` — histogramme de latence
- `predictions_total` — compteur de prédictions
- `fraud_predictions_total` — compteur de prédictions de fraude
- `errors_total` — compteur d'erreurs
- `drift_psi_aggregate` — score PSI agrégé (drift)
- `drift_detected` — indicateur binaire de drift
- `drift_features_drifted` — nombre de features en drift

### Grafana

- **Port** : `3002`
- **Login** : `admin` / `admin123`
- **Datasource** : Prometheus (auto-provisionné)
- **Dashboard** : `mlops-view` avec 10 panels (fraude, RPS, latence p95, taux d'erreur, drift PSI, etc.)

Le datasource Prometheus est provisionné automatiquement via `monitoring/grafana/provisioning/datasources/datasource.yml`.

---

## Interface utilisateur (Streamlit)

- **Port** : `8501`
- **Code** : `ui/app.py`

L'application Streamlit permet :
- De soumettre des transactions manuellement ou aléatoirement pour obtenir une prédiction
- De visualiser les résultats du modèle
- D'explorer le dataset Credit Card Fraud
- De consulter les métriques et le statut du modèle via MLflow

---

## Tests

Le projet contient des tests unitaires et d'intégration avec **pytest**.

```bash
# Installer les dépendances
pip install -r requirements.txt

# Lancer tous les tests
PYTHONPATH=. pytest tests/ -v --tb=short
```

| Fichier                   | Description                                            |
|--------------------------|--------------------------------------------------------|
| `tests/test_api.py`     | Tests d'intégration de l'API (health, predict, metrics)|
| `tests/test_train.py`   | Smoke tests d'entraînement sur dataset synthétique     |
| `tests/test_validate_data.py` | Tests de validation des données                  |

---

## Validation des données

Le script `src/validate_data.py` vérifie la qualité du dataset avant entraînement :

- **Schéma** : présence des 30 colonnes requises (Time, Amount, V1–V28, Class)
- **Valeurs nulles** : détection et comptage des NaN
- **Types** : vérification que toutes les colonnes sont numériques
- **Plage** : montants négatifs, valeurs de la cible (0/1 uniquement)

```bash
python -m src.validate_data --no-mlflow
```

Le rapport est sauvegardé dans `reports/data_validation_report.json`.

---

## Drift Detection

La détection de drift compare les données de production à une baseline calculée durant l'entraînement.

**Méthodes utilisées :**
- **PSI (Population Stability Index)** : mesure le décalage de distribution par feature
  - PSI < 0.10 → pas de shift significatif
  - PSI 0.10–0.25 → shift modéré (à investiguer)
  - PSI > 0.25 → shift significatif (réentraînement recommandé)
- **Mean Shift (z-score)** : compare la moyenne actuelle à la baseline

**Usage via l'API :**

```bash
# Envoyer un batch de transactions pour vérifier le drift
curl -X POST http://localhost:8001/drift \
  -H "Content-Type: application/json" \
  -d '{"data": [{"Time":0,"Amount":100,"V1":0,...}, {"Time":1,"Amount":200,"V1":0.5,...}]}'
```

Les résultats sont aussi exposés en temps réel comme métriques Prometheus (panels Grafana dédiés).

---

## CI/CD & Déploiement GCP

### Pipeline CI (GitHub Actions)

Le fichier `.github/workflows/ci.yml` définit 5 jobs exécutés sur chaque push :

```
lint ──▶ test ──────────▶ docker-push ──▶ trigger-deploy
   └──▶ validate-data ──┘
```

| Job             | Description                                                  |
|----------------|--------------------------------------------------------------|
| `lint`         | Analyse statique avec Ruff sur `src/`, `api/`, `tests/`     |
| `test`         | Exécution de pytest                                          |
| `validate-data`| Validation du dataset (checkout LFS + rapport JSON)          |
| `docker-push`  | Build & push des 3 images Docker vers Google Artifact Registry + upload des configs monitoring et data vers GCS |
| `trigger-deploy`| Déclenche le workflow de déploiement dans le repo `housing_mlops_projects_deployment` |

### Images Docker

3 images sont construites et poussées vers **Google Artifact Registry** (`europe-west1-docker.pkg.dev/{PROJECT_ID}/mlops-depots/`) :

| Image          | Dockerfile         | Description                     |
|---------------|-------------------|---------------------------------|
| `fraud-api`   | `Dockerfile.infer` | API FastAPI d'inférence         |
| `fraud-train` | `Dockerfile.train` | Script d'entraînement           |
| `fraud-ui`    | `Dockerfile.ui`    | Application Streamlit           |

**Tags** : `v{run_number}` + `latest`

### Upload vers GCS

Le CI upload également vers le bucket `bucket-mlops-junia` :
- `monitoring/` — configs Prometheus et Grafana
- `data/` — dataset Credit Card Fraud

### Déploiement sur GCP

Le déploiement est géré par un repo séparé ([housing_mlops_projects_deployment](https://github.com/AxeRaoRt/housing_mlops_projects_deployment)) qui :
1. Reçoit un trigger dispatch depuis le CI avec le tag d'image
2. Déploie les containers sur une VM GCP (Compute Engine)
3. Récupère les configs monitoring et le dataset depuis le bucket GCS

### Secrets GitHub requis

| Secret             | Description                                          |
|-------------------|------------------------------------------------------|
| `GCP_PROJECT_ID`  | ID du projet Google Cloud                            |
| `GCP_SA_KEY`      | JSON du Service Account GCP (Artifact Registry + GCS)|
| `MY_GITHUB_TOKEN` | PAT GitHub pour déclencher le workflow de déploiement|

---

## Structure du projet

```
housing_mlops_projects/
├── .github/workflows/
│   └── ci.yml                    # Pipeline CI/CD GitHub Actions
├── api/
│   ├── main.py                   # API FastAPI (predict, drift, health, metrics)
│   └── schemas.py                # Schémas Pydantic (request/response)
├── data/
│   └── creditcard.csv            # Dataset Credit Card Fraud (Git LFS)
├── models/
│   ├── model_v1.joblib           # Modèle entraîné v1
│   ├── model_v1_metrics.json     # Métriques v1 (ROC-AUC, PR-AUC, etc.)
│   ├── model_v1_schema.json      # Features attendues par le modèle v1
│   ├── model_v1_baseline.json    # Baseline de drift v1
│   ├── model_v2.joblib           # Modèle v2 (idem)
│   ├── model_v2_metrics.json
│   ├── model_v2_schema.json
│   └── model_v2_baseline.json
├── monitoring/
│   ├── prometheus.yml            # Config Prometheus (scrape de l'API)
│   └── grafana/
│       ├── dashboards/
│       │   ├── dashboard.yml     # Provider de dashboards Grafana
│       │   └── mlops-view.json   # Dashboard JSON (10 panels)
│       └── provisioning/
│           └── datasources/
│               └── datasource.yml # Datasource Prometheus auto-provisionnée
├── reports/
│   └── data_validation_report.json
├── src/
│   ├── config.py                 # Configuration centralisée (env vars)
│   ├── drift.py                  # Détection de drift (PSI + mean shift)
│   ├── io_utils.py               # Utilitaires I/O (save/load model, MLflow)
│   ├── predict.py                # Fonctions de prédiction
│   ├── train.py                  # Script d'entraînement
│   └── validate_data.py          # Validation du dataset
├── tests/
│   ├── test_api.py               # Tests d'intégration API
│   ├── test_train.py             # Tests d'entraînement
│   └── test_validate_data.py     # Tests de validation
├── ui/
│   ├── app.py                    # Application Streamlit
│   └── assets/                   # Ressources statiques
├── docker-compose.yml            # Orchestration de tous les services
├── Dockerfile.infer              # Image API d'inférence
├── Dockerfile.train              # Image d'entraînement
├── Dockerfile.ui                 # Image UI Streamlit
└── requirements.txt              # Dépendances Python
```

---

## Variables d'environnement

| Variable                  | Défaut                       | Description                                  |
|--------------------------|------------------------------|----------------------------------------------|
| `DATA_PATH`              | `data/creditcard.csv`        | Chemin vers le dataset                       |
| `MODELS_DIR`             | `models`                     | Répertoire de sortie des modèles             |
| `REPORTS_DIR`            | `reports`                    | Répertoire des rapports de validation        |
| `SEED`                   | `42`                         | Graine de reproductibilité                   |
| `TARGET_COL`             | `Class`                      | Colonne cible                                |
| `MLFLOW_TRACKING_URI`    | `http://localhost:5000`      | URL du serveur MLflow                        |
| `MLFLOW_EXPERIMENT_NAME` | `fraud-detection`            | Nom de l'expérience MLflow                   |
| `MLFLOW_MODEL_NAME`      | `fraud-model`                | Nom du modèle dans le registry MLflow        |
| `MLFLOW_MODEL_STAGE`     | `Production`                 | Stage ou alias du modèle à charger           |
| `MODEL_VERSION`          | `v1`                         | Version locale du modèle                     |
| `PRED_THRESHOLD`         | `0.5`                        | Seuil de probabilité pour classifier fraude  |
| `DRIFT_PSI_THRESHOLD`    | `0.25`                       | Seuil PSI pour déclarer un drift             |
| `DRIFT_Z_THRESHOLD`      | `2.0`                        | Seuil z-score pour déclarer un mean shift    |
| `LOG_LEVEL`              | `INFO`                       | Niveau de logs de l'API                      |
