# ChurnGuard — Full-Stack ML Tutorial

A complete, production-style machine learning project that teaches you how to take a model from a CSV file all the way to a monitored, containerised, CI/CD-driven service.

**What you will build:**
- A churn-prediction model trained with MLflow experiment tracking
- A FastAPI prediction service with Prometheus metrics
- A Streamlit web UI for interactive predictions
- A full Docker Compose stack (API + UI + Prometheus + Grafana)
- A GitHub Actions pipeline that tests, lints, and publishes Docker images

---

## Table of Contents

1. [Architecture overview](#1-architecture-overview)
2. [Prerequisites](#2-prerequisites)
3. [Project structure](#3-project-structure)
4. [Step 1 — The dataset and ML problem](#4-step-1--the-dataset-and-ml-problem)
5. [Step 2 — Training the model](#5-step-2--training-the-model)
6. [Step 3 — Experiment tracking with MLflow](#6-step-3--experiment-tracking-with-mlflow)
7. [Step 4 — Serving predictions with FastAPI](#7-step-4--serving-predictions-with-fastapi)
8. [Step 5 — Interactive UI with Streamlit](#8-step-5--interactive-ui-with-streamlit)
9. [Step 6 — Testing the API](#9-step-6--testing-the-api)
10. [Step 7 — Containerisation with Docker](#10-step-7--containerisation-with-docker)
11. [Step 8 — Monitoring with Prometheus and Grafana](#11-step-8--monitoring-with-prometheus-and-grafana)
12. [Step 9 — Running the full stack with Docker Compose](#12-step-9--running-the-full-stack-with-docker-compose)
13. [Step 10 — CI/CD with GitHub Actions](#13-step-10--cicd-with-github-actions)
14. [Ports and service reference](#14-ports-and-service-reference)

---

## 1. Architecture overview

```
                  ┌──────────────────────────────────────────────────────┐
                  │  Developer Workflow                                   │
                  │                                                       │
                  │  train.py ──► models/ ──► api.py ──► tests/          │
                  │     │                                                 │
                  │     └──► mlflow.db (MLflow SQLite tracking)            │
                  └──────────────────────────────────────────────────────┘
                                        │
                                  git push main
                                        │
                  ┌──────────────────────────────────────────────────────┐
                  │  GitHub Actions CI/CD                                 │
                  │                                                       │
                  │  ┌─────────┐   ┌──────┐   ┌─────────────────────┐  │
                  │  │  lint   │   │ test │──►│  build & push image  │  │
                  │  └─────────┘   └──────┘   └─────────────────────┘  │
                  │                                  │                   │
                  │                          ghcr.io/…/churnguard-api   │
                  │                          ghcr.io/…/churnguard-ui    │
                  └──────────────────────────────────────────────────────┘
                                        │
                              docker compose up
                                        │
         ┌──────────────────────────────────────────────────────────────┐
         │  Runtime Stack                                                │
         │                                                               │
         │  :8501  Streamlit UI ──────────────────────────────────────► │
         │                    └─► :8000  FastAPI  ◄── :9090 Prometheus  │
         │                             │                     │           │
         │                         models/             :3000 Grafana    │
         └──────────────────────────────────────────────────────────────┘
```

---
## 2. Prerequisites

| Tool | Minimum version | Purpose |
|------|----------------|---------|
| Python | 3.11 | Training, API, UI |
| Docker | 24 | Containerisation |
| Docker Compose | v2 | Multi-service stack |
| Git | 2.x | Version control and CI/CD |

Install the Python dependencies for local development:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.train.txt
python -m pip install -r requirements.streamlit.txt
python -m pip install pytest httpx
```

---

## 3. Project structure

```text
churnguard/
├── src/
│   ├── __init__.py
│   ├── feature_encoding.py    # Shared categorical encodings
│   ├── train.py               # Preprocessing, training, and artifact export
│   ├── api.py                 # FastAPI service with Prometheus metrics
│   └── streamlit_app.py       # Interactive prediction UI
├── tests/
│   ├── test_api.py
│   └── test_feature_encoding.py
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
│       ├── dashboards/
│       │   └── churnguard.json
│       └── provisioning/
│           ├── dashboards/
│           │   └── dashboard.yml
│           └── datasources/
│               └── prometheus.yml
├── models/                    # Generated model artifacts
├── Dockerfile.api
├── Dockerfile.streamlit
├── docker-compose.yml
├── requirements.api.txt
├── requirements.streamlit.txt
├── requirements.train.txt
├── .dockerignore
├── .env.example
├── ruff.toml
└── .github/workflows/ci.yml   # GitHub Actions pipeline
```

---

## 4. Step 1 — The dataset and ML problem

**Dataset**: IBM Telco Customer Churn (≈7,000 customers, 20 features).  
Loaded at training time directly from GitHub — no manual download needed.

**Goal**: Binary classification — predict whether a customer will churn (`Churn = 1`) or stay (`Churn = 0`).

**Input features**:

| Feature | Type | Notes |
|---------|------|-------|
| gender | Binary | 0 = Female, 1 = Male |
| SeniorCitizen | Binary | 0 / 1 |
| Partner, Dependents | Binary | 0 = No, 1 = Yes |
| tenure | Integer | Months with the company |
| PhoneService, PaperlessBilling | Binary | 0 / 1 |
| MultipleLines | 0–2 | No phone service / No / Yes |
| InternetService | 0–2 | DSL / Fiber optic / No |
| OnlineSecurity … StreamingMovies | 0–2 | No internet / No / Yes |
| Contract | 0–2 | Month-to-month / 1-year / 2-year |
| PaymentMethod | 0–3 | Four payment methods |
| MonthlyCharges | Float | Must be > 0 |
| TotalCharges | Float | Must be ≥ 0 |

---

## 5. Step 2 — Training the model

All training logic lives in `src/train.py`. Run it once before starting the API:

```bash
python -m src.train
```

**What happens, step by step:**

### 5.1 Load and preprocess data

```python
df = pd.read_csv(DATA_URL)

# Fix TotalCharges (whitespace rows arrive as empty strings)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

# Encode binary and multi-class columns
for col in binary_cols:
    df[col] = (df[col] == "Yes").astype(int)

le = LabelEncoder()
for col in multi_cols:
    df[col] = le.fit_transform(df[col])
```

### 5.2 Train / test split and scaling

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y   # stratify preserves class balance
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
```

> **Why scale?** Random Forests are not distance-based and don't strictly require scaling, but it makes the pipeline consistent when you swap in other algorithms later.

### 5.3 Model configuration

```python
params = {
    "n_estimators": 200,
    "max_depth": 8,
    "min_samples_split": 5,
    "class_weight": "balanced",   # compensates for the ~27% churn minority class
    "random_state": 42,
}
model = RandomForestClassifier(**params)
model.fit(X_train_sc, y_train)
```

### 5.4 Save artifacts

```python
joblib.dump(model,              "models/model.pkl")
joblib.dump(scaler,             "models/scaler.pkl")
joblib.dump(list(X.columns),   "models/feature_names.pkl")
```

Three separate files are saved intentionally:
- `model.pkl` — the trained classifier
- `scaler.pkl` — the fitted scaler (must use the same scale at inference time)
- `feature_names.pkl` — the ordered list of column names, so the API can reorder any incoming dictionary to match training order

---

## 6. Step 3 — Experiment tracking with MLflow

MLflow stores every training run locally in the SQLite database `mlflow.db`.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db          # open http://localhost:5000 to see run history
```

Each run records:
- **Metrics**: `accuracy`, `roc_auc`
- **Model artifact**: the serialised RandomForestClassifier (skipped in CI to keep runs fast)

```python
with mlflow.start_run():
    # ... train ...
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("roc_auc",  auc)
    if not os.getenv("CI"):
        mlflow.sklearn.log_model(model, name="model")
```

> **Why skip `log_model` in CI?** Serialising a 200-tree forest adds several seconds and megabytes to every CI run. The final `.pkl` artefacts saved by `joblib.dump` are what the API actually loads, so the MLflow model artefact is optional.

---

## 7. Step 4 — Serving predictions with FastAPI

`src/api.py` exposes three endpoints on port **8000**.

### 7.1 Start the API locally

```bash
uvicorn src.api:app --reload --port 8000
```

Open the auto-generated interactive docs: `http://localhost:8000/docs`

### 7.2 Input schema (Pydantic)

Pydantic validates every incoming request before it reaches the model:

```python
class CustomerFeatures(BaseModel):
    gender: int         = Field(..., ge=0, le=1)
    tenure: int         = Field(..., ge=0)
    MonthlyCharges: float = Field(..., gt=0)   # must be strictly positive
    # ... 16 more fields
```

If a request violates any constraint, FastAPI returns **422 Unprocessable Entity** automatically — no custom error handling needed.

### 7.3 Prediction logic

```python
@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    data = pd.DataFrame([customer.model_dump()])[feature_names]  # enforce column order
    data_scaled = scaler.transform(data)

    proba = model.predict_proba(data_scaled)[0][1]   # probability of churn
    prediction = bool(proba >= 0.5)

    risk = "high" if proba >= 0.7 else "medium" if proba >= 0.4 else "low"
    return PredictionResponse(churn_probability=round(float(proba), 4),
                              churn_prediction=prediction, risk_level=risk)
```

### 7.4 Health check

```bash
curl http://localhost:8000/health
# {"status":"ok","model":"RandomForestClassifier","version":"1.0.0"}
```

Used by Docker to decide when the container is ready to accept traffic.

### 7.5 Prometheus metrics

```bash
curl http://localhost:8000/metrics
```

Three counters/histograms are exposed:

| Metric | Type | Labels |
|--------|------|--------|
| `churnguard_requests_total` | Counter | endpoint, status |
| `churnguard_request_latency_seconds` | Histogram | endpoint |
| `churnguard_churn_predicted_total` | Counter | — |

---

## 8. Step 5 — Interactive UI with Streamlit

`src/streamlit_app.py` provides a no-code interface for exploring predictions.

```bash
streamlit run src/streamlit_app.py
# http://localhost:8501
```

The sidebar collects all 19 customer features through dropdowns, sliders, and number inputs. When you click **Predict churn risk**, the app:

1. Encodes the UI inputs into the same integer format the API expects
2. POSTs to `/predict`
3. Displays a probability gauge, prediction card, and risk-tier recommendations

The API URL is configured via an environment variable:

```bash
API_URL=http://api:8000 streamlit run src/streamlit_app.py   # Docker internal DNS
API_URL=http://localhost:8000 streamlit run src/streamlit_app.py   # local dev
```

---

## 9. Step 6 — Testing the API

The test suite in `tests/test_api.py` uses FastAPI's `TestClient`, which runs the app in-process — no network required.

```bash
pytest tests/ -v
```

**32 automated tests cover:**

| Class | What it covers |
|-------|---------------|
| `TestHealth` | `/health` returns 200 with correct structure |
| `TestPredict` | Probability in [0,1], boolean prediction, valid risk level, comparative high vs. low risk |
| `TestValidation` | Missing fields → 422, out-of-range values → 422, type errors → 422 |
| `TestMetrics` | `/metrics` returns 200, Prometheus content type, metric names present |

> **Key insight**: Always test the contract your API exposes, not the model internals. If you change the thresholds or the model later, these tests catch regressions at the boundary.

---

## 10. Step 7 — Containerisation with Docker

The project uses **two separate images** so the API and UI can be scaled and deployed independently.

### 10.1 API image (`Dockerfile.api`)

```dockerfile
FROM python:3.11-slim
RUN apt-get install -y curl          # needed for Docker health checks
COPY requirements.api.txt .
RUN pip install -r requirements.api.txt
COPY src/ src/
COPY models/ models/                 # pre-trained artifacts baked into the image
EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run manually:

```bash
python -m src.train                           # generate models/ first
docker build -f Dockerfile.api -t churnguard-api .
docker run -p 8000:8000 churnguard-api
```

### 10.2 Streamlit image (`Dockerfile.streamlit`)

```dockerfile
FROM python:3.11-slim
COPY requirements.streamlit.txt .
RUN pip install -r requirements.streamlit.txt
COPY src/streamlit_app.py src/
EXPOSE 8501
CMD ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

The Streamlit image contains **no model files** — it only talks to the API over HTTP.

---
## 11. Step 8 — Monitoring with Prometheus and Grafana

### 11.1 How metrics flow

```text
FastAPI /metrics ──► Prometheus (scrapes every 15s) ──► Grafana dashboard
```

The API records:

- Successful and failed prediction requests
- Customers predicted to churn
- Prediction latency
- Standard Python process metrics

### 11.2 Prometheus configuration

Prometheus reads `monitoring/prometheus.yml` and scrapes the API container:

```yaml
scrape_configs:
  - job_name: churnguard-api
    static_configs:
      - targets: ["api:8000"]
    metrics_path: /metrics
```

Check Prometheus health at [http://localhost:9090/-/healthy](http://localhost:9090/-/healthy).

### 11.3 Grafana provisioning

The Prometheus datasource and ChurnGuard dashboard are provisioned automatically from version-controlled files:

```text
monitoring/grafana/
├── dashboards/churnguard.json
└── provisioning/
    ├── dashboards/dashboard.yml
    └── datasources/prometheus.yml
```

The datasource uses the stable UID `prometheus`, allowing the exported dashboard to work on a fresh Grafana installation.

Create the local environment file before starting the stack:

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder with a secure local password:

```dotenv
GRAFANA_ADMIN_PASSWORD=replace-with-a-secure-password
```

Do not commit `.env`; it is excluded by `.gitignore`.

After starting Docker Compose, sign in at [http://localhost:3000](http://localhost:3000) with username `admin` and the password stored in `.env`. Open the provisioned **ChurnGuard Monitoring** dashboard.

The dashboard contains four panels:

| Panel | PromQL |
|---|---|
| Total Prediction Requests | `sum(churnguard_requests_total)` |
| Customers Predicted to Churn | `churnguard_churn_predicted_total` |
| Average Prediction Latency | `sum(churnguard_request_latency_seconds_sum) / clamp_min(sum(churnguard_request_latency_seconds_count), 1)` |
| Failed Prediction Requests | `sum(churnguard_requests_total{status="error"}) or vector(0)` |

The provisioned dashboard is read-only in Grafana because its JSON file is the source of truth.

---

## 12. Step 9 — Running the full stack with Docker Compose

`docker-compose.yml` wires up all four services:

```
api        (port 8000)  ← FastAPI + trained model
streamlit  (port 8501)  ← UI, depends on api being healthy
prometheus (port 9090)  ← scrapes api/metrics every 15 s
grafana    (port 3000)  ← reads from prometheus
```

### 12.1 Start everything

```bash
python -m src.train       # build model artifacts once (needed by the api image)
docker compose up --build
```

### 12.2 Service dependency order

Docker Compose respects the `depends_on` + `condition: service_healthy` chain:

```
prometheus ──► api  ◄── streamlit
grafana    ──► prometheus
```

The `api` service has a health check:
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
```

Streamlit will not start until the API passes its health check, preventing connection errors on startup.

### 12.3 Access the running stack

| Service | URL | Credentials |
|---------|-----|-------------|
| API docs | http://localhost:8000/docs | — |
| Streamlit UI | http://localhost:8501 | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / password from `.env` |

### 12.4 Stop and clean up

```bash
docker compose down          # stop containers, keep volumes
docker compose down -v       # stop containers and remove volumes (wipes Grafana state)
```

---

## 13. Step 10 — CI/CD with GitHub Actions

`.github/workflows/ci.yml` runs on every push and pull request to `main`.

### 13.1 Pipeline overview

```
push to main
    │
    ├── lint (parallel)
    │       ruff check src/ tests/
    │       ruff format --check src/ tests/
    │
    ├── test (parallel)
    │       pip install dependencies
    │       python -m src.train          ← train model so artifacts exist for tests
    │       pytest tests/ -v
    │
    └── build-and-push  (only on main, requires test to pass)
            python -m src.train          ← bake fresh artifacts into the image
            docker build Dockerfile.api  ──► ghcr.io/…/churnguard-api:latest
            docker build Dockerfile.streamlit ──► ghcr.io/…/churnguard-streamlit:latest
```

### 13.2 Why train the model in CI?

The `models/` directory is excluded from git (it contains large binary files). The CI pipeline trains a fresh model so that:
- Tests always run against a real, loadable model
- The published Docker images contain up-to-date artifacts without committing binaries

### 13.3 Docker image tagging strategy

Each published image gets two tags:

| Tag | Example | Purpose |
|-----|---------|---------|
| `sha-<commit>` | `sha-062da9e` | Pinned, immutable reference |
| `latest` | `latest` | Convenience tag for the most recent build |

Always deploy using the SHA tag in production — `latest` can change under you.

### 13.4 Code quality with ruff

`ruff.toml` configures three rule sets:

```toml
select = ["E", "F", "I"]   # pycodestyle errors, pyflakes, isort
ignore = ["E501"]          # line length handled separately (100-char limit)
```

The lint job enforces both **correctness** (`ruff check`) and **formatting** (`ruff format --check`), so style debates never reach code review.

---

## 14. Ports and service reference

| Service | Port | Technology | Role |
|---------|------|-----------|------|
| FastAPI | 8000 | FastAPI + Uvicorn | Predictions, health, metrics |
| Streamlit | 8501 | Streamlit | Web UI |
| Prometheus | 9090 | Prometheus | Metrics collection |
| Grafana | 3000 | Grafana | Monitoring dashboards |
| MLflow UI | 5000 | MLflow | Experiment tracking (local only) |

---

## Quick-start cheat sheet

```bash
# 1. Create and activate the Python 3.11 environment
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Install development dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.train.txt
python -m pip install -r requirements.streamlit.txt
python -m pip install pytest httpx ruff

# 3. Train the model and create the MLflow database
python -m src.train

# 4a. Run locally in separate terminals
uvicorn src.api:app --reload --port 8000
streamlit run src/streamlit_app.py
mlflow ui --backend-store-uri sqlite:///mlflow.db
pytest -q

# 4b. Or run the complete containerised stack
cp .env.example .env
# Replace the placeholder password in .env before continuing
docker compose up --build -d
docker compose ps

# 5. Check code quality
ruff check src/ tests/
ruff format --check src/ tests/

# 6. Stop the containers
docker compose down
```
