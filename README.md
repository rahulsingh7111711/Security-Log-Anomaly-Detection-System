# Security Log Anomaly Detection System

An end-to-end anomaly detection pipeline that identifies abnormal security
events in system/application logs using **Isolation Forest** on unlabeled
data, served via a **Dockerized FastAPI** real-time inference service.

## What it does

- Ingests security log entries (login attempts, file access, privilege
  changes, API calls, config changes).
- Engineers **17 behavioral and time-based features** per event: off-hours
  activity, event frequency per user/source, failure ratios, source-IP
  trust/rarity, data-volume anomalies (z-scores), and more.
- Trains an **unsupervised Isolation Forest** (no labels used at train time)
  to flag abnormal events.
- Benchmarks the ML model against a naive rule-based baseline (flag
  off-hours OR failed login OR untrusted IP) to quantify false-positive
  reduction.
- Serves real-time scoring through a FastAPI endpoint, containerized with
  Docker.

## Results (on synthetic dataset, 1,350 log entries)

| Metric | Rule-Based Baseline | Isolation Forest |
|---|---|---|
| Precision | 0.364 | 0.673 |
| Recall | 1.000 | 0.673 |
| False Positive Rate | 0.218 | 0.041 |

**False positive reduction: 81.3%** vs. the rule-based baseline.

> Note: results depend on dataset composition, random seed, and how the
> baseline is defined. Re-run `train_model.py` and report your own numbers
> if you regenerate the dataset or change parameters — don't reuse these
> exact figures without reproducing them yourself.

Real-time inference latency (measured locally): **~25-30ms per request**.

## Project structure

```
security-log-anomaly-detection/
├── data/
│   └── security_logs.csv        # synthetic log dataset (generated)
├── models/
│   └── isolation_forest.joblib  # trained model (generated)
├── src/
│   ├── generate_logs.py         # synthetic log data generator
│   ├── features.py              # feature engineering (17 features)
│   ├── train_model.py           # trains Isolation Forest, benchmarks baseline
│   └── app.py                   # FastAPI inference service
├── Dockerfile
├── requirements.txt
└── README.md
```

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate data and train the model
```bash
cd src
python generate_logs.py
python train_model.py
```

### 3. Run the API locally
```bash
uvicorn app:app --reload --port 8000
```

### 4. Test it
```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "log_id": "TEST001",
    "timestamp": "2026-01-15T02:30:00",
    "user": "user_12",
    "source_ip": "203.0.113.5",
    "event_type": "FILE_ACCESS",
    "status": "SUCCESS",
    "bytes_transferred": 55000
  }'
```

### 5. Run with Docker
```bash
docker build -t log-anomaly-detector .
docker run -p 8000:8000 log-anomaly-detector
```

## Design notes / known limitations (be ready to discuss these in interviews)

- **Synthetic data**: this uses a generated dataset with realistic patterns
  (off-hours access, brute force, data exfiltration signatures), not real
  production logs, for demonstration and portfolio purposes.
- **Live scoring simplification**: `features.py` builds rich rolling-window
  features (event counts per hour, historical failure ratios) using full
  batch history — this is what the model is *trained* on. The live `/score`
  endpoint in `app.py` approximates these features from a single event only,
  since real historical context would come from a feature store or cache in
  a production system. This is a common training-serving skew problem in ML
  systems — be ready to explain it if asked.
- **Contamination parameter**: Isolation Forest's `contamination` is set
  using the known anomaly rate from the synthetic labels for demonstration.
  In a real unlabeled setting, this would be tuned via domain knowledge or
  a small labeled validation set.
