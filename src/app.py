"""
app.py
FastAPI inference service for the Security Log Anomaly Detection System.
Loads the trained Isolation Forest and exposes a real-time scoring endpoint.

Run locally:
    uvicorn app:app --reload --port 8000

Test:
    curl -X POST http://localhost:8000/score -H "Content-Type: application/json" -d '{...}'
"""

import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(_THIS_DIR, "..", "models", "isolation_forest.joblib"),
)

app = FastAPI(
    title="Security Log Anomaly Detection API",
    description="Real-time inference service scoring security log events for anomalies.",
    version="1.0.0",
)

_bundle = joblib.load(MODEL_PATH)
MODEL = _bundle["model"]
FEATURE_COLS = _bundle["feature_cols"]


class LogEvent(BaseModel):
    log_id: str
    timestamp: str          # ISO format
    user: str
    source_ip: str
    event_type: str
    status: str
    bytes_transferred: int


class ScoreResponse(BaseModel):
    log_id: str
    is_anomaly: bool
    anomaly_score: float
    latency_ms: float


def _build_single_row_features(event: LogEvent) -> pd.DataFrame:
    """
    Lightweight, single-event feature builder for real-time scoring.
    (Batch feature engineering with rolling windows -- as in features.py --
    is used for training; live scoring approximates the same feature space
    using only the current event's attributes, since historical context
    would come from a feature store / cache in a production system.)
    """
    ts = pd.to_datetime(event.timestamp)
    row = {
        "hour_of_day": ts.hour,
        "is_off_hours": 1 if (ts.hour >= 22 or ts.hour < 6) else 0,
        "is_weekend": 1 if ts.dayofweek >= 5 else 0,
        "seconds_since_last_event": 0,  # no history available for a cold single request
        "user_event_count_1h": 1,
        "user_event_count_24h": 1,
        "source_ip_event_count_1h": 1,
        "user_failed_ratio_24h": 1.0 if event.status == "FAILED" else 0.0,
        "source_failed_ratio_24h": 1.0 if event.status == "FAILED" else 0.0,
        "event_type_rarity_for_user": 0.5,
        "is_rare_source_for_user": 0 if event.source_ip.startswith("10.0.1.") else 1,
        "is_untrusted_network": 0 if event.source_ip.startswith("10.0.1.") else 1,
        "bytes_transferred": event.bytes_transferred,
        "bytes_log_transformed": np.log1p(event.bytes_transferred),
        "bytes_zscore_for_user": 0.0,
        "event_type_encoded": hash(event.event_type) % 8,
        "status_encoded": 1 if event.status == "FAILED" else 0,
    }
    return pd.DataFrame([row])[FEATURE_COLS]


@app.get("/health")
def health():
    return {"status": "ok", "model": "IsolationForest", "features": len(FEATURE_COLS)}


@app.post("/score", response_model=ScoreResponse)
def score_event(event: LogEvent):
    start = time.perf_counter()

    X = _build_single_row_features(event)
    pred = MODEL.predict(X)[0]                 # -1 anomaly, 1 normal
    raw_score = MODEL.decision_function(X)[0]   # higher = more normal

    latency_ms = (time.perf_counter() - start) * 1000

    return ScoreResponse(
        log_id=event.log_id,
        is_anomaly=bool(pred == -1),
        anomaly_score=round(float(-raw_score), 4),  # flip sign so higher = more anomalous
        latency_ms=round(latency_ms, 2),
    )
