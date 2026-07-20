"""
generate_logs.py
Generates synthetic system/application security logs for the anomaly detection
project. Simulates realistic normal user/service behavior plus injected
anomalous patterns (brute force, off-hours access, privilege escalation,
unusual source IPs, error spikes).
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

N_NORMAL = 1200
N_ANOMALOUS = 150

USERS = [f"user_{i}" for i in range(1, 41)] + ["svc_backup", "svc_deploy", "svc_monitor"]
SOURCES = ["10.0.1." + str(i) for i in range(2, 30)]  # trusted internal subnet
RARE_SOURCES = ["203.0.113." + str(i) for i in range(1, 15)]  # external/unusual
EVENT_TYPES = ["LOGIN", "LOGOUT", "FILE_ACCESS", "FILE_WRITE", "PRIVILEGE_CHANGE",
               "API_CALL", "CONFIG_CHANGE", "DB_QUERY"]
STATUS_CODES = ["SUCCESS", "SUCCESS", "SUCCESS", "SUCCESS", "FAILED"]  # normal ~20% fail rate baseline

start_time = datetime(2026, 1, 1, 0, 0, 0)


def random_normal_timestamp():
    """Normal activity clusters around business hours (9am-7pm), weekday-biased."""
    day_offset = np.random.randint(0, 30)
    hour = int(np.clip(np.random.normal(loc=13, scale=3.5), 0, 23))
    minute = np.random.randint(0, 60)
    second = np.random.randint(0, 60)
    return start_time + timedelta(days=day_offset, hours=hour, minutes=minute, seconds=second)


def random_anomalous_timestamp():
    """Anomalies skew toward off-hours (11pm - 5am)."""
    day_offset = np.random.randint(0, 30)
    hour = np.random.choice([23, 0, 1, 2, 3, 4], p=[0.15, 0.2, 0.2, 0.2, 0.15, 0.1])
    minute = np.random.randint(0, 60)
    second = np.random.randint(0, 60)
    return start_time + timedelta(days=day_offset, hours=int(hour), minutes=minute, seconds=second)


def gen_normal_rows(n):
    rows = []
    for _ in range(n):
        rows.append({
            "timestamp": random_normal_timestamp(),
            "user": np.random.choice(USERS),
            "source_ip": np.random.choice(SOURCES, p=_uniform(SOURCES)),
            "event_type": np.random.choice(EVENT_TYPES),
            "status": np.random.choice(STATUS_CODES),
            "bytes_transferred": max(0, int(np.random.normal(2000, 800))),
            "label": 0,  # normal
        })
    return rows


def gen_anomalous_rows(n):
    rows = []
    patterns = ["brute_force", "off_hours_access", "privilege_escalation",
                "rare_source_spike", "data_exfil"]
    for _ in range(n):
        pattern = np.random.choice(patterns)
        base = {
            "timestamp": random_anomalous_timestamp(),
            "user": np.random.choice(USERS),
            "source_ip": np.random.choice(SOURCES + RARE_SOURCES),
            "event_type": np.random.choice(EVENT_TYPES),
            "status": "SUCCESS",
            "bytes_transferred": max(0, int(np.random.normal(2000, 800))),
            "label": 1,  # anomalous
        }
        if pattern == "brute_force":
            base["event_type"] = "LOGIN"
            base["status"] = "FAILED"
        elif pattern == "off_hours_access":
            base["event_type"] = np.random.choice(["FILE_ACCESS", "CONFIG_CHANGE"])
        elif pattern == "privilege_escalation":
            base["event_type"] = "PRIVILEGE_CHANGE"
            base["status"] = "SUCCESS"
        elif pattern == "rare_source_spike":
            base["source_ip"] = np.random.choice(RARE_SOURCES)
            base["event_type"] = "API_CALL"
        elif pattern == "data_exfil":
            base["event_type"] = "FILE_ACCESS"
            base["bytes_transferred"] = int(np.random.normal(50000, 15000))
        rows.append(base)
    return rows


def _uniform(seq):
    p = np.ones(len(seq)) / len(seq)
    return p


def main():
    normal = gen_normal_rows(N_NORMAL)
    anomalous = gen_anomalous_rows(N_ANOMALOUS)
    df = pd.DataFrame(normal + anomalous)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["log_id"] = [f"LOG{100000+i}" for i in range(len(df))]
    df = df[["log_id", "timestamp", "user", "source_ip", "event_type",
              "status", "bytes_transferred", "label"]]
    out_path = "/home/claude/security-log-anomaly-detection/data/security_logs.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} log entries -> {out_path}")
    print(f"  Normal: {N_NORMAL} | Anomalous (ground truth, for eval only): {N_ANOMALOUS}")


if __name__ == "__main__":
    main()
