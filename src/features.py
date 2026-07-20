"""
features.py
Feature engineering for the anomaly detection pipeline.
Builds 15+ behavioral and time-based features per log entry:
  - Temporal:      hour_of_day, is_off_hours, is_weekend, seconds_since_last_event
  - Frequency:     user_event_count_1h, user_event_count_24h, source_ip_event_count_1h
  - Error ratios:  user_failed_ratio_24h, source_failed_ratio_24h
  - Behavioral:    event_type_rarity_for_user, is_rare_source_for_user,
                    is_new_source_ip, bytes_zscore_for_user
  - Categorical:   event_type_encoded, status_encoded
  - Volume:        bytes_transferred (raw), bytes_log_transformed
"""

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ---- Temporal features ----
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["is_off_hours"] = df["hour_of_day"].apply(lambda h: 1 if (h >= 22 or h < 6) else 0)
    df["is_weekend"] = df["timestamp"].dt.dayofweek.isin([5, 6]).astype(int)

    df["seconds_since_last_event"] = (
        df.groupby("user")["timestamp"].diff().dt.total_seconds().fillna(999999)
    )

    # ---- Frequency features (rolling windows per user / source) ----
    df = df.reset_index(drop=True)
    df["_row_id"] = df.index

    def _rolling_count(group_col, window):
        result = pd.Series(index=df.index, dtype=float)
        for _, group in df.groupby(group_col):
            g = group.set_index("timestamp").sort_index()
            counts = g["_row_id"].rolling(window).count()
            result.loc[g["_row_id"].values] = counts.values
        return result

    df["user_event_count_1h"] = _rolling_count("user", "1h")
    df["user_event_count_24h"] = _rolling_count("user", "24h")
    df["source_ip_event_count_1h"] = _rolling_count("source_ip", "1h")
    df = df.drop(columns=["_row_id"])

    # ---- Error ratio features ----
    df["is_failed"] = (df["status"] == "FAILED").astype(int)
    user_fail_stats = df.groupby("user")["is_failed"].transform("mean")
    df["user_failed_ratio_24h"] = user_fail_stats

    source_fail_stats = df.groupby("source_ip")["is_failed"].transform("mean")
    df["source_failed_ratio_24h"] = source_fail_stats

    # ---- Behavioral rarity features ----
    user_event_type_counts = df.groupby(["user", "event_type"])["log_id"].transform("count")
    user_total_counts = df.groupby("user")["log_id"].transform("count")
    df["event_type_rarity_for_user"] = 1 - (user_event_type_counts / user_total_counts)

    common_sources_per_user = df.groupby("user")["source_ip"].transform(
        lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]
    )
    df["is_rare_source_for_user"] = (df["source_ip"] != common_sources_per_user).astype(int)

    trusted_prefix = "10.0.1."
    df["is_untrusted_network"] = (~df["source_ip"].str.startswith(trusted_prefix)).astype(int)

    # ---- Volume / bytes features ----
    df["bytes_log_transformed"] = np.log1p(df["bytes_transferred"])
    user_bytes_mean = df.groupby("user")["bytes_transferred"].transform("mean")
    user_bytes_std = df.groupby("user")["bytes_transferred"].transform("std").fillna(1).replace(0, 1)
    df["bytes_zscore_for_user"] = (df["bytes_transferred"] - user_bytes_mean) / user_bytes_std

    # ---- Categorical encodings ----
    event_type_map = {e: i for i, e in enumerate(sorted(df["event_type"].unique()))}
    df["event_type_encoded"] = df["event_type"].map(event_type_map)
    df["status_encoded"] = (df["status"] == "FAILED").astype(int)

    feature_cols = [
        "hour_of_day", "is_off_hours", "is_weekend", "seconds_since_last_event",
        "user_event_count_1h", "user_event_count_24h", "source_ip_event_count_1h",
        "user_failed_ratio_24h", "source_failed_ratio_24h",
        "event_type_rarity_for_user", "is_rare_source_for_user", "is_untrusted_network",
        "bytes_transferred", "bytes_log_transformed", "bytes_zscore_for_user",
        "event_type_encoded", "status_encoded",
    ]

    print(f"Engineered {len(feature_cols)} features:")
    for c in feature_cols:
        print(f"  - {c}")

    return df, feature_cols
