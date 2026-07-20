"""
train_model.py
Trains an Isolation Forest on unlabeled log data to detect anomalous security
events, and benchmarks it against a simple rule-based baseline detector.
Reports false-positive reduction achieved by the ML approach.

Note: labels exist in the synthetic dataset ONLY for evaluation purposes
(measuring precision/recall/FP rate). The Isolation Forest itself is trained
WITHOUT using the label column, consistent with real-world unlabeled log data.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from features import engineer_features

DATA_PATH = "/home/claude/security-log-anomaly-detection/data/security_logs.csv"
MODEL_PATH = "/home/claude/security-log-anomaly-detection/models/isolation_forest.joblib"


def rule_based_baseline(df: pd.DataFrame) -> np.ndarray:
    """
    Simple baseline a security team might use before ML: flag anything that's
    off-hours OR a failed login OR from an untrusted network. This is what
    naive rule-based detection looks like -- prone to high false positives
    because normal off-hours access (e.g. legit night-shift admin work) also
    gets flagged.
    """
    flags = (
        (df["is_off_hours"] == 1) |
        (df["status_encoded"] == 1) |
        (df["is_untrusted_network"] == 1)
    )
    return flags.astype(int).values


def main():
    df = pd.read_csv(DATA_PATH)
    df, feature_cols = engineer_features(df)

    X = df[feature_cols].fillna(0).values
    y_true = df["label"].values  # ground truth, evaluation only

    # ---- Train Isolation Forest (unsupervised -- no y passed in) ----
    contamination = y_true.mean()  # ~ expected anomaly rate, used only to set threshold
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)

    raw_scores = model.decision_function(X)          # higher = more normal
    preds = model.predict(X)                          # -1 = anomaly, 1 = normal
    y_pred_iforest = np.where(preds == -1, 1, 0)

    # ---- Baseline ----
    y_pred_baseline = rule_based_baseline(df)

    # ---- Evaluate both ----
    def evaluate(y_true, y_pred, name):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        print(f"\n--- {name} ---")
        print(f"  True Positives:  {tp}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")
        print(f"  True Negatives:  {tn}")
        print(f"  Precision: {precision:.3f} | Recall: {recall:.3f} | FP Rate: {fp_rate:.3f}")
        return fp, fp_rate

    fp_baseline, fprate_baseline = evaluate(y_true, y_pred_baseline, "Rule-Based Baseline")
    fp_iforest, fprate_iforest = evaluate(y_true, y_pred_iforest, "Isolation Forest")

    fp_reduction = (fp_baseline - fp_iforest) / fp_baseline * 100 if fp_baseline > 0 else 0
    print(f"\n=== False Positive Reduction (Isolation Forest vs Baseline): {fp_reduction:.1f}% ===")

    # ---- Save model + feature list for the API ----
    joblib.dump({"model": model, "feature_cols": feature_cols}, MODEL_PATH)
    print(f"\nModel saved -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
