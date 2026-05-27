from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from c2meta.features import build_features, feature_columns, load_flows


DEFAULT_RULES = {
    "threshold": 0.65,
    "weights": {
        "flow_count": 0.18,
        "interval_regularity": 0.24,
        "small_payload": 0.18,
        "outbound_ratio": 0.14,
        "short_duration": 0.12,
        "low_port_diversity": 0.14,
    },
    "feature_limits": {
        "high_flow_count": 20,
        "small_avg_bytes": 900,
        "short_duration_ms": 2500,
    },
}


def load_rules(path: str | Path | None) -> dict:
    if path is None:
        return DEFAULT_RULES

    with Path(path).open("r", encoding="utf-8") as file:
        rules = json.load(file)

    merged = DEFAULT_RULES.copy()
    merged.update(rules)
    merged["weights"] = DEFAULT_RULES["weights"] | rules.get("weights", {})
    merged["feature_limits"] = DEFAULT_RULES["feature_limits"] | rules.get("feature_limits", {})
    return merged


def _clamp(value: pd.Series | float) -> pd.Series | float:
    return value.clip(0, 1) if isinstance(value, pd.Series) else max(0, min(1, value))


def score_with_rules(features: pd.DataFrame, rules: dict | None = None) -> pd.DataFrame:
    rules = rules or DEFAULT_RULES
    weights = rules["weights"]
    limits = rules["feature_limits"]
    scored = features.copy()

    flow_component = _clamp(scored["flow_count"] / limits["high_flow_count"])
    regularity_component = _clamp(1 - scored["interval_cv"])
    payload_component = _clamp(1 - (scored["avg_total_bytes"] / limits["small_avg_bytes"]))
    outbound_component = _clamp(scored["bytes_out_ratio"] * 1.6)
    duration_component = _clamp(1 - (scored["avg_duration_ms"] / limits["short_duration_ms"]))
    interval_diversity_component = _clamp(1 - (scored["unique_interval_count"] / scored["flow_count"]))

    scored["risk_score"] = (
        weights["flow_count"] * flow_component
        + weights["interval_regularity"] * regularity_component
        + weights["small_payload"] * payload_component
        + weights["outbound_ratio"] * outbound_component
        + weights["short_duration"] * duration_component
        + weights["low_port_diversity"] * interval_diversity_component
    ).round(3)
    scored["verdict"] = scored["risk_score"].ge(rules["threshold"]).map(
        {True: "suspicious", False: "benign"}
    )
    return scored.sort_values("risk_score", ascending=False).reset_index(drop=True)


def scan_csv(csv_path: str | Path, rules_path: str | Path | None = None) -> pd.DataFrame:
    flows = load_flows(csv_path)
    features = build_features(flows)
    return score_with_rules(features, load_rules(rules_path))


def train_model(csv_path: str | Path, model_path: str | Path, test_size: float = 0.3) -> str:
    flows = load_flows(csv_path)
    features = build_features(flows)
    if "label" not in features.columns:
        raise ValueError("Training requires a 'label' column with values like 'benign' and 'c2'")

    x = features[feature_columns(features)]
    y = features["label"].map({"benign": 0, "c2": 1}).fillna(features["label"]).astype(int)

    stratify = y if y.nunique() > 1 and y.value_counts().min() > 1 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=42, stratify=stratify
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=120, random_state=42)),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    report = classification_report(y_test, predictions, target_names=["benign", "c2"], zero_division=0)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": feature_columns(features)}, model_path)
    return report


def score_with_model(csv_path: str | Path, model_path: str | Path) -> pd.DataFrame:
    flows = load_flows(csv_path)
    features = build_features(flows)
    bundle = joblib.load(model_path)
    columns = bundle["feature_columns"]
    probabilities = bundle["model"].predict_proba(features[columns])[:, 1]

    scored = features.copy()
    scored["risk_score"] = probabilities.round(3)
    scored["verdict"] = scored["risk_score"].ge(0.5).map({True: "suspicious", False: "benign"})
    return scored.sort_values("risk_score", ascending=False).reset_index(drop=True)
