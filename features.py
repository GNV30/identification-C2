from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "timestamp",
    "src_ip",
    "dst_ip",
    "dst_port",
    "protocol",
    "duration_ms",
    "bytes_out",
    "bytes_in",
    "packets_out",
    "packets_in",
}

GROUP_COLUMNS = ["src_ip", "dst_ip", "dst_port", "protocol"]


def load_flows(path: str | Path) -> pd.DataFrame:
    flows = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(flows.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Input CSV is missing required columns: {missing_text}")

    flows = flows.copy()
    flows["timestamp"] = pd.to_datetime(flows["timestamp"], errors="coerce")
    if flows["timestamp"].isna().any():
        raise ValueError("Column 'timestamp' contains values that cannot be parsed as dates")

    numeric_columns = ["dst_port", "duration_ms", "bytes_out", "bytes_in", "packets_out", "packets_in"]
    for column in numeric_columns:
        flows[column] = pd.to_numeric(flows[column], errors="coerce").fillna(0)

    return flows.sort_values("timestamp")


def coefficient_of_variation(values: pd.Series) -> float:
    if len(values) < 2:
        return 1.0

    mean = values.mean()
    if mean == 0:
        return 1.0

    return float(values.std(ddof=0) / mean)


def _interval_stats(group: pd.DataFrame) -> pd.Series:
    intervals = group["timestamp"].sort_values().diff().dt.total_seconds().dropna()
    if intervals.empty:
        return pd.Series(
            {
                "avg_interval_sec": 0.0,
                "interval_cv": 1.0,
                "unique_interval_count": 0,
            }
        )

    rounded_intervals = intervals.round().astype(int)
    return pd.Series(
        {
            "avg_interval_sec": float(intervals.mean()),
            "interval_cv": coefficient_of_variation(intervals),
            "unique_interval_count": int(rounded_intervals.nunique()),
        }
    )


def build_features(flows: pd.DataFrame) -> pd.DataFrame:
    flows = flows.copy()
    flows["total_bytes"] = flows["bytes_out"] + flows["bytes_in"]
    flows["total_packets"] = flows["packets_out"] + flows["packets_in"]

    grouped = flows.groupby(GROUP_COLUMNS, dropna=False)
    features = grouped.agg(
        flow_count=("timestamp", "size"),
        first_seen=("timestamp", "min"),
        last_seen=("timestamp", "max"),
        avg_duration_ms=("duration_ms", "mean"),
        avg_bytes_out=("bytes_out", "mean"),
        avg_bytes_in=("bytes_in", "mean"),
        avg_total_bytes=("total_bytes", "mean"),
        std_total_bytes=("total_bytes", "std"),
        avg_total_packets=("total_packets", "mean"),
    ).reset_index()

    intervals = grouped.apply(_interval_stats).reset_index()
    features = features.merge(intervals, on=GROUP_COLUMNS, how="left")
    features["std_total_bytes"] = features["std_total_bytes"].fillna(0.0)
    features["bytes_out_ratio"] = features["avg_bytes_out"] / (
        features["avg_bytes_out"] + features["avg_bytes_in"] + 1
    )
    features["observation_window_sec"] = (
        features["last_seen"] - features["first_seen"]
    ).dt.total_seconds()
    features["flows_per_minute"] = features["flow_count"] / (
        np.maximum(features["observation_window_sec"], 60) / 60
    )

    if "label" in flows.columns:
        labels = grouped["label"].agg(lambda values: values.mode().iat[0]).reset_index()
        features = features.merge(labels, on=GROUP_COLUMNS, how="left")

    return features.sort_values(["flow_count", "flows_per_minute"], ascending=False).reset_index(drop=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = set(GROUP_COLUMNS + ["first_seen", "last_seen", "label", "risk_score", "verdict"])
    return [column for column in frame.columns if column not in excluded]
