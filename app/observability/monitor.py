import pandas as pd


def sentiment_distribution(df: pd.DataFrame, sentiment_col: str = "sentiment"):
    if sentiment_col not in df.columns:
        return {}
    return df[sentiment_col].value_counts(normalize=True).to_dict()


def detect_drift(history_distribution: dict, current_distribution: dict, threshold: float = 0.1):
    # Simple L1 difference
    keys = set(history_distribution) | set(current_distribution)
    diff = sum(abs(history_distribution.get(k, 0) - current_distribution.get(k, 0)) for k in keys)
    return diff > threshold
