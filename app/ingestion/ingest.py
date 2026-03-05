import pandas as pd


def read_csv(path: str) -> pd.DataFrame:
    """Read CSV containing reviews. Expected columns: id, text, metadata..."""
    df = pd.read_csv(path)
    # basic cleaning placeholder
    df = df.dropna(subset=[df.columns[0]])
    return df
