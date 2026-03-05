from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from app.ingestion.ingest import read_csv
from app.embeddings.embed import embed_texts
from app.clustering.cluster import cluster_embeddings
from app.agent.assistant import propose_improvements
import re

POS_WORDS = {"good", "great", "love", "excellent", "awesome", "nice", "amazing", "happy", "fast"}
NEG_WORDS = {"bad", "terrible", "hate", "crash", "crashes", "slow", "broken", "bug", "error", "fail", "problem", "confusing"}

STOPWORDS = set(["the","and","is","in","on","a","an","to","for","of","with","it","this","that"])


def _is_spam(text: str) -> bool:
    if not isinstance(text, str):
        return True
    t = text.lower()
    if len(t) < 8:
        return True
    if re.search(r"buy now|free money|click here|http|www\.|!!!|\$\$\$", t):
        return True
    # too many repeated characters
    if re.search(r"(.)\1{5,}", t):
        return True
    return False


def _classify_sentiment(text: str) -> str:
    if not isinstance(text, str) or len(text.strip()) == 0:
        return "neutral"
    t = re.findall(r"\w+", text.lower())
    pos = sum(1 for w in t if w in POS_WORDS)
    neg = sum(1 for w in t if w in NEG_WORDS)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


def _top_terms(texts, n=5):
    vec = CountVectorizer(stop_words=list(STOPWORDS)).fit(texts)
    X = vec.transform(texts)
    sums = np.asarray(X.sum(axis=0)).ravel()
    terms = np.array(vec.get_feature_names_out())
    top_idx = np.argsort(sums)[::-1][:n]
    return [terms[i] for i in top_idx if sums[i] > 0]


def run_pipeline(df: pd.DataFrame, n_clusters: int = 3) -> Tuple[pd.DataFrame, Dict[int, Dict[str, Any]], Any]:
    """Run ingestion->spam filter->sentiment->embeddings->clustering->agent.

    Returns: (df_with_metadata, cluster_summaries, suggestions)
    """
    df = df.copy()
    text_col = None
    # guess text column
    for c in df.columns:
        if c.lower() in ("text", "review", "comment", "message"):
            text_col = c
            break
    if text_col is None:
        text_col = df.columns[0]

    df["_text"] = df[text_col].astype(str)
    df["is_spam"] = df["_text"].apply(_is_spam)
    df_clean = df[~df["is_spam"]].reset_index(drop=True)
    df_clean["sentiment"] = df_clean["_text"].apply(_classify_sentiment)

    # embeddings
    texts = df_clean["_text"].tolist()
    if len(texts) == 0:
        return df_clean, {}, []
    embeddings = embed_texts(texts)

    labels = cluster_embeddings(embeddings, n_clusters=n_clusters)
    df_clean["cluster"] = labels

    # build cluster summaries
    summaries = {}
    for lbl in sorted(df_clean["cluster"].unique()):
        rows = df_clean[df_clean["cluster"] == lbl]
        samples = rows["_text"].tolist()[:3]
        top_terms = _top_terms(rows["_text"].tolist(), n=5)
        summaries[int(lbl)] = {
            "count": int(len(rows)),
            "top_terms": top_terms,
            "samples": samples,
            "sentiment_dist": rows["sentiment"].value_counts(normalize=True).to_dict(),
        }

    # prepare textual cluster summary for agent
    cluster_summaries = []
    for lbl, meta in summaries.items():
        cluster_summaries.append(f"Cluster {lbl}: count={meta['count']}; top_terms={', '.join(meta['top_terms'])}; samples={meta['samples']}")
    cluster_summaries_text = "\n".join(cluster_summaries)

    suggestions = propose_improvements(cluster_summaries_text)

    return df_clean, summaries, suggestions


if __name__ == "__main__":
    # quick local test runner
    import os
    p = os.path.join(os.path.dirname(__file__), "..", "..", "sample_data", "reviews.csv")
    df = pd.read_csv(p)
    r = run_pipeline(df, n_clusters=2)
    print(r[1])
    print(r[2])
