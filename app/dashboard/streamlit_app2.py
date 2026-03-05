import streamlit as st
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os, sys

# Ensure project root is on sys.path so `from app...` imports work when Streamlit runs
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(dotenv_path=ROOT / ".env")

from app.pipeline.pipeline import run_pipeline

st.set_page_config(page_title="Product Review — Sentiment & Action", layout="wide")

st.title("Product Review — Sentiment & Action (PoC)")

uploaded = st.file_uploader("Upload reviews CSV", type=["csv"]) 
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    sample = Path(__file__).parents[2] / "sample_data" / "reviews1.csv"
    if sample.exists():
        st.info("No file uploaded — using sample_data/reviews1.csv")
        df = pd.read_csv(sample)
    else:
        st.warning("No data available. Upload a CSV or add sample_data/reviews1.csv")
        st.stop()

st.subheader("Preview")
st.dataframe(df.head())

st.sidebar.header("Controls")
n_clusters = st.sidebar.slider("Number of clusters", min_value=1, max_value=8, value=3)

if st.sidebar.button("Run analysis"):
    with st.spinner("Running pipeline: ingestion → embeddings → clustering → agent..."):
        df_clean, summaries, suggestions = run_pipeline(df, n_clusters=n_clusters)

    st.success("Pipeline finished")

    st.markdown("**Overview**")
    st.write(f"Total reviews: {len(df)} — Kept: {len(df_clean)} — Spam filtered: {len(df) - len(df_clean)}")

    st.markdown("**Sentiment distribution (kept reviews)**")
    sd = df_clean["sentiment"].value_counts(normalize=True)
    st.bar_chart(sd)

    st.markdown("**Clusters**")
    # show cluster summaries
    for lbl, meta in summaries.items():
        st.subheader(f"Cluster {lbl} — {meta['count']} reviews")
        st.write("Top terms: " + ", ".join(meta["top_terms"]))
        st.write("Sentiment distribution: ", meta["sentiment_dist"])
        st.write("Examples:")
        for s in meta["samples"]:
            st.write("- ", s)

    st.markdown("**Agent suggestions**")
    if suggestions:
        try:
            for item in suggestions:
                st.write(f"**{item.get('action')}** — {item.get('rationale')} (priority: {item.get('priority')})")
        except Exception:
            st.write(suggestions)
    else:
        st.write("No suggestions returned.")

    st.markdown("---")
    st.header("Kept reviews (sample)")
    st.dataframe(df_clean.head(200))
    st.write("Pipeline run complete. Replace any placeholders with production integrations as needed.")
