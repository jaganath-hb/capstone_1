import json
import re
from collections import defaultdict
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

# ---------------- Agent suggestions (enhanced, structured, session-safe) ----------------

    # ---------------- Agent suggestions (clean, structured) ----------------


st.markdown("## Agent suggestions")

# -- Helpers --
def _normalize_suggestions(suggestions_in):
    """Normalize suggestions into a list of dicts: {action, rationale, priority}."""
    if suggestions_in is None:
        return []

    if isinstance(suggestions_in, str):
        try:
            suggestions_in = json.loads(suggestions_in)
        except Exception:
            return [{"action": suggestions_in, "rationale": "Returned as plain text.", "priority": 999}]

    if isinstance(suggestions_in, dict):
        if {"action", "rationale", "priority"}.issubset({k.lower() for k in suggestions_in.keys()}):
            suggestions_in = [suggestions_in]
        else:
            picked = None
            for v in suggestions_in.values():
                if isinstance(v, list):
                    picked = v
                    break
            suggestions_in = picked if picked is not None else [suggestions_in]

    if not isinstance(suggestions_in, list):
        return []

    out = []
    for i, it in enumerate(suggestions_in):
        if isinstance(it, dict):
            action = it.get("action") or it.get("title") or f"Item {i+1}"
            rationale = it.get("rationale") or it.get("reason") or "No rationale provided."
            pr = it.get("priority", it.get("rank", 999))
            try:
                pr = int(pr)
            except Exception:
                pr = 999
            out.append({"action": action, "rationale": rationale, "priority": pr})
        else:
            out.append({"action": str(it), "rationale": "Derived from plain-text suggestion.", "priority": 999})
    return out

def _extract_keywords(text: str):
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    words = [w for w in text.split() if len(w) > 2]
    seen, out = set(), []
    for w in words:
        if w not in seen:
            out.append(w); seen.add(w)
    return out[:12]

def _rank_related_clusters(suggestion_keywords, summaries_dict, top_k=1):
    """Pick up to top_k clusters that best match the suggestion, but do NOT print raw headers."""
    if not summaries_dict:
        return []

    scores = []
    for lbl, meta in summaries_dict.items():
        top_terms = [t.lower() for t in meta.get("top_terms", [])]
        overlap = len(set(suggestion_keywords) & set(top_terms))
        partial = sum(1 for kw in suggestion_keywords for tt in top_terms if kw in tt and kw != tt)
        score = overlap * 2 + partial
        if score > 0:
            scores.append((lbl, score))

    if not scores:
        # fallback to the most populous cluster just for context
        by_count = sorted(summaries_dict.items(), key=lambda kv: kv[1].get("count", 0), reverse=True)[:1]
        scores = [(by_count[0][0], 1)]

    scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
    return [lbl for lbl, _ in scores]

def _highlight(text, terms):
    if not terms:
        return text
    terms = sorted(set([t for t in terms if len(t) > 2]), key=len, reverse=True)
    out = text
    for t in terms:
        out = re.sub(rf"(?i)(\b{re.escape(t)}\b)", r"<mark>\1</mark>", out)
    return out

# -- Render --
items = _normalize_suggestions(suggestions)

if items:
    # sort by priority (1 = highest)
    items = sorted(items, key=lambda x: x.get("priority", 999))

    # Compact metrics row
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total suggestions", len(items))
    with c2:
        st.metric("High priority (≤3)", sum(1 for i in items if (i.get("priority", 999) or 999) <= 3))
    with c3:
        st.metric("Clusters detected", len(summaries))

    st.write("")

    for idx, it in enumerate(items, start=1):
        action = it.get("action", "Untitled action")
        rationale = it.get("rationale", "No rationale provided.")
        pr = it.get("priority", 999)

        # Badge color by priority
        badge_color = "#e91e63" if pr == 1 else "#ff9800" if pr <= 3 else "#607d8b"

        # Card-like container (no extra cluster noise)
        st.markdown(
            f"""
<div style="border:1px solid #e5e7eb;border-radius:10px;padding:14px;margin-bottom:10px;">
  <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
    <div style="font-weight:700;font-size:1.05rem;">{idx}. {action}</div>
    <div style="background-color:{badge_color}; color:white; padding:2px 8px; border-radius:12px; font-size:0.85em;">
      Priority {pr}
    </div>
  </div>
  <div style="margin-top:6px;color:#374151;">{rationale}</div>
""",
            unsafe_allow_html=True,
        )

        # Related top terms + samples (cleanly formatted; no raw "Cluster X ..." line)
        keywords = _extract_keywords(action + " " + rationale)
        related = _rank_related_clusters(keywords, summaries, top_k=1)

        for r_lbl in related:
            meta = summaries.get(r_lbl, {})
            top_terms = meta.get("top_terms", [])[:8]
            #samples = meta.get("samples", [])[:2]  # keep it succinct

            # Top terms chips (bold any that match)
            if top_terms:
                chips = " ".join(
                    [
                        f"<span style='background:#eef3ff;border:1px solid #d6e2ff;border-radius:12px;padding:2px 8px;margin-right:6px;'>"
                        f"{('<strong>'+tt+'</strong>') if any(tt.lower() == k for k in keywords) else tt}"
                        f"</span>"
                        for tt in top_terms
                    ]
                )
                st.markdown(f"<div style='margin-top:8px;'><strong>Top terms</strong>: {chips}</div>", unsafe_allow_html=True)

            # Sample reviews with highlights
            if samples:
                st.markdown("<div style='margin-top:6px;'><strong>Samples</strong>:</div>", unsafe_allow_html=True)
                for s in samples:
                    highlighted = _highlight(s, keywords + top_terms)
                    st.markdown(f"<div style='color:#111827;'>• {highlighted}</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)  # close the card container

    # Optional raw JSON inspector
    with st.expander("Show raw suggestions JSON"):
        st.code(json.dumps(items, indent=2), language="json")

else:
    st.info("No suggestions returned.")
# ---------------- end Agent suggestions ----------------