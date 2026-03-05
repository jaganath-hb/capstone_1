import json
import re
from collections import defaultdict
import streamlit as st
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os, sys

# ────────────────────────────────────────────────────────────────────────────────
# Bootstrap
# ────────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(dotenv_path=ROOT / ".env")

from app.pipeline.pipeline import run_pipeline

st.set_page_config(page_title="Product Review — Sentiment & Action", layout="wide")

# Global cosmetic CSS (theme-friendly, subtle)
st.markdown(
    """
<style>
:root {
  --card-bg: rgba(255,255,255,0.9);
  --card-br: 12px;
  --card-bd: 1px solid rgba(0,0,0,0.06);
  --card-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 12px rgba(0,0,0,0.04);
  --text-muted: #6b7280;
  --badge-1: linear-gradient(90deg,#ef4444,#f43f5e);
  --badge-2: linear-gradient(90deg,#f59e0b,#f97316);
  --badge-3: linear-gradient(90deg,#64748b,#475569);
  --chip-bg: #eef2ff;
  --chip-bd: #c7d2fe;
}
@media (prefers-color-scheme: dark) {
  :root {
    --card-bg: rgba(17,24,39,0.55);
    --card-bd: 1px solid rgba(255,255,255,0.08);
    --card-shadow: 0 1px 3px rgba(0,0,0,0.4), 0 4px 12px rgba(0,0,0,0.25);
    --text-muted: #9ca3af;
    --chip-bg: rgba(99,102,241,0.15);
    --chip-bd: rgba(129,140,248,0.5);
  }
}
.section-title { font-size:1.55rem; font-weight:800; margin: 6px 0 4px 0; }
.kpi { border-radius: var(--card-br); border: var(--card-bd); background: var(--card-bg); box-shadow: var(--card-shadow); padding:14px 16px; }
.card { border-radius: var(--card-br); border: var(--card-bd); background: var(--card-bg); box-shadow: var(--card-shadow); padding:16px 18px; margin-bottom:12px; }
.card-title { font-weight:750; font-size:1.05rem; display:flex; gap:10px; align-items:center; }
.badge { color:#fff; padding:2px 10px; border-radius:999px; font-size:0.85rem; font-weight:700; }
.badge-1 { background: var(--badge-1); }
.badge-2 { background: var(--badge-2); }
.badge-3 { background: var(--badge-3); }
.subtle { color: var(--text-muted); }
.divider { height:1px; background: linear-gradient(90deg, rgba(0,0,0,0), rgba(148,163,184,0.35), rgba(0,0,0,0)); margin: 10px 0 4px; }
.chips span { display:inline-block; background: var(--chip-bg); border:1px solid var(--chip-bd); color:inherit; padding:4px 10px; border-radius:999px; margin:3px 6px 0 0; font-size:0.85rem; }
.sample-item { margin: 2px 0; }
.download-wrap { text-align:right; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("✨ Product Review — Sentiment & Action (PoC)")

# ────────────────────────────────────────────────────────────────────────────────
# Ingestion
# ────────────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────────────
# Controls
# ────────────────────────────────────────────────────────────────────────────────
st.sidebar.header("Controls")
n_clusters = st.sidebar.slider("Number of clusters", min_value=1, max_value=8, value=3)

# ────────────────────────────────────────────────────────────────────────────────
# Run pipeline
# ────────────────────────────────────────────────────────────────────────────────
if st.sidebar.button("Run analysis"):
    with st.spinner("Running pipeline: ingestion → embeddings → clustering → agent..."):
        df_clean, summaries, suggestions = run_pipeline(df, n_clusters=n_clusters)

    # Persist results to survive Streamlit reruns (prevents NameError)
    st.session_state.update(
        df_clean=df_clean,
        summaries=summaries,
        suggestions=suggestions,
    )

    st.success("✅ Pipeline finished")

    st.markdown("**Overview**")
    st.write(
        f"Total reviews: **{len(df)}** — Kept: **{len(df_clean)}** — "
        f"Spam filtered: **{len(df) - len(df_clean)}**"
    )

    st.markdown("**Sentiment distribution (kept reviews)**")
    sd = df_clean["sentiment"].value_counts(normalize=True)
    st.bar_chart(sd)

    st.markdown("**Clusters**")
    # show cluster summaries in cards
    grid = st.columns(2) if len(st.session_state["summaries"]) > 1 else [st]
    i = 0
    for lbl, meta in st.session_state["summaries"].items():
        st.subheader(f"Cluster {lbl} — {meta['count']} reviews")
        st.write("Sentiment distribution: ", meta["sentiment_dist"])
        #st.write("Sentiment distribution: ", meta["sentiment_dist"])
## ────────────────────────────────────────────────────────────────────────────────
# Agent Suggestions — exact format + cosmetics
# ────────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🧠 Agent suggestions</div>', unsafe_allow_html=True)

# Pull from session state; safe on first load
_suggestions = st.session_state.get("suggestions")
_summaries   = st.session_state.get("summaries", {})
_df_clean    = st.session_state.get("df_clean")

# ---------- Helpers: normalization + sanitization ----------
_META_PATTERNS = [
    r"\bcluster\s*\d+\b",
    r"\bcluster:\s*\d+\b",
    r"\bcount\s*=\s*\d+\b",
    r"\btop_terms\s*=\s*\[?.*?\]?",
    r"\bsentiment_dist\s*=\s*\{.*?\}",
    r"\bsamples\s*=\s*\[.*?\]",
]
_META_RE = re.compile("|".join(_META_PATTERNS), flags=re.IGNORECASE | re.DOTALL)

def _strip_meta(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = _META_RE.sub("", s)
    s = re.sub(r"\s*[,;:\-\|]\s*", " ", s)   # normalize leftover separators
    s = re.sub(r"\s{2,}", " ", s).strip(" .,:;-")
    return s.strip()

def _normalize_suggestions(sug):
    """Return list[{action,rationale,priority}] with meta removed; accept list|dict|JSON str."""
    if sug is None:
        return []
    if isinstance(sug, str):
        try:
            sug = json.loads(sug)
        except Exception:
            sug = [{"action": sug, "rationale": "Returned as plain text.", "priority": 999}]
    if isinstance(sug, dict):
        if {"action", "rationale", "priority"}.issubset({k.lower() for k in sug.keys()}):
            sug = [sug]
        else:
            arr = None
            for v in sug.values():
                if isinstance(v, list):
                    arr = v; break
            sug = arr if arr is not None else [sug]
    if not isinstance(sug, list):
        return []
    out = []
    for i, it in enumerate(sug):
        if isinstance(it, dict):
            action    = _strip_meta(it.get("action") or it.get("title") or f"Item {i+1}")
            rationale = _strip_meta(it.get("rationale") or it.get("reason") or "")
            pr        = it.get("priority", it.get("rank", 999))
            try:
                pr = int(pr)
            except Exception:
                pr = 999
            out.append({"action": action, "rationale": rationale, "priority": pr})
        else:
            out.append({"action": _strip_meta(str(it)), "rationale": "", "priority": 999})
    return out

def _extract_keywords(text: str):
    text = _strip_meta(text or "").lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    words = [w for w in text.split() if len(w) > 2]
    seen = set(); ordered = []
    for w in words:
        if w not in seen:
            ordered.append(w); seen.add(w)
    return ordered[:12]

def _best_cluster_ids_for_suggestion(action, rationale, summaries, k=1):
    """Pick up to k clusters by overlap of suggestion keywords with cluster top_terms."""
    if not summaries:
        return []
    kw = _extract_keywords(f"{action} {rationale}")
    scored = []
    for lbl, meta in summaries.items():
        top_terms = [str(t).lower() for t in meta.get("top_terms", [])]
        if not top_terms:
            continue
        overlap = len(set(kw) & set(top_terms))
        partial = sum(1 for t in kw for tt in top_terms if t in tt and t != tt)
        score = overlap * 2 + partial
        if score > 0:
            scored.append((lbl, score))
    if not scored and summaries:
        try:
            biggest = max(summaries.items(), key=lambda kv: kv[1].get("count", 0))[0]
            scored = [(biggest, 1)]
        except Exception:
            return []
    scored.sort(key=lambda x: x[1], reverse=True)
    return [lbl for lbl, _ in scored[:k]]

# ---------- Render (exact format + polished UI) ----------
items = _normalize_suggestions(_suggestions)

if items:
    # Compute metrics
    high_pri = sum(1 for i in items if (i.get("priority", 999) or 999) <= 3)
    items.sort(key=lambda x: x.get("priority", 999))

    k1, k2, k3, kSpacer, kDl = st.columns([1,1,1,2,1])
    with k1:
        st.markdown('<div class="kpi"><div class="subtle">Total suggestions</div>'
                    f'<div style="font-size:1.4rem;font-weight:800;">{len(items)}</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown('<div class="kpi"><div class="subtle">High priority (≤3)</div>'
                    f'<div style="font-size:1.4rem;font-weight:800;">{high_pri}</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown('<div class="kpi"><div class="subtle">Clusters detected</div>'
                    f'<div style="font-size:1.4rem;font-weight:800;">{len(_summaries)}</div></div>', unsafe_allow_html=True)
    with kDl:
        # Download normalized suggestions
        st.download_button(
            label="⬇️ Download JSON",
            data=json.dumps(items, indent=2),
            file_name="agent_suggestions.json",
            mime="application/json",
            use_container_width=True
        )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    for idx, it in enumerate(items, start=1):
        action    = it.get("action", "").strip() or "Untitled action"
        rationale = it.get("rationale", "").strip()
        pr        = it.get("priority", 999)

        badge_class = "badge-1" if pr == 1 else "badge-2" if pr <= 3 else "badge-3"

        # Card
        st.markdown(
            f"""
<div class="card">
  <div class="card-title">✅ {idx}. {action}
    <span class="badge {badge_class}">Priority {pr}</span>
  </div>
  {"<div class='subtle' style='margin-top:6px;'>" + _strip_meta(rationale) + "</div>" if rationale else ""}
""",
            unsafe_allow_html=True,
        )

        # Related cluster → show Top terms (5) and Samples (3)
        for cid in _best_cluster_ids_for_suggestion(action, rationale, _summaries, k=1):
            meta = _summaries.get(cid, {})
            top_terms = [str(t) for t in meta.get("top_terms", [])][:5]
            samples   = [str(s) for s in meta.get("samples", [])][:3]

            if top_terms:
                st.markdown(
                    f"<div style='margin-top:10px; font-weight:700;'>Top terms:</div>"
                    f"<div class='chips'>{' '.join([f'<span>{t}</span>' for t in top_terms])}</div>",
                    unsafe_allow_html=True
                )
            if samples:
                st.markdown("<div style='margin-top:10px; font-weight:700;'>Samples:</div>", unsafe_allow_html=True)
                for s in samples:
                    st.markdown(f"<div class='sample-item'>• {_strip_meta(s)}</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)  # close card

else:
    st.info("No suggestions yet. Click **Run analysis** in the sidebar to generate them.")

# ────────────────────────────────────────────────────────────────────────────────
# Footer section — kept reviews preview (optional)
# ────────────────────────────────────────────────────────────────────────────────
if st.session_state.get("df_clean") is not None:
    st.markdown('<div class="section-title">📄 Kept reviews (sample)</div>', unsafe_allow_html=True)
    st.dataframe(st.session_state["df_clean"].head(200))
    st.caption("Pipeline run complete. Replace any placeholders with production integrations as needed.")