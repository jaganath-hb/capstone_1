import os
import json
import re
import time
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple

import openai  # Assumes the legacy openai package; if you use the new SDK, see note below.

PROMPT_TEMPLATE = """
You are a product improvement assistant. Given cluster summaries, list top 5 actionable product improvements, prioritized, each with a one-line rationale.

Cluster summaries:
{cluster_summaries}

Return a JSON array of items: [{{"action":"...","rationale":"...","priority":1}}]
"""

# ---------------------------
# Utilities
# ---------------------------

def _configure_openai_for_azure(key: str, base: str, deployment: str, api_version: str = "2023-05-15"):
    openai.api_type = "azure"
    openai.api_key = key
    openai.api_base = base
    openai.api_version = api_version
    return deployment

def _try_parse_json(text: str) -> Any:
    """
    Attempt to extract and parse a JSON array/dict from a possibly messy LLM response.
    - Strips code fences
    - Finds first JSON array/dict via regex
    """
    def _strip_fences(s: str) -> str:
        s = s.strip()
        if s.startswith("```"):
            s = re.sub(r"^```(?:json)?", "", s, flags=re.IGNORECASE).strip()
        if s.endswith("```"):
            s = s[: s.rfind("```")].strip()
        return s

    s = _strip_fences(text)

    # Quick path
    try:
        return json.loads(s)
    except Exception:
        pass

    # Try to locate first JSON array or object in the text
    m = re.search(r"(\[.*\]|\{.*\})", s, flags=re.DOTALL)
    if m:
        candidate = m.group(1)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    raise ValueError("Could not parse JSON from model output")

# ---------------------------
# Offline heuristic generator
# ---------------------------

_SEVERITY_KEYWORDS = {
    "crash": 4, "outage": 4, "data loss": 5, "security": 5, "breach": 5, "leak": 5,
    "fail": 3, "failure": 3, "error": 3, "bug": 3, "latency": 3, "slow": 2, "timeout": 3,
    "broken": 3, "corrupt": 4, "freeze": 3, "hang": 3, "infinite loop": 4
}
_IMPACT_KEYWORDS = {
    "payment": 4, "checkout": 4, "login": 4, "onboarding": 3, "signup": 3, "billing": 4,
    "export": 2, "reporting": 2, "notifications": 2, "search": 2, "sync": 3, "mobile": 3
}
_RECENCY_KEYWORDS = {"latest": 1, "new release": 1, "recent": 1, "v2.": 1, "v3.": 1, "today": 1}

_STOPWORDS = set("""
a an the and or but if while to for of on in with by from at as is are was were be been being this that these those
""".split())

def _tokenize_lines(text: str) -> List[str]:
    lines = [ln.strip("•-* \t") for ln in text.splitlines()]
    return [ln for ln in lines if ln]

def _score_line(line: str) -> int:
    s = line.lower()
    score = 0
    for k, w in _SEVERITY_KEYWORDS.items():
        if k in s:
            score += w
    for k, w in _IMPACT_KEYWORDS.items():
        if k in s:
            score += w
    for k, w in _RECENCY_KEYWORDS.items():
        if k in s:
            score += w
    # length heuristic (not too short, not too long)
    n = len(s.split())
    if 5 <= n <= 25:
        score += 1
    return score

def _normalize_action_phrase(s: str) -> str:
    s = s.strip().rstrip(".")
    s_low = s.lower()
    # Map common complaint patterns -> action verbs
    replacements = [
        (r"(login|authentication|auth).*fail", "Improve login reliability"),
        (r"(payment|checkout).*fail", "Fix checkout/payment failures"),
        (r"(crash|freeze|hang)", "Fix app stability issues"),
        (r"(latency|slow|timeout|performance)", "Reduce latency and improve performance"),
        (r"(onboarding|signup|registration)", "Streamline onboarding flow"),
        (r"(security|breach|leak|vuln)", "Address security vulnerabilities"),
        (r"(report|export)", "Stabilize reporting/export workflows"),
        (r"(sync|synchronization)", "Improve data sync reliability"),
        (r"(mobile|iOS|android)", "Fix critical mobile issues"),
        (r"(search)", "Improve search relevance and speed"),
    ]
    for pat, action in replacements:
        if re.search(pat, s_low):
            return action

    # Fallback: compress to a clear imperative starting with a verb
    # Keep nouns and verbs only (very rough)
    cleaned = re.sub(r"[^a-zA-Z0-9 ]+", " ", s).strip()
    words = [w for w in cleaned.split() if w.lower() not in _STOPWORDS]
    if not words:
        words = ["issue"]
    # Start with a generic verb
    return "Improve " + " ".join(words[:6])

def _group_similar(lines_scored: List[Tuple[str, int]]) -> Dict[str, Dict[str, Any]]:
    """
    Bucket lines that map to the same normalized action. Aggregate score & examples.
    """
    buckets: Dict[str, Dict[str, Any]] = {}
    for line, score in lines_scored:
        action = _normalize_action_phrase(line)
        b = buckets.setdefault(action, {"action": action, "score": 0, "examples": [], "count": 0})
        b["score"] += score
        b["count"] += 1
        if len(b["examples"]) < 3:
            b["examples"].append(line)
    return buckets

def _rank_to_priorities(items: List[Dict[str, Any]]) -> List[int]:
    """
    Convert rank (0..n-1) to priority scale (1 is highest). Ties handled by order.
    """
    return list(range(1, len(items) + 1))

def _offline_improvement_proposals(cluster_summaries: str) -> List[Dict[str, Any]]:
    lines = _tokenize_lines(cluster_summaries)
    if not lines:
        return [{"action": "Clarify user pain points", "rationale": "No cluster summaries provided.", "priority": 1}]

    scored = [(ln, _score_line(ln)) for ln in lines]
    # Boost frequently repeated problems
    freq = Counter([re.sub(r"\s+", " ", ln.lower()).strip() for ln in lines])
    scored = [(ln, sc + min(3, freq[re.sub(r'\s+', ' ', ln.lower()).strip()])) for ln, sc in scored]

    buckets = _group_similar(scored)
    ranked = sorted(buckets.values(), key=lambda x: (x["score"], x["count"]), reverse=True)

    # Compose final list with rationales
    top = ranked[:5]
    priorities = _rank_to_priorities(top)

    results = []
    for i, item in enumerate(top):
        examples = "; ".join(item["examples"])
        rationale = f"High impact and severity indicated by {item['count']} related reports (e.g., {examples})."
        results.append({"action": item["action"], "rationale": rationale, "priority": priorities[i]})
    return results

# ---------------------------
# Main entry
# ---------------------------

def propose_improvements(cluster_summaries: str):
    """Call an LLM to propose improvements.

    If Azure/OpenAI is configured, it uses the model.
    Otherwise, it falls back to a local heuristic generator that analyzes the text
    and produces the top 5 actionable improvements with rationales and priorities.
    """
    azure_key = os.environ.get("AZURE_OPENAI_KEY")
    azure_base = os.environ.get("AZURE_OPENAI_BASE")
    azure_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    azure_api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2023-05-15")

    prompt = PROMPT_TEMPLATE.format(cluster_summaries=cluster_summaries)
    model_name = os.environ.get("AZURE_OPENAI_MODEL", "gpt-5-mini")

    # Try Azure first
    if azure_key and azure_base and azure_deployment:
        try:
            deployment = _configure_openai_for_azure(azure_key, azure_base, azure_deployment, azure_api_version)
            messages = [
                {"role": "system", "content": "You are a helpful product improvement assistant."},
                {"role": "user", "content": prompt},
            ]
            resp = openai.ChatCompletion.create(
                engine=deployment,
                messages=messages,
                temperature=0.2,
                max_tokens=600,
                request_timeout=30,
            )
            text = resp.choices[0].message.content
            try:
                return _try_parse_json(text)
            except Exception:
                # If parsing fails, still provide *automatic* suggestions offline
                return _offline_improvement_proposals(cluster_summaries)
        except Exception:
            # If the API call fails for any reason, use offline heuristic suggestions
            return _offline_improvement_proposals(cluster_summaries)

    # Try non-Azure OpenAI
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        try:
            openai.api_key = openai_key
            messages = [
                {"role": "system", "content": "You are a helpful product improvement assistant."},
                {"role": "user", "content": prompt},
            ]
            resp = openai.ChatCompletion.create(
                model=model_name,
                messages=messages,
                temperature=0.2,
                max_tokens=600,
                request_timeout=30,
            )
            text = resp.choices[0].message.content
            try:
                return _try_parse_json(text)
            except Exception:
                return _offline_improvement_proposals(cluster_summaries)
        except Exception:
            return _offline_improvement_proposals(cluster_summaries)

    # No credentials present → offline automatic suggestions
    return _offline_improvement_proposals(cluster_summaries)