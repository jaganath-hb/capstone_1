import os
import json
import openai

PROMPT_TEMPLATE = """
You are a product improvement assistant. Given cluster summaries, list top 5 actionable product improvements, prioritized, each with a one-line rationale.

Cluster summaries:
{cluster_summaries}

Return a JSON array of items: [{{"action":"...","rationale":"...","priority":1}}]
"""


def _configure_openai_for_azure(key: str, base: str, deployment: str, api_version: str = "2023-05-15"):
    openai.api_type = "azure"
    openai.api_key = key
    openai.api_base = base
    openai.api_version = api_version
    return deployment


def propose_improvements(cluster_summaries: str):
    """Call an LLM to propose improvements.

    Supports Azure OpenAI when `AZURE_OPENAI_KEY` and `AZURE_OPENAI_BASE` are set.
    Falls back to returning a small static suggestion list if no API keys are configured.
    """
    azure_key = os.environ.get("AZURE_OPENAI_KEY")
    azure_base = os.environ.get("AZURE_OPENAI_BASE")
    azure_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    azure_api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2023-05-15")

    prompt = PROMPT_TEMPLATE.format(cluster_summaries=cluster_summaries)
    model_name = os.environ.get("AZURE_OPENAI_MODEL", "gpt-5-mini")

    if azure_key and azure_base and azure_deployment:
        try:
            deployment = _configure_openai_for_azure(azure_key, azure_base, azure_deployment, azure_api_version)
            messages = [{"role": "system", "content": "You are a helpful product improvement assistant."},
                        {"role": "user", "content": prompt}]
            # For Azure OpenAI the `engine`/`deployment` must match a deployment configured in the resource.
            resp = openai.ChatCompletion.create(engine=deployment, messages=messages, temperature=0.2, max_tokens=600)
            text = resp.choices[0].message.content
            # Attempt to parse JSON from the model output
            try:
                return json.loads(text)
            except Exception:
                # try to recover if the model returns code block
                cleaned = text.strip().strip('```')
                return json.loads(cleaned)
        except Exception:
            # On any error, fall back to static suggestions
            pass

    # If Azure credentials weren't provided try direct OpenAI API using `model_name`.
    try:
        # Use regular OpenAI model name if API key is configured (non-Azure)
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            openai.api_key = openai_key
            messages = [{"role": "system", "content": "You are a helpful product improvement assistant."},
                        {"role": "user", "content": prompt}]
            resp = openai.ChatCompletion.create(model=model_name, messages=messages, temperature=0.2, max_tokens=600)
            text = resp.choices[0].message.content
            try:
                return json.loads(text)
            except Exception:
                cleaned = text.strip().strip('```')
                return json.loads(cleaned)
    except Exception:
        pass

    # Fallback static suggestions (useful for offline dev)
    return [
        {"action": "Improve onboarding flow", "rationale": "Many users complain about confusion during signup.", "priority": 1},
        {"action": "Fix login error on iOS", "rationale": "Crash reports and negative reviews reference iOS login failure.", "priority": 2},
    ]
