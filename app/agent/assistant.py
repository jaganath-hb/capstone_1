from dotenv import load_dotenv
load_dotenv()

import os
import json
import re
from typing import Any
from openai import AzureOpenAI


# ---------------------------------------------------
# Prompt Template
# ---------------------------------------------------

PROMPT_TEMPLATE = """
You are a product improvement assistant.

Given the cluster summaries below, generate the TOP 5 actionable product improvements.

Rules:
- Prioritize by impact
- Provide clear action
- Give one-line rationale
- Assign priority (1 = highest)

Cluster summaries:
{cluster_summaries}

Return ONLY JSON in this format:

{{
 "improvements":[
  {{"action":"...","rationale":"...","priority":1}},
  {{"action":"...","rationale":"...","priority":2}},
  {{"action":"...","rationale":"...","priority":3}},
  {{"action":"...","rationale":"...","priority":4}},
  {{"action":"...","rationale":"...","priority":5}}
 ]
}}
"""


# ---------------------------------------------------
# Robust JSON Parser
# ---------------------------------------------------

def parse_json_response(text: str) -> Any:

    if not text:
        raise ValueError("Empty response from model")

    text = text.strip()

    # remove markdown code blocks
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)

    try:
        data = json.loads(text)

        if isinstance(data, dict) and "improvements" in data:
            return data["improvements"]

        return data

    except json.JSONDecodeError:
        pass

    # try extracting JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if match:
        try:
            data = json.loads(match.group(0))

            if isinstance(data, dict) and "improvements" in data:
                return data["improvements"]

            return data

        except Exception:
            pass

    print("\n----- RAW MODEL OUTPUT -----\n")
    print(text)
    print("\n----------------------------\n")

    raise ValueError("Could not parse JSON from model output")


# ---------------------------------------------------
# Azure Client
# ---------------------------------------------------

def create_client():

    return AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_KEY"],
        api_version="2024-02-15-preview",
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"]
    )


# ---------------------------------------------------
# Main LLM Function
# ---------------------------------------------------

def propose_improvements(cluster_summaries: str):

    client = create_client()

    deployment_name = os.environ["AZURE_OPENAI_DEPLOYMENT"]

    prompt = PROMPT_TEMPLATE.format(
        cluster_summaries=cluster_summaries
    )

    try:

        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful product improvement assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_completion_tokens=600
        )

        message = response.choices[0].message
        text = message.content if message and message.content else ""

        if not text:
            print("\n⚠️ Model returned empty content")
            print(response)

            return [
                {
                    "action": "Investigate customer complaint clusters",
                    "rationale": "Model returned empty response",
                    "priority": 1
                }
            ]

        print("\nLLM OUTPUT:\n", text)

        return parse_json_response(text)

    except Exception as e:

        print("\n⚠️ LLM call failed:", str(e))

        return [
            {
                "action": "Review major customer complaints manually",
                "rationale": "LLM processing failed",
                "priority": 1
            }
        ]
