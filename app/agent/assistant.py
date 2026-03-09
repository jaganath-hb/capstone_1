from dotenv import load_dotenv
load_dotenv()

import os
import json
import re
from typing import TypedDict, List, Dict, Any

from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END


# ------------------------------------------------
# Agent State
# ------------------------------------------------

class AgentState(TypedDict):
    cluster_summaries: str
    improvements: List[Dict[str, Any]]


# ------------------------------------------------
# Azure LLM
# ------------------------------------------------

llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    api_version="2024-02-15-preview",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_KEY"],
)


# ------------------------------------------------
# Prompt
# ------------------------------------------------

PROMPT = """
You are a product improvement assistant.

Based on the cluster summaries, generate the TOP 5 product improvement actions.

Rules:
- Action must be MAX 12 words
- Rationale must be MAX 15 words
- Be concise
- No paragraphs
- No numbering
- No explanations

Cluster summaries:
{cluster_summaries}

Return ONLY JSON in this format:

{{
 "improvements":[
  {{"action":"short action","rationale":"short reason","priority":1}},
  {{"action":"short action","rationale":"short reason","priority":2}},
  {{"action":"short action","rationale":"short reason","priority":3}},
  {{"action":"short action","rationale":"short reason","priority":4}},
  {{"action":"short action","rationale":"short reason","priority":5}}
 ]
}}
"""


# ------------------------------------------------
# JSON Parser
# ------------------------------------------------

def parse_json(text: str):

    if not text:
        return []

    text = text.strip()

    # remove markdown if present
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)

    try:
        data = json.loads(text)

        if isinstance(data, dict) and "improvements" in data:
            return data["improvements"]

        return data

    except Exception:

        # fallback extraction
        match = re.search(r"\{.*\}", text, re.DOTALL)

        if match:
            try:
                data = json.loads(match.group(0))

                if "improvements" in data:
                    return data["improvements"]

            except Exception:
                pass

    print("\nRAW LLM OUTPUT:\n", text)

    return []


# ------------------------------------------------
# Agent Node
# ------------------------------------------------

def improvement_agent(state: AgentState):

    prompt = PROMPT.format(
        cluster_summaries=state["cluster_summaries"]
    )

    response = llm.invoke(prompt)

    improvements = parse_json(response.content)

    return {
        "improvements": improvements
    }


# ------------------------------------------------
# Build Graph
# ------------------------------------------------

graph = StateGraph(AgentState)

graph.add_node("improvement_agent", improvement_agent)

graph.set_entry_point("improvement_agent")

graph.add_edge("improvement_agent", END)

app = graph.compile()


# ------------------------------------------------
# Function for Pipeline
# ------------------------------------------------

def propose_improvements(cluster_summaries: str):

    result = app.invoke({
        "cluster_summaries": cluster_summaries
    })

    return result.get("improvements", [])
