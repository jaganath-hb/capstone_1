from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv
from app.agent.assistant import propose_improvements

test_cases = [
    "Cluster 0: many iOS crashes; Cluster 1: onboarding confusing",
    "Cluster 0: battery drains fast; Cluster 1: app slow; Cluster 2: login fails sometimes",
    "Cluster 0: users love the UI; Cluster 1: app performance is very fast",
    "",
    "Cluster 0: payment gateway failing during checkout",
    "Cluster 0: users report frequent crashes when opening the camera module",
    "Cluster 0: UI good but app crashes sometimes; Cluster 1: onboarding confusing",
    "Cluster 0: aplicación se bloquea en iOS; Cluster 1: proceso de registro confuso",
    "Cluster 0: users report password reset not working",
    "Cluster 0: app loading very slow; Cluster 1: search results take too long"
]

for i, s in enumerate(test_cases):
    print(f"\nTest Case {i+1}")
    try:
        res = propose_improvements(s)
        print(res)
    except Exception as e:
        print("Error:", e)
