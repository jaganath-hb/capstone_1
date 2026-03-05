from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv(dotenv_path=Path('C:/Jagan/Project/Test/.env'))
from app.agent.assistant import propose_improvements

s = 'Cluster 0: many iOS crashes; Cluster 1: onboarding confusing'
print('Using envs:', os.environ.get('AZURE_OPENAI_BASE'), os.environ.get('AZURE_OPENAI_DEPLOYMENT'), os.environ.get('AZURE_OPENAI_KEY') is not None)
try:
    res = propose_improvements(s)
    print('Result:')
    print(res)
except Exception as e:
    print('Error calling propose_improvements:', e)
