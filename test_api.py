import requests
import os
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Request list of models
res = requests.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}")
print("Available Models:")
for m in res.json().get('models', []):
    print(m['name'])

# Let's test just gemini-pro if flashing fails
try:
    res = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}",
        headers={"Content-Type": "application/json"},
        json={"contents": [{"parts": [{"text": "hey"}]}]}
    )
    print("\nGemini Flash Response:", res.status_code, res.text)
except Exception as e:
    pass
