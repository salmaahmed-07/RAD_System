# test_api_key.py
import os
import requests

# Replace this with your actual API key or set via environment
API_KEY = os.getenv("OPENROUTER_API_KEY", "REDACTED_OPENROUTER_API_KEY")

print("Testing API key...")
print(f"API Key length: {len(API_KEY)}")
print(f"API Key starts with: {API_KEY[:10]}...")

# Test the API
response = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    },
    json={
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "Say hello"}
        ]
    }
)

print(f"\nStatus Code: {response.status_code}")
print(f"Response: {response.text}")