import os
import requests

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    print("API key not found.")
    exit()

response = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    },
    json={
        "model": "openai/gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": "Say hello in one short sentence."
            }
        ]
    }
)

print("Status code:", response.status_code)
print(response.json())