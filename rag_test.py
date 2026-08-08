import os
import requests

api_key = os.getenv("OPENROUTER_API_KEY")

question = "ازاي اعرف رصيدي؟"

retrieved_context = """
الإستعلام عن الرصيد:
يمكنك الإستعلام عن رصيدك من خلال الإتصال بـ #550* وسوف تصلك رسالة مؤقتة تظهر لك المبلغ المتبقي من رصيدك.
"""

prompt = f"""
أنت موظف خدمة عملاء لشركة WE.

أجب عن سؤال العميل باستخدام المعلومات الموجودة في السياق فقط.

إذا كانت الإجابة غير موجودة في السياق، قل إن المعلومات المتاحة لا تكفي للإجابة.

لا تخترع أي معلومات.

السياق:
{retrieved_context}

سؤال العميل:
{question}

اكتب إجابة قصيرة وواضحة للعميل.
"""

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
                "content": prompt
            }
        ]
    }
)

data = response.json()

if response.status_code == 200:
    answer = data["choices"][0]["message"]["content"]
    print("\nLLM Answer:\n")
    print(answer)
else:
    print("Error:", response.status_code)
    print(data)