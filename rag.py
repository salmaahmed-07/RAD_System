import os
import json
import numpy as np
import requests
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. Load E5 embedding model
# ==========================================

model = SentenceTransformer(
    "intfloat/multilingual-e5-base"
)


# ==========================================
# 2. Load saved chunks + embeddings
# ==========================================

with open("embeddings.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)


# ==========================================
# 3. Ask the customer
# ==========================================

question = input("Ask your question: ")


# ==========================================
# 4. Embed the question
# ==========================================

question_embedding = model.encode(
    "query: " + question,
    normalize_embeddings=True
)


# ==========================================
# 5. Find the most relevant chunk
# ==========================================

best_chunk = None
best_score = -1

for chunk in chunks:

    chunk_embedding = np.array(
        chunk["embedding"]
    )

    similarity = np.dot(
        question_embedding,
        chunk_embedding
    )

    if similarity > best_score:
        best_score = similarity
        best_chunk = chunk


# ==========================================
# 6. Show what was retrieved
# ==========================================

print("\nRetrieved information:\n")

print("Title:", best_chunk["title"])
print("Score:", round(best_score, 4))
print("Text:", best_chunk["text"])


# ==========================================
# 7. Send retrieved information to LLM
# ==========================================

api_key = os.getenv("OPENROUTER_API_KEY")

prompt = f"""
أنت موظف خدمة عملاء لشركة WE.

أجب عن سؤال العميل باستخدام المعلومات الموجودة في السياق فقط.

إذا كانت الإجابة غير موجودة في السياق، قل إن المعلومات المتاحة لا تكفي للإجابة.

لا تخترع أي معلومات.

السياق:

{best_chunk["title"]}
{best_chunk["text"]}

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


# ==========================================
# 8. Print the final answer
# ==========================================

data = response.json()

if response.status_code == 200:

    answer = data["choices"][0]["message"]["content"]

    print("\nLLM Answer:\n")
    print(answer)

else:

    print("\nLLM Error:")
    print(data)