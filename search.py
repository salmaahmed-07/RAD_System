import json
import numpy as np
from sentence_transformers import SentenceTransformer

print("STEP 1: Model loading...")

model = SentenceTransformer(
    "intfloat/multilingual-e5-base"
)

print("STEP 2: Model loaded")

# 2. Load our saved embeddings

with open("embeddings.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# 3. Ask the user a question

question = input("Ask your question: ")

# 4. Convert the question into an embedding
# E5 requires the "query:" prefix for questions

question_embedding = model.encode(
    "query: " + question,
    normalize_embeddings=True
)

# 5. Calculate similarity with every chunk

results = []

for chunk in chunks:

    chunk_embedding = np.array(chunk["embedding"])

    # Because both embeddings are normalized,
    # dot product = cosine similarity

    similarity = np.dot(
        question_embedding,
        chunk_embedding
    )

    results.append({
        "title": chunk["title"],
        "text": chunk["text"],
        "score": similarity
    })

# 6. Sort from most similar to least similar

results.sort(
    key=lambda x: x["score"],
    reverse=True
)

# 7. Show the top 3 results

print("\nMost relevant results:\n")

for i, result in enumerate(results[:3], 1):

    print(f"--- Result {i} ---")
    print("Title:", result["title"])
    print("Score:", round(result["score"], 4))
    print("Text:", result["text"])
    print()