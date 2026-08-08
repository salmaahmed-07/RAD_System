import json
from sentence_transformers import SentenceTransformer

# Load the embedding model

model = SentenceTransformer(
    "intfloat/multilingual-e5-base"
)

# Load our chunks

with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Prepare the text we want to embed

texts = []

for chunk in chunks:
    text = f"passage: {chunk['title']}\n{chunk['text']}"
    texts.append(text)

# Create embeddings for all chunks

embeddings = model.encode(
    texts,
    normalize_embeddings=True
)

print("Number of chunks:", len(chunks))
print("Number of embeddings:", len(embeddings))
print("Vector size:", len(embeddings[0]))

# Save everything together

for i, chunk in enumerate(chunks):
    chunk["embedding"] = embeddings[i].tolist()

with open("embeddings.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print("Embeddings saved to embeddings.json")