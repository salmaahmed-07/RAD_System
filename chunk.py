import re

# Read the Markdown file
with open("mobile_services.md", "r", encoding="utf-8") as f:
    content = f.read()

# Split whenever we reach a ## heading
sections = re.split(r"(?=^## )", content, flags=re.MULTILINE)

chunks = []

for section in sections:
    section = section.strip()

    # Ignore the main "# خدمات الموبايل" heading
    if not section.startswith("## "):
        continue

    # Extract the service title
    lines = section.split("\n")
    title = lines[0].replace("## ", "").strip()

    # Everything after the title is the service content
    text = "\n".join(lines[1:]).strip()

    chunks.append({
        "title": title,
        "text": text
    })

# Display the chunks
print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks, 1):
    print(f"\n--- Chunk {i} ---")
    print("Title:", chunk["title"])
    print("Text:", chunk["text"])


import json

with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print("\nChunks saved to chunks.json")