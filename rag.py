import os
import json
import re
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from pathlib import Path
#from dotenv import load_dotenv
#load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "REDACTED_OPENROUTER_API_KEY")
# ==========================================
# 0. Suppress warnings and use existing cache
# ==========================================
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # Disable HF transfer

# ==========================================
# 1. Load model from existing cache
# ==========================================

# Check if model exists in cache
cache_path = Path.home() / ".cache" / "huggingface" / "hub"
model_name = "intfloat/multilingual-e5-base"

if cache_path.exists():
    # Try to find existing model files
    model_files = list(cache_path.glob(f"*{model_name.replace('/', '_')}*"))
    if model_files:
        print(f"Found existing model cache at: {model_files[0]}")
        # Load model (will use cache if available)
        model = SentenceTransformer(model_name, cache_folder=str(cache_path))
    else:
        print("Model not found in cache, will download (this is one-time)")
        model = SentenceTransformer(model_name)
else:
    print("Cache path not found, will download")
    model = SentenceTransformer(model_name)

# ==========================================
# 2. Load saved chunks + embeddings
# ==========================================

with open("embeddings.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# ==========================================
# 3. RAG System with Citations
# ==========================================

class RAGWithCitations:
    def __init__(self, chunks: List[Dict], model):
        self.chunks = chunks
        self.model = model
        self.api_key = os.getenv("OPENROUTER_API_KEY", "REDACTED_OPENROUTER_API_KEY")
        
    def retrieve_top_k(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve top-k relevant chunks with citation info"""
        # Embed the question
        question_embedding = self.model.encode(
            "query: " + query,
            normalize_embeddings=True
        )
        
        # Calculate similarities
        results = []
        for chunk in self.chunks:
            chunk_embedding = np.array(chunk["embedding"])
            similarity = np.dot(question_embedding, chunk_embedding)
            results.append({
                "chunk": chunk,
                "score": float(similarity),
                "source": chunk.get("title", "Unknown")
            })
        
        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top-k
        return results[:k]
    
    def prepare_context_with_citations(self, retrieved: List[Dict]) -> tuple:
        """Format retrieved chunks with citation markers"""
        context_with_citations = []
        citations = []
        
        for i, result in enumerate(retrieved, 1):
            chunk_text = result["chunk"]["text"]
            source = result["source"]
            score = result["score"]
            
            # Add citation marker
            context_with_citations.append(f"[{i}] {chunk_text}")
            
            citations.append({
                "id": i,
                "source": source,
                "text": chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
                "relevance_score": round(score, 4)
            })
        
        return context_with_citations, citations
    
    def detect_language(self, text: str) -> str:
        """Detect if text is Arabic or English"""
        # Check for Arabic characters
        if any('\u0600' <= c <= '\u06FF' for c in text):
            return "arabic"
        return "english"
    
    def generate_response(self, query: str, context_with_citations: List[str], citations: List[Dict]) -> Dict[str, Any]:
        """Generate response with citations using LLM"""
        
        context_str = "\n\n".join(context_with_citations)
        lang = self.detect_language(query)
        
        # Build prompt based on language
        if lang == "arabic":
            prompt = f"""أنت مساعد Telecom Egypt الذكي. أجب عن سؤال المستخدم بناءً على السياق المقدم فقط.

السياق (مع الاستشهادات):
{context_str}

سؤال المستخدم: {query}

تعليمات:
1. أجب فقط بناءً على السياق المقدم
2. استخدم الاستشهادات مثل [1]، [2] للإشارة إلى المصادر
3. إذا لم يحتوي السياق على الإجابة، قل "لا يمكنني العثور على هذه المعلومات في المستندات المتوفرة"
4. كن مفيداً ومحترفاً

الإجابة:"""
        else:
            prompt = f"""You are Telecom Egypt's intelligent assistant. Answer the user's question based ONLY on the provided context.

Context (with citations):
{context_str}

User Question: {query}

Instructions:
1. Answer based ONLY on the context provided
2. Use citations like [1], [2] to reference sources
3. If the context doesn't contain the answer, say "I cannot find this information in the provided documents"
4. Be helpful and professional

Answer:"""
        
        # Call LLM via OpenRouter
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
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
        
        if response.status_code == 200:
            data = response.json()
            answer = data["choices"][0]["message"]["content"]
        else:
            answer = f"Error: {response.status_code} - {response.text}"
        
        # Extract which citations were used in the response
        citation_pattern = r'\[(\d+)\]'
        used_citation_ids = set(map(int, re.findall(citation_pattern, answer)))
        used_citations = [c for c in citations if c['id'] in used_citation_ids]
        
        return {
            "query": query,
            "response": answer,
            "citations": used_citations,
            "all_citations": citations,
            "language_detected": lang
        }
    
    def query(self, question: str, k: int = 5) -> Dict[str, Any]:
        """Complete RAG pipeline with citations"""
        # Step 1: Retrieve top-k relevant chunks
        retrieved = self.retrieve_top_k(question, k=k)
        
        # Step 2: Prepare context with citations
        context_with_citations, citations = self.prepare_context_with_citations(retrieved)
        
        # Step 3: Generate response with citations
        result = self.generate_response(question, context_with_citations, citations)
        
        # Add retrieval info
        result["retrieved_chunks"] = retrieved
        
        return result

# ==========================================
# 4. Main execution
# ==========================================

if __name__ == "__main__":
    # Initialize the RAG system
    rag = RAGWithCitations(chunks, model)
    
    # Get user question
    question = input("Ask your question: ")
    
    # Query the system
    print("\n" + "="*60)
    print("SEARCHING FOR RELEVANT INFORMATION...")
    print("="*60 + "\n")
    
    result = rag.query(question, k=5)  # Get top 5 chunks
    
    # Print retrieved chunks
    print("RETRIEVED CHUNKS (Top 5):")
    for i, chunk_info in enumerate(result["retrieved_chunks"], 1):
        print(f"\n[{i}] Source: {chunk_info['source']}")
        print(f"    Score: {chunk_info['score']:.4f}")
        print(f"    Preview: {chunk_info['chunk']['text'][:150]}...")
    
    print("\n" + "="*60)
    print("LLM RESPONSE:")
    print("="*60 + "\n")
    print(result["response"])
    
    # Print citations
    if result["citations"]:
        print("\n" + "="*60)
        print("CITATIONS USED:")
        print("="*60)
        for citation in result["citations"]:
            print(f"\n[{citation['id']}] Source: {citation['source']}")
            print(f"    Relevance: {citation['relevance_score']:.4f}")
            print(f"    Excerpt: {citation['text']}")
    else:
        print("\nNo citations found in the response.")