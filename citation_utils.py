# citation_utils.py
import hashlib
import re
from typing import List, Dict, Any
from datetime import datetime

def generate_citation_id(text: str, source: str) -> str:
    """
    Generate a unique citation ID based on content and source.
    Useful for tracking citations across sessions.
    """
    content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
    source_hash = hashlib.md5(source.encode()).hexdigest()[:8]
    return f"cit_{source_hash}_{content_hash}"

def extract_citations_from_response(response_text: str) -> List[int]:
    """
    Extract citation numbers from the LLM response.
    Returns list of citation IDs used.
    """
    pattern = r'\[(\d+)\]'
    matches = re.findall(pattern, response_text)
    return [int(m) for m in matches]

def format_citation_text(citation: Dict[str, Any]) -> str:
    """
    Format a single citation as a readable string.
    """
    return f"[{citation['id']}] Source: {citation['source']} (Relevance: {citation['relevance_score']:.4f})\n    {citation['text']}"

def format_citations_html(citations: List[Dict[str, Any]]) -> str:
    """
    Format citations as HTML for display in Streamlit/Gradio.
    """
    if not citations:
        return ""
    
    html = """
    <div style="background-color: #f5f5f5; padding: 15px; border-radius: 8px; margin-top: 10px; border-left: 4px solid #ff9800;">
        <h4 style="margin-top: 0;">📚 Sources Used:</h4>
        <ul style="list-style-type: none; padding-left: 0;">
    """
    
    for citation in citations:
        html += f"""
        <li style="margin-bottom: 10px; padding: 10px; background-color: white; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <strong>[{citation['id']}]</strong>
            <span style="color: #e65100; font-weight: bold;">{citation['source']}</span>
            <span style="color: #666; font-size: 0.9em;">(Relevance: {citation['relevance_score']:.4f})</span>
            <br>
            <span style="color: #333; font-size: 0.95em;">{citation['text']}</span>
        </li>
        """
    
    html += """
        </ul>
    </div>
    """
    return html

def format_citations_markdown(citations: List[Dict[str, Any]]) -> str:
    """
    Format citations as Markdown for display in Jupyter notebooks.
    """
    if not citations:
        return ""
    
    md = "\n## 📚 Sources Used\n\n"
    for citation in citations:
        md += f"**[{citation['id']}]** *{citation['source']}* "
        md += f"(Relevance: {citation['relevance_score']:.4f})\n\n"
        md += f"> {citation['text']}\n\n"
    return md

def format_citations_simple(citations: List[Dict[str, Any]]) -> str:
    """
    Format citations as simple text for console output.
    """
    if not citations:
        return "No citations found."
    
    output = "\n" + "="*60 + "\n"
    output += "📚 SOURCES USED:\n"
    output += "="*60 + "\n"
    
    for citation in citations:
        output += f"\n[{citation['id']}] Source: {citation['source']}\n"
        output += f"    Relevance: {citation['relevance_score']:.4f}\n"
        output += f"    {citation['text']}\n"
    
    return output

def merge_citation_metadata(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge all citation-related metadata into a single object.
    """
    return {
        "response": response.get("response", ""),
        "citation_ids": extract_citations_from_response(response.get("response", "")),
        "citations": response.get("citations", []),
        "all_context": response.get("all_citations", []),
        "citation_count": len(response.get("citations", [])),
        "timestamp": datetime.now().isoformat()
    }

def get_citation_by_id(citations: List[Dict[str, Any]], citation_id: int) -> Dict[str, Any]:
    """
    Get a specific citation by its ID.
    """
    for citation in citations:
        if citation["id"] == citation_id:
            return citation
    return None

def create_citation_summary(citations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create a summary of citations including:
    - Total citations used
    - Unique sources
    - Average relevance score
    """
    if not citations:
        return {
            "total": 0,
            "unique_sources": [],
            "avg_relevance": 0.0,
            "max_relevance": 0.0,
            "min_relevance": 0.0
        }
    
    sources = list(set([c["source"] for c in citations]))
    scores = [c["relevance_score"] for c in citations]
    
    return {
        "total": len(citations),
        "unique_sources": sources,
        "avg_relevance": sum(scores) / len(scores),
        "max_relevance": max(scores),
        "min_relevance": min(scores)
    }

def print_citation_summary(citations: List[Dict[str, Any]]) -> None:
    """
    Print a human-readable summary of citations.
    """
    summary = create_citation_summary(citations)
    
    print("\n" + "="*60)
    print("📊 CITATION SUMMARY")
    print("="*60)
    print(f"Total Citations Used: {summary['total']}")
    print(f"Unique Sources: {len(summary['unique_sources'])}")
    if summary['total'] > 0:
        print(f"Average Relevance: {summary['avg_relevance']:.4f}")
        print(f"Highest Relevance: {summary['max_relevance']:.4f}")
        print(f"Lowest Relevance: {summary['min_relevance']:.4f}")
        print("\nSources:")
        for source in summary['unique_sources']:
            print(f"  - {source}")
    print("="*60 + "\n")