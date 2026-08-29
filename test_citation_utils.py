# test_citation_utils.py
from citation_utils import *

# Sample citations from your RAG system
sample_citations = [
    {
        "id": 1,
        "source": "الإستعلام عن الرصيد",
        "text": "يمكنك الإستعلام عن رصيدك من خلال الإتصال بـ #550* وسوف تصلك رسالة مؤقتة تظهر لك المبلغ المتبقي من رصيدك.",
        "relevance_score": 0.8240
    },
    {
        "id": 2,
        "source": "التنبيه عند الوصول",
        "text": "يمكنك معرفة الأرقام الغير متاحة أو مغلقة واصبحت متاحة عن طريق رسالة نصية لإخطارك بمعاودة الإتصال.",
        "relevance_score": 0.8036
    }
]

# Test various formatting functions
print("="*60)
print("TESTING CITATION UTILITIES")
print("="*60)

print("\n1. SIMPLE FORMAT:")
print(format_citations_simple(sample_citations))

print("\n2. HTML FORMAT:")
print(format_citations_html(sample_citations))

print("\n3. MARKDOWN FORMAT:")
print(format_citations_markdown(sample_citations))

print("\n4. CITATION SUMMARY:")
print_citation_summary(sample_citations)

print("\n5. MERGE METADATA:")
result = {
    "response": "يمكنك الاستعلام عن رصيدك من خلال الاتصال بـ #550* [1]",
    "citations": sample_citations
}
merged = merge_citation_metadata(result)
print(f"Response: {merged['response']}")
print(f"Citation IDs found: {merged['citation_ids']}")
print(f"Total citations: {merged['citation_count']}")