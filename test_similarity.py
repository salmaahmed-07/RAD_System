from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

query = "ازاي اعرف رصيدي؟"

documents = {
    "Correct - الإستعلام عن الرصيد":
        "يمكنك الإستعلام عن رصيدك من خلال الإتصال بـ #550* وسوف تصلك رسالة مؤقتة تظهر لك المبلغ المتبقي من رصيدك.",

    "Wrong - إستعلام عن الباقة":
        "لمعرفة عدد وحداتك الحالي، اطلب *414#",

    "Wrong - تمديد صلاحية خطك":
        "خدمة لكل عملاء الدفع المسبق لشراء صلاحيه و الإحتفاظ بالخط",

    "Wrong - الشريحة الإلكترونية":
        "الشريحة الإلكترونية"
}

query_embedding = model.encode(
    query,
    normalize_embeddings=True
)

for title, document in documents.items():

    document_embedding = model.encode(
        document,
        normalize_embeddings=True
    )

    similarity = np.dot(query_embedding, document_embedding)

    print(f"{title}: {similarity:.4f}")