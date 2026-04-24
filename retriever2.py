import chromadb
import os
import fitz
from PIL import Image
import io
import numpy as np
from sentence_transformers import SentenceTransformer

# Load Model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_db_storage")
collection = chroma_client.get_collection(name="data")

def get_page_image(pdf_filename, page_num):
    pdf_path = None
    for root, dirs, files in os.walk("./data"):
        if pdf_filename in files:
            pdf_path = os.path.join(root, pdf_filename)
            break
    
    if not pdf_path: return None
    
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num - 1)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        return Image.open(io.BytesIO(pix.tobytes("png")))
    except:
        return None

def get_relevant_course_context(user_query: str, subject: str = "All", level: str = "Beginner", max_distance: float = 1.5):
    query_vector = embedding_model.encode(user_query).tolist()

    # Filter by Subject AND Level
    where_clause = {"level": level}
    if subject != "All":
        where_clause = {"$and": [{"subject": subject}, {"level": level}]}

    query_params = {
        "query_embeddings": [query_vector],
        "n_results": 4, # Just grab the top 4 most semantically similar chunks directly
        "where": where_clause
    }

    results = collection.query(**query_params)
    valid_chunks = []

    if results['documents'] and len(results['documents']) > 0:
        docs = results['documents'][0]
        dists = results['distances'][0]
        metas = results['metadatas'][0]

        for i in range(len(docs)):
            if dists[i] <= max_distance:
                meta = metas[i]
                valid_chunks.append({
                    "text": docs[i],
                    "source": meta.get('source', 'Unknown'),
                    "page": meta.get('page', 1)
                })

    return valid_chunks

def calculate_hallucination_score(ai_answer: str, textbook_context: str) -> int:
    ans_emb = embedding_model.encode(ai_answer)
    ctx_emb = embedding_model.encode(textbook_context)
    
    # Cosine similarity between what the AI said and what the book says
    similarity = np.dot(ans_emb, ctx_emb) / (np.linalg.norm(ans_emb) * np.linalg.norm(ctx_emb))
    
    # Convert similarity to a Hallucination Percentage (0% = perfect match)
    hallucination_pct = max(0, min(100, (1 - similarity) * 100))
    # We tweak it slightly because summaries naturally differ from raw text
    adjusted_pct = max(0, int(hallucination_pct - 25)) 
    return adjusted_pct


# --- QUICK TEST ---
if __name__ == "__main__":
    subject = "OS" 
    answer = get_relevant_course_context("What is a zombie process?", subject=subject)
    print("Testing " + subject + " subject filter...")
    print(answer)