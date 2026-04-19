import os
import fitz
import chromadb
import re
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

print("Waking up the embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

print("Opening the vault (ChromaDB)...")
chroma_client = chromadb.PersistentClient(path="./chroma_db_storage")
collection = chroma_client.get_or_create_collection(name="data")

def extract_text_from_pdf(pdf_path):
    pages_data = []
    with fitz.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            if text.strip():
                pages_data.append({"text": text, "page": page_num + 1})
    return pages_data

def chunk_text(text, max_words=150, overlap_sentences=1):
    sentences = re.split(r'(?<=[.?!])\s+', text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length <= max_words:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            overlap = current_chunk[-overlap_sentences:] if overlap_sentences > 0 else []
            current_chunk = overlap + [sentence]
            current_length = sum(len(s.split()) for s in current_chunk)
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def build_database():
    data_folder = "./data"
    
    if not os.path.exists(data_folder):
        print(f"Error: I can't find the '{data_folder}' folder. Create it and add PDFs.")
        return

    for root, dirs, files in os.walk(data_folder):
        parts = os.path.normpath(root).split(os.sep)
        
        if len(parts) >= 3 and parts[-3] == 'data':
            subject_name = parts[-2]
            level_name = parts[-1]

            for filename in files:
                if filename.endswith(".pdf"):
                    print(f"\nExtracting text from [{subject_name} - {level_name}]: {filename}...")
                    pdf_path = os.path.join(root, filename)
                    
                    pages_data = extract_text_from_pdf(pdf_path)
                    
                    all_chunks = []
                    all_ids = []
                    all_metas = []
                    
                    for page_info in pages_data:
                        chunks = chunk_text(page_info["text"])
                        for i, chunk in enumerate(chunks):
                            all_chunks.append(chunk)
                            all_ids.append(f"{filename}_p{page_info['page']}_c{i}")
                            all_metas.append({
                                "source": filename, 
                                "subject": subject_name, 
                                "level": level_name, 
                                "page": page_info["page"]
                            })
                    
                    batch_size = 100 
                    
                    print(f"Vectorizing {len(all_chunks)} chunks for {filename}...")
                    
                    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Processing Batches", unit="batch"):
                        batch_chunks = all_chunks[i:i + batch_size]
                        batch_ids = all_ids[i:i + batch_size]
                        batch_metas = all_metas[i:i + batch_size]
                        
                        batch_vectors = embedding_model.encode(batch_chunks).tolist()
                        
                        collection.upsert(
                            ids=batch_ids,
                            embeddings=batch_vectors,
                            documents=batch_chunks,
                            metadatas=batch_metas
                        )
                
    print("\nDatabase build complete! The RAG is loaded and ready.")

if __name__ == "__main__":
    build_database()