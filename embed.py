# === EMBED.PY ===
# This script extracts text from multiple PDFs, chunks it, embeds it with a sentence transformer,
# and saves a FAISS index + chunk metadata file (for use in a RAG system).

import os, re, json
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Step 1: Extract text from all PDFs in a folder
def extract_text_from_folder(folder_path):
    all_docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            try:
                reader = PdfReader(pdf_path)
                raw_text = ""
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        raw_text += text + "\n"
                all_docs.append((filename, raw_text))
            except Exception as e:
                print(f"❌ Failed to read {filename}: {e}")
    return all_docs

# Step 2: Clean and chunk the extracted text (removing references, keeping formulas)
def clean_and_chunk(filename, text, max_words=300):
    text = text.replace("\n", " ")  # Flatten text to single line
    text = re.sub(r"(?i)References.*", "", text)  # Remove everything after "References"
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append({ "source": filename, "text": chunk })
    return chunks

# Step 3: Embed each chunk using a SentenceTransformer
def embed_chunks(chunks, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    texts = [chunk['text'] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    return np.array(embeddings)

# Step 4: Save the FAISS index and metadata for retrieval in Gradio app
def build_and_save_index(embeddings, chunks, index_path="multi_pdf_index.faiss", metadata_path="multi_pdf_chunks.jsonl"):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")
    print(f"✅ Saved index: {index_path}")
    print(f"✅ Saved metadata: {metadata_path}")

# Main function: Run this script to embed all PDFs in the ./pdfs folder
if __name__ == "__main__":
    folder = "./pdfs"
    all_chunks = []
    for filename, text in extract_text_from_folder(folder):
        chunks = clean_and_chunk(filename, text)
        all_chunks.extend(chunks)
    print(f"Total chunks: {len(all_chunks)}")
    embeddings = embed_chunks(all_chunks)
    build_and_save_index(embeddings, all_chunks)
