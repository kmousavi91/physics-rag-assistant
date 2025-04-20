from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI()

# Load models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-rw-1b")
model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-rw-1b")

# Load updated FAISS index and chunk metadata
index = faiss.read_index("multi_pdf_index.faiss")
with open("multi_pdf_chunks.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    documents = [{"source": line.split("|||")[0], "text": line.split("|||")[1]} for line in lines]

# Request model
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

# API endpoint
@app.post("/ask")
def ask_question(request: QueryRequest):
    try:
        query_vector = embedding_model.encode([request.query])
        distances, indices = index.search(np.array(query_vector), request.top_k)
        retrieved = [documents[i] for i in indices[0]]

        context = "\n".join([doc["text"] for doc in retrieved])
        context = context[:700]  # truncate to avoid model overload

        prompt = f"Context:\n{context}\n\nQuestion: {request.query}\nAnswer:"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        output_ids = model.generate(input_ids, max_new_tokens=100)
        output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return {
            "query": request.query,
            "answer": output.strip(),
            "sources": [{"source": doc["source"], "excerpt": doc["text"][:200]} for doc in retrieved]
        }
    except Exception as e:
        print("🔥 ERROR:", e)
        return {"error": str(e)}
