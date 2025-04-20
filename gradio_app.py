# === GRADIO_APP.PY WITH OLD STYLE RESTORED ===
# Includes: Textbox for answer, source preview, and feedback buttons

import gradio as gr
import faiss, json, datetime, os
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

# Load models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-rw-1b")
model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-rw-1b")

# Load FAISS index + metadata
index = faiss.read_index("multi_pdf_index.faiss")
with open("multi_pdf_chunks.jsonl", "r", encoding="utf-8") as f:
    documents = [json.loads(line) for line in f.readlines()]

# Feedback storage
FEEDBACK_LOG = "feedback_log.csv"
if not os.path.exists(FEEDBACK_LOG):
    with open(FEEDBACK_LOG, "w") as f:
        f.write("timestamp,query,answer,feedback\n")

# RAG logic with timing

def rag_query(query, top_k=3):
    start_time = time.time()
    query_vector = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_vector), top_k)
    retrieved = [documents[i] for i in indices[0]]

    context = "\n".join([doc["text"] for doc in retrieved])[:700]
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids, max_new_tokens=100)
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    source_md = ""
    for doc in retrieved:
        source_md += f"**From:** `{doc['source']}`\n\n> {doc['text'][:200]}...\n\n"

    end_time = time.time()
    duration = end_time - start_time

    return answer.strip(), source_md, f"⏱️ Answer generated in {duration:.2f} seconds"

# Feedback capture
def handle_feedback(query, answer, feedback):
    with open(FEEDBACK_LOG, "a") as f:
        f.write(f"{datetime.datetime.now()},{query},{answer},{feedback}\n")
    return "✅ Feedback recorded. Thank you!"

# UI layout with older style: textbox for answer and footer
with gr.Blocks() as app:
    gr.Markdown("""# 🧠 Physics AI Assistant
Ask a physics question based on your scientific PDFs. Formulas are supported via Markdown.
""")

    with gr.Row():
        query_input = gr.Textbox(label="Ask a Physics Question", placeholder="e.g., What are Higgs boson couplings?", lines=1)
        ask_btn = gr.Button("Ask")

    answer_output = gr.Textbox(label="Answer", lines=4)
    source_output = gr.Textbox(label="Source Chunks", lines=6)
    time_output = gr.Textbox(label="Response Time")

    with gr.Row():
        good = gr.Button("👍")
        bad = gr.Button("👎")
        feedback_msg = gr.Textbox(visible=False)

    ask_btn.click(rag_query, inputs=query_input, outputs=[answer_output, source_output, time_output])
    good.click(lambda q, a: handle_feedback(q, a, "positive"), inputs=[query_input, answer_output], outputs=feedback_msg)
    bad.click(lambda q, a: handle_feedback(q, a, "negative"), inputs=[query_input, answer_output], outputs=feedback_msg)

app.launch(share=False, inbrowser=True)