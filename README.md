# 🧠 Physics RAG Assistant  
*A Lightweight LLM-Powered Research Assistant for Higgs Boson Physics*

This project is a fully functional **Retrieval-Augmented Generation (RAG)** system designed to answer scientific questions using **recent public research publications about the Higgs boson from CERN**.

It leverages:
- 🔍 **FAISS** for document indexing and fast vector search  
- 🧠 **SentenceTransformers** and **Falcon-RW-1B** from Hugging Face for contextual understanding  
- 🎛️ **Gradio** for an interactive user interface  

---

## 📚 Project Overview

This assistant:
- Ingests and indexes multiple PDF articles (e.g., CERN technical papers)
- Answers questions using real context from those documents
- Supports **LaTeX-style math rendering** (e.g., `$E = mc^2$`)
- Tracks user feedback for continuous evaluation

---

## 📁 Project Structure

```
.
├── pdfs/                  # ← Put CERN Higgs boson papers here (PDF format)
├── embed.py               # ← Script to extract, chunk, and index PDF content
├── gradio_app.py          # ← Interactive UI for querying the assistant
├── multi_pdf_index.faiss  # ← FAISS vector index (generated)
├── multi_pdf_chunks.jsonl # ← Metadata about text chunks (generated)
├── feedback_log.csv       # ← Logs user feedback from the app
```

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/physics-rag-assistant.git
cd physics-rag-assistant
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

If missing, create `requirements.txt` with:

```
faiss-cpu
PyPDF2
sentence-transformers
transformers
gradio
```

---

### 3. Add Articles

Download and place your **recent Higgs boson papers from CERN** in the `pdfs/` folder.  
You can get them from the [CERN Document Server](https://cds.cern.ch/) or [arXiv](https://arxiv.org/) (e.g. `CMS`, `ATLAS`, `Run 3`, etc).

---

### 4. Index the Documents

```bash
python embed.py
```

This will:
- Extract and clean the text
- Chunk it into manageable blocks
- Embed and store them in a FAISS index

---

### 5. Launch the Assistant

```bash
python gradio_app.py
```

The UI will open at:  
[http://localhost:7860](http://localhost:7860)

---

## 🧠 Example Questions

Try asking:

- “What is the measured mass of the Higgs boson?”
- “What are the latest results from CMS and ATLAS?”
- “How are anomalous Higgs couplings constrained at the LHC?”

The assistant will search your indexed PDFs and provide answers grounded in actual research text.

---

## ✨ Features

- ✅ LLM-generated answers based on real context (not hallucinated)
- ✅ Fast vector search with FAISS
- ✅ Markdown + LaTeX formula rendering
- ✅ CSV-based feedback collection (👍/👎)
- ✅ Fully offline and private — your data stays local

---

## 📌 Notes

- Best used with **machine-readable PDFs** (not scanned images)
- You can extend it with larger models, image support, or feedback analysis

---

## 📜 License

Kourosh (Mohammad) Mousavi — use it free for personal, educational, or research use.

---

## 🙌 Credits

- [Hugging Face Transformers](https://huggingface.co/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Gradio](https://www.gradio.app/)
- [CERN Open Access](https://cds.cern.ch/)
- [CMS / ATLAS Collaborations](https://home.cern/science/experiments/atlas)

