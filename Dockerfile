# Dockerfile to containerize the Physics RAG Gradio Assistant

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir gradio faiss-cpu sentence-transformers transformers torch

# Expose the Gradio port
EXPOSE 7860

# Default command to run the Gradio app
CMD ["python", "gradio_app.py"]
