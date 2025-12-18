# LlamaIndex Extractive RAG (Local, Ollama-Based)

This project demonstrates a **local Retrieval-Augmented Generation (RAG)**
pipeline using **LlamaIndex** and **Ollama**, without relying on OpenAI
or any external APIs.

---

## 🔹 Features

- Fully local RAG setup
- PDF and document ingestion
- Vector search using local embeddings
- Answer generation using LLaMA 3
- No OpenAI API key required
- Works offline

---

## 🔹 Tech Stack

- **LlamaIndex** – RAG framework
- **Ollama** – Local LLM & embedding runtime
- **mxbai-embed-large** – Embedding model
- **LLaMA 3** – Generative LLM
- **Python 3.10+**

---

## 🔹 Project Structure

```text
llamaindex-extractive-rag/
│
├── data/                  # Input documents (PDFs, text files)
│   └── your_document.pdf
│
├── main.py                # RAG pipeline implementation
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## 🔹 Setup Instructions
1️⃣ Create virtual environment
python -m venv venv
source venv/bin/activate

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Install & start Ollama
ollama serve

4️⃣ Pull required models
ollama pull llama3
ollama pull mxbai-embed-large

## 🔹 Running the Project
```
python main.py
```

## 🔹 How It Works

- Documents are loaded from the data/ directory

- Text is split and embedded using a local Ollama embedding model

- A vector index is created for similarity search

- Relevant chunks are retrieved for a query

- LLaMA 3 generates a response using retrieved context

## 🔹 Important Note on Answer Accuracy

This project uses a generative RAG approach.

✔ Good for explanations and summaries

❌ Not guaranteed to return verbatim text from PDFs

If you need exact text from documents:

Use a retriever-only approach instead of an LLM-based query engine.

## 🔹 Example Query
```
Question:
What do we mean by valuable?

Answer: <Generated response based on document context>

```
## 🔹 Use Cases

- Internal knowledge base search

- Document Q&A

- Policy or handbook analysis

- Learning RAG with local models