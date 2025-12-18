"""
main.py

Purpose:
--------
Demonstrates a local Retrieval-Augmented Generation (RAG) pipeline
using LlamaIndex + Ollama.

- Documents are loaded from the `data/` directory
- Text is embedded using a local Ollama embedding model
- A local LLM (LLaMA 3 via Ollama) generates answers
- No OpenAI / no external API keys required

NOTE:
-----
This script uses a *generative* query engine.
For exact / verbatim answers from PDFs, use a retriever-only approach.
"""

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama


def main() -> None:
    """
    Entry point for the RAG pipeline.
    """

    # ---------------------------------------------------------
    # 1. Load documents from the 'data/' directory
    #    Supported formats: PDF, TXT, MD, DOCX, etc.
    # ---------------------------------------------------------
    documents = SimpleDirectoryReader("data").load_data()

    # ---------------------------------------------------------
    # 2. Configure local embedding model (Ollama)
    #    Converts text chunks into vector embeddings
    # ---------------------------------------------------------
    embed_model = OllamaEmbedding(
        model_name="mxbai-embed-large"
    )

    # ---------------------------------------------------------
    # 3. Configure local LLM (Ollama)
    #    Used only for answer generation
    # ---------------------------------------------------------
    llm = Ollama(
        model="llama3",
        temperature=0  # lower temperature = less creativity
    )

    # ---------------------------------------------------------
    # 4. Build a vector index from documents
    # ---------------------------------------------------------
    index = VectorStoreIndex.from_documents(
        documents=documents,
        embed_model=embed_model
    )

    # ---------------------------------------------------------
    # 5. Create a query engine (RAG)
    # ---------------------------------------------------------
    query_engine = index.as_query_engine(
        llm=llm
    )

    # ---------------------------------------------------------
    # 6. Ask a question
    # ---------------------------------------------------------
    while True:
        try:
            question = input("Ask a question: ").strip()

            if question.lower() in {"exit", "quit"}:
                print("Exiting. Goodbye!")
                break

            if not question:
                print("Please enter a valid question.\n")
                continue

            response = query_engine.query(question)

            print("\nAnswer:")
            print(response)
            print("-" * 60)

        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting.")
            break


if __name__ == "__main__":
    main()
