import os
from typing import List

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# ----------------------------
# Global objects (loaded once)
# ----------------------------

EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

CHROMA_CLIENT = chromadb.Client(
    Settings(
        persist_directory="./chroma_db",
        anonymized_telemetry=False
    )
)

COLLECTION = CHROMA_CLIENT.get_or_create_collection(
    name="documents"
)


# ----------------------------
# PDF processing
# ----------------------------

def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    pages_text = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)

    return "\n".join(pages_text)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks


def process_document(file_path: str) -> None:
    text = extract_text_from_pdf(file_path)
    chunks = chunk_text(text)

    embeddings = EMBEDDING_MODEL.encode(chunks).tolist()
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    COLLECTION.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )


# ----------------------------
# Question answering
# ----------------------------

def process_prompt(user_prompt: str) -> str:
    query_embedding = EMBEDDING_MODEL.encode([user_prompt]).tolist()

    results = COLLECTION.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    if not results["documents"]:
        return "I don't have any documents yet. Please upload a PDF first."

    context = "\n\n".join(results["documents"][0])

    # VERY simple response (no LLM yet)
    response = (
        "Based on the uploaded document, here is the most relevant information:\n\n"
        f"{context}\n\n"
        "If you want AI-generated answers, we can add an LLM next."
    )

    return response

