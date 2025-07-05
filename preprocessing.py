import os
import json
import re
from typing import List

import fitz  # PyMuPDF for PDF parsing
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

# Load environment variables from .env
load_dotenv()

# Constants
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "RAG_DOCS"
CHROMA_DB_PATH = "chromaDB/saved/"
PDF_DIR = "PDFs/"
ALL_DOCS_JSON = "all_docs.json"


def recursive_split(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
    """
    Split text using character-level recursion based on newlines and punctuation.
    
    Args:
        text: The raw text to split.
        chunk_size: Maximum size of each chunk.
        chunk_overlap: Overlap between adjacent chunks.
    
    Returns:
        List of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(text)


def semantic_chunker(text: str, embeddings_model) -> List[str]:
    """
    Split the text into semantically meaningful chunks using a semantic chunker.

    Args:
        text: Text to split.
        embeddings_model: Pre-loaded embedding model.

    Returns:
        List of semantically coherent text chunks.
    """
    chunks = []
    for segment in recursive_split(text):
        chunker = SemanticChunker(embeddings_model)
        chunks.extend(chunker.split_text(segment))
    return chunks


def extract_chunks_from_pdf(pdf_path: str, embeddings_model) -> List[Document]:
    """
    Extract semantic chunks from a given PDF.

    Args:
        pdf_path: Path to the PDF file.
        embeddings_model: Embedding model for semantic chunking.

    Returns:
        List of Document objects containing chunked text and metadata.
    """
    documents = []
    try:
        print(f'🗂️  Getting PDF from: {pdf_path}\n')
        pdf = fitz.open(pdf_path)

        for page_idx, page in enumerate(pdf):
            print(f'📖 Reading Page no: {page_idx + 1}')
            raw_text = re.sub(r"\s+", " ", page.get_text("text")).strip()

            if not raw_text:
                continue

            for idx, chunk in enumerate(semantic_chunker(raw_text, embeddings_model)):
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "page": page_idx + 1,
                            "chunk": idx,
                            "source": os.path.basename(pdf_path)
                        }
                    )
                )
        pdf.close()

    except Exception as e:
        print(f"❌ Error while processing PDF '{pdf_path}': {e}")

    return documents


def save_docs(docs: List[Document], filepath: str = ALL_DOCS_JSON) -> None:
    """
    Save documents to JSON file.

    Args:
        docs: List of Document objects to save.
        filepath: Destination file path.
    """
    try:
        print("📥📄 Saving chunks for future use...")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                [{"page_content": d.page_content, "metadata": d.metadata} for d in docs],
                f,
                indent=2
            )
    except Exception as e:
        print(f"❌ Error saving documents: {e}")


def load_docs(filepath: str = ALL_DOCS_JSON) -> List[Document]:
    """
    Load documents from a previously saved JSON file.

    Args:
        filepath: Path to the JSON file.

    Returns:
        List of Document objects.
    """
    try:
        print("📤📄 Loading chunks...")
        if not os.path.exists(filepath):
            return []

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        return [
            Document(page_content=item["page_content"], metadata=item["metadata"])
            for item in data
        ]
    except Exception as e:
        print(f"❌ Error loading documents: {e}")
        return []


def init_chroma() -> Chroma:
    """
    Initialize or load a Chroma vector store.

    Returns:
        A Chroma object ready for indexing or retrieval.
    """
    print('🧭 Creating or loading ChromaDB...')
    return Chroma(
        persist_directory=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME,
        embedding_function=HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    )


def create_chunks(PDF_FILES: List[str]) -> None:
    """
    Process PDF files and generate semantic chunks for RAG.

    Args:
        PDF_FILES: List of PDF filenames (without extension).
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    docs = load_docs(ALL_DOCS_JSON)

    if not docs:
        for name in PDF_FILES:
            path = os.path.join(PDF_DIR, f"{name}.pdf")
            if os.path.exists(path):
                docs.extend(extract_chunks_from_pdf(path, embeddings))
            else:
                print(f"⚠️  File not found: {path}")
        save_docs(docs)
    else:
        print("✅ Pre-saved chunks loaded. Skipping PDF parsing.")


def hybrid_search(query: str) -> List[Document]:
    """
    Perform hybrid search using both BM25 and Chroma vector retrievers.

    Args:
        query: The user query.

    Returns:
        List of retrieved Document objects.
    """
    try:
        chroma_store = init_chroma()

        # Load and index documents if not already indexed
        if chroma_store._collection.count() == 0:
            docs = load_docs(ALL_DOCS_JSON)
            if docs:
                chroma_store.add_documents(docs)
            else:
                print("⚠️  No documents available for retrieval.")
                return []

        # Load for BM25
        docs = load_docs(ALL_DOCS_JSON)
        if not docs:
            return []

        chroma_retriever = chroma_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 10}
        )

        bm25_retriever = BM25Retriever.from_texts(
            [d.page_content for d in docs],
            metadatas=[d.metadata for d in docs],
            k=5
        )

        ensemble = EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_retriever],
            weights=[1, 1]
        )

        return ensemble.invoke(query)

    except Exception as e:
        print(f"❌ Error in hybrid search: {e}")
        return []
