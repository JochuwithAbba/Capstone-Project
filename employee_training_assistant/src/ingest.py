"""Build FAISS vectorstores from documents in the data folders."""

from __future__ import annotations

from pathlib import Path

from src.config import DATA_PATHS, EMBEDDING_MODEL_NAME, VECTORSTORE_PATHS


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def build_all_vectorstores() -> dict[str, str]:
    """Build vectorstores for every configured route."""
    results: dict[str, str] = {}
    for route in DATA_PATHS:
        results[route] = build_vectorstore_for_route(route)
    return results


def build_vectorstore_for_route(route: str) -> str:
    """
    Build a FAISS vectorstore for a single route.

    Documents are read from data/<route folder> and saved to the matching
    vectorstores/<route folder>. This is the missing link between uploaded PDFs
    and Groq-based answer generation.
    """
    data_path = DATA_PATHS.get(route)
    vectorstore_path = VECTORSTORE_PATHS.get(route)

    if not data_path or not vectorstore_path:
        return f"No data/vectorstore path is configured for route '{route}'."

    source_files = _list_source_files(data_path)
    if not source_files:
        return f"No supported documents found in {data_path}."

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import PyPDFLoader, TextLoader
        from langchain_community.vectorstores import FAISS
    except ImportError as exc:
        return f"Missing ingestion dependency: {exc}. Run 'pip install -r requirements.txt'."

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings

    documents = []
    for file_path in source_files:
        try:
            if file_path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file_path))
            else:
                loader = TextLoader(str(file_path), encoding="utf-8")

            loaded_documents = loader.load()
            for document in loaded_documents:
                document.metadata["source"] = str(file_path.name)
                document.metadata["route"] = route
            documents.extend(loaded_documents)
        except Exception as exc:
            return f"Could not load {file_path.name}. Reason: {exc}"

    if not documents:
        return f"No text could be extracted from documents in {data_path}."

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        return f"Documents were loaded, but no chunks were created for route '{route}'."

    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore_path.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(vectorstore_path))
    except Exception as exc:
        return f"Could not build vectorstore for route '{route}'. Reason: {exc}"

    return (
        f"Built vectorstore for '{route}' with {len(chunks)} chunks "
        f"from {len(source_files)} files."
    )


def _list_source_files(data_path: Path) -> list[Path]:
    """Return supported source documents from a data directory."""
    if not data_path.exists():
        return []

    return sorted(
        file_path
        for file_path in data_path.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
