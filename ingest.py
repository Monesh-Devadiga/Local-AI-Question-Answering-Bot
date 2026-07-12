import os
import shutil
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
DOCS_DIR = os.path.join(os.path.dirname(__file__), "documents")

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv", ".log"}


def load_documents(docs_dir: str = DOCS_DIR) -> list:
    all_docs = []
    if not os.path.isdir(docs_dir):
        os.makedirs(docs_dir, exist_ok=True)
        return all_docs

    for ext in SUPPORTED_EXTENSIONS:
        for filepath in glob.glob(os.path.join(docs_dir, f"**/*{ext}"), recursive=True):
            try:
                if ext == ".pdf":
                    loader = PyPDFLoader(filepath)
                else:
                    loader = TextLoader(filepath, encoding="utf-8")
                all_docs.extend(loader.load())
            except Exception as e:
                print(f"[WARN] Failed to load {filepath}: {e}")
    return all_docs


def split_documents(docs: list, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def build_vectorstore(
    docs_dir: str = DOCS_DIR,
    chroma_dir: str = CHROMA_DIR,
    embedding_model: str = "nomic-embed-text",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> Chroma:
    raw_docs = load_documents(docs_dir)
    if not raw_docs:
        raise FileNotFoundError(
            f"No documents found in '{docs_dir}'. "
            "Add PDF or text files and re-run ingestion."
        )

    chunks = split_documents(raw_docs, chunk_size, chunk_overlap)
    print(f"[INGEST] Loaded {len(raw_docs)} docs -> {len(chunks)} chunks")

    embeddings = OllamaEmbeddings(model=embedding_model)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_dir,
    )
    print(f"[INGEST] Vector store saved to {chroma_dir}")
    return vectorstore


def load_vectorstore(
    chroma_dir: str = CHROMA_DIR,
    embedding_model: str = "nomic-embed-text",
) -> Chroma:
    if not os.path.isdir(chroma_dir):
        raise FileNotFoundError(
            f"Vector store not found at '{chroma_dir}'. "
            "Run ingest first to build the store."
        )
    embeddings = OllamaEmbeddings(model=embedding_model)
    return Chroma(persist_directory=chroma_dir, embedding_function=embeddings)


def ingest_uploaded_file(
    file_name: str,
    file_bytes: bytes,
    chroma_dir: str = CHROMA_DIR,
    embedding_model: str = "nomic-embed-text",
) -> int:
    os.makedirs(DOCS_DIR, exist_ok=True)

    ext = os.path.splitext(file_name)[1].lower()
    save_path = os.path.join(DOCS_DIR, file_name)
    with open(save_path, "wb") as f:
        f.write(file_bytes)

    if ext == ".pdf":
        loader = PyPDFLoader(save_path)
    else:
        loader = TextLoader(save_path, encoding="utf-8")

    docs = loader.load()
    chunks = split_documents(docs)

    embeddings = OllamaEmbeddings(model=embedding_model)
    if os.path.isdir(chroma_dir):
        vectorstore = Chroma(
            persist_directory=chroma_dir, embedding_function=embeddings
        )
        vectorstore.add_documents(chunks)
    else:
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=chroma_dir,
        )

    return len(chunks)


if __name__ == "__main__":
    build_vectorstore()
