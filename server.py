import os
import shutil
import time
import hashlib
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import requests
from bs4 import BeautifulSoup

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

# Constants and setup
UPLOAD_FOLDER = "uploaded_documents"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = FastAPI(title="RAG Llama3/OpenAI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AppState:
    def __init__(self):
        self.embeddings = None
        self.vectors: Optional[FAISS] = None
        self.final_documents = []
        self.last_fingerprint: Optional[str] = None


STATE = AppState()


prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Question:{input}
    """
)


# Models
class QueryRequest(BaseModel):
    provider: str = Field(description="'groq' or 'openai'")
    api_key: str
    prompt: str
    temperature: float = 0.7
    max_tokens: int = 512


class UrlsRequest(BaseModel):
    urls: List[str]


# Helpers

def hash_file(file_path: str) -> str:
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def load_documents_from_uploads():
    docs = []
    for name in sorted(os.listdir(UPLOAD_FOLDER)):
        path = os.path.join(UPLOAD_FOLDER, name)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(name)[1].lower()
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(path)
            elif ext == ".txt":
                loader = TextLoader(path, encoding='utf-8')
            elif ext == ".docx":
                loader = Docx2txtLoader(path)
            else:
                # skip unsupported
                continue
            docs.extend(loader.load())
        except Exception:
            # skip files that fail to load
            continue
    return docs


def ensure_embeddings():
    """Ensure embeddings client is initialized (uses OpenAI Embeddings to keep slug small)."""
    if STATE.embeddings is None:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set on the server. Configure it as an environment variable.")
        STATE.embeddings = OpenAIEmbeddings(openai_api_key=api_key, model="text-embedding-3-small")


def create_vector_embedding():
    # Check for files
    if not os.path.exists(UPLOAD_FOLDER) or not os.listdir(UPLOAD_FOLDER):
        return {"status": "no_files", "message": "No documents found to embed."}

    current_files = sorted([f for f in os.listdir(UPLOAD_FOLDER) if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))])
    file_hashes = [hash_file(os.path.join(UPLOAD_FOLDER, file)) for file in current_files]
    current_fingerprint = "_".join(file_hashes)

    if STATE.last_fingerprint == current_fingerprint and STATE.vectors is not None:
        return {"status": "cached", "message": "No new documents. Using cached embeddings."}

    docs = load_documents_from_uploads()
    if not docs:
        return {"status": "error", "message": "No documents found or failed to load."}

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = splitter.split_documents(docs)

    try:
        ensure_embeddings()
        vectors = FAISS.from_documents(final_documents, STATE.embeddings)
        STATE.vectors = vectors
        STATE.final_documents = final_documents
        STATE.last_fingerprint = current_fingerprint
        return {"status": "ok", "message": "Vector embeddings created.", "chunks": len(final_documents)}
    except Exception as e:
        return {"status": "error", "message": f"Embedding creation failed: {str(e)}"}


def add_url_documents(urls: List[str]):
    ensure_embeddings()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_url_docs = []
    failed_urls = []

    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            if not response.ok:
                raise Exception(f"Request failed with status code {response.status_code}")
            soup = BeautifulSoup(response.text, "html.parser")
            for script in soup(["script", "style"]):
                script.extract()
            page_text = soup.get_text(separator=" ").strip()
            if len(page_text) >= 100:
                all_url_docs.extend(splitter.create_documents([page_text]))
            else:
                failed_urls.append(url)
        except Exception as e:
            failed_urls.append(url)

    if all_url_docs:
        if STATE.vectors is None:
            STATE.vectors = FAISS.from_documents(all_url_docs, STATE.embeddings)
        else:
            STATE.vectors.add_documents(all_url_docs)

    return {"embedded": len(all_url_docs), "failed": failed_urls}


def get_llm(provider: str, api_key: str, temperature: float, max_tokens: int):
    provider = provider.lower().strip()
    if provider in ["groq", "llama", "llama3", "llama-3", "llama-3.3"]:
        return ChatGroq(
            groq_api_key=api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif provider in ["openai", "gpt", "gpt-4o", "gpt-4o-mini"]:
        return ChatOpenAI(
            openai_api_key=api_key,
            model_name="gpt-4o-mini",
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        raise ValueError("Unsupported provider. Use 'groq' or 'openai'.")


# Routes
@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    saved = []
    for file in files:
        filename = file.filename
        ext = os.path.splitext(filename)[1].lower()
        if ext not in [".pdf", ".txt", ".docx"]:
            continue
        dest = os.path.join(UPLOAD_FOLDER, filename)
        with open(dest, "wb") as f:
            f.write(await file.read())
        saved.append(filename)
    # Invalidate fingerprint if new files added
    if saved:
        STATE.last_fingerprint = None
    return {"saved": saved}


@app.post("/embed")
async def embed_documents():
    result = create_vector_embedding()
    return result


@app.post("/urls")
async def ingest_urls(req: UrlsRequest):
    urls = [u.strip() for u in req.urls if u and u.strip()]
    if not urls:
        return {"embedded": 0, "failed": [], "message": "No valid URLs provided."}
    result = add_url_documents(urls)
    return result


@app.post("/query")
async def query_rag(body: QueryRequest):
    if STATE.vectors is None:
        return {"error": "No embeddings available. Upload and embed documents first or ingest URLs."}
    try:
        llm = get_llm(body.provider, body.api_key, body.temperature, body.max_tokens)
    except Exception as e:
        return {"error": str(e)}

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = STATE.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    try:
        response = retrieval_chain.invoke({'input': body.prompt})
        answer = response.get('answer', '')
        contexts = []
        if 'context' in response and response['context']:
            for doc in response['context']:
                try:
                    contexts.append(doc.page_content)
                except Exception:
                    continue
        elapsed = time.process_time() - start
        return {"answer": answer, "contexts": contexts, "latency_sec": elapsed}
    except Exception as e:
        return {"error": f"LLM failed: {str(e)}"}


@app.post("/reset")
async def reset_state():
    # Clear uploaded files
    if os.path.exists(UPLOAD_FOLDER):
        for name in os.listdir(UPLOAD_FOLDER):
            path = os.path.join(UPLOAD_FOLDER, name)
            if os.path.isfile(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
    # Reset state
    STATE.embeddings = None
    STATE.vectors = None
    STATE.final_documents = []
    STATE.last_fingerprint = None
    return {"status": "ok"}
