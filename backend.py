from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import requests
from bs4 import BeautifulSoup
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import time
import uuid
from datetime import datetime

load_dotenv()

app = FastAPI(title="Freshservice API RAG Assistant", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global sessions
sessions: Dict[str, Dict[str, Any]] = {}

# Env
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# -------------------- Models --------------------
class QueryRequest(BaseModel):
    query: str
    session_id: str
    model_choice: str = "Groq (GPT-OSS-120B)"
    openai_api_key: Optional[str] = None

class SessionResponse(BaseModel):
    session_id: str
    message: str

class QueryResponse(BaseModel):
    answer: str
    context: List[Dict[str, Any]]
    response_time: float

# -------------------- Prompt --------------------
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based only on the provided Freshservice API documentation.
    If you don’t know the answer from the docs, say you don’t know.
    <context>
    {context}
    </context>
    Question: {input}
    """
)

# -------------------- Utils --------------------
def get_llm(model_choice: str, openai_api_key: Optional[str] = None):
    if model_choice == "Groq (GPT-OSS-120B)":
        if not GROQ_API_KEY:
            raise HTTPException(status_code=400, detail="GROQ_API_KEY not configured")
        return ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="openai/gpt-oss-120b"
        )
    else:
        if not openai_api_key:
            raise HTTPException(status_code=400, detail="OpenAI API Key required")
        return ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name="gpt-4o-mini"
        )

def scrape_freshservice_docs() -> List[str]:
    """Scrape multiple Freshservice API doc sections."""
    base_url = "https://api.freshservice.com/"
    sections = [
        "#ticket_attributes",
        "#view_a_ticket",
        "#filter_tickets",
        "#view_all_tickets",
        "#update_a_ticket",
        "#move_a_ticket"

    ]

    all_content = []
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0 Safari/537.36"
        )
    }

    for section in sections:
        url = f"{base_url}{section}"
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            print(f"⚠️ Skipping {url} (status {response.status_code})")
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        section_tag = soup.find(id=section.replace("#", ""))

        if not section_tag:
            print(f"⚠️ Section {section} not found in {url}")
            continue

        for tag in section_tag.find_all_next(["h1", "h2", "h3", "p", "code", "pre", "li"]):
            text = tag.get_text(" ", strip=True)
            if text:
                all_content.append(text)

    return all_content

# -------------------- Endpoints --------------------
@app.get("/")
async def root():
    return {"message": "Freshservice RAG API is running"}

@app.post("/scrape_docs", response_model=SessionResponse)
async def scrape_docs(session_id: Optional[str] = None):
    """Scrape Freshservice docs and embed into vector store."""
    try:
        session_id = session_id or str(uuid.uuid4())
        if session_id not in sessions:
            sessions[session_id] = {
                "created_at": datetime.now(),
                "embeddings": None,
                "vectors": None
            }

        session = sessions[session_id]

        # Scrape docs
        docs_text = scrape_freshservice_docs()
        if not docs_text:
            raise Exception("No documentation text extracted")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.create_documents(docs_text)

        # Initialize embeddings + vector store
        if session["embeddings"] is None:
            session["embeddings"] = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        session["vectors"] = FAISS.from_documents(docs, session["embeddings"])

        return SessionResponse(session_id=session_id, message="Freshservice docs scraped and embedded successfully.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_content(request: QueryRequest):
    """Query the scraped Freshservice docs."""
    try:
        if request.session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        session = sessions[request.session_id]
        if session["vectors"] is None:
            raise HTTPException(status_code=400, detail="Docs not scraped yet")

        llm = get_llm(request.model_choice, request.openai_api_key)

        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = session["vectors"].as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start_time = time.time()
        response = retrieval_chain.invoke({"input": request.query})
        response_time = time.time() - start_time

        context = []
        if "context" in response:
            for doc in response["context"]:
                context.append({
                    "content": doc.page_content,
                    "metadata": getattr(doc, "metadata", {})
                })

        return QueryResponse(
            answer=response["answer"],
            context=context,
            response_time=response_time
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
        return {"message": f"Session {session_id} cleared successfully"}
    raise HTTPException(status_code=404, detail="Session not found")

@app.get("/sessions")
async def list_sessions():
    return {
        "sessions": [
            {
                "session_id": sid,
                "created_at": session["created_at"].isoformat(),
                "has_vectors": session["vectors"] is not None
            }
            for sid, session in sessions.items()
        ]
    }

@app.delete("/sessions")
async def clear_all_sessions():
    sessions.clear()
    return {"message": "All sessions cleared successfully"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "active_sessions": len(sessions),
        "groq_configured": bool(GROQ_API_KEY)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
