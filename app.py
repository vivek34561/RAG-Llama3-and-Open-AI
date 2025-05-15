import streamlit as st
import os
import shutil # this is code to delete the folder
import requests
import hashlib
from io import BytesIO
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from bs4 import BeautifulSoup
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import openai

from dotenv import load_dotenv
load_dotenv()
## load the GROQ API Key
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")



st.sidebar.title("🔧 Model Settings")

model_choice = st.sidebar.selectbox("Choose LLM provider", ["Groq (LLaMA3)", "OpenAI (GPT-4)"])

# Input API key based on model choice
if model_choice == "Groq (LLaMA3)":
    user_groq_key = st.sidebar.text_input("🔑 Enter your GROQ API Key", type="password")
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    max_tokens = st.sidebar.slider("Max Tokens", 100, 2048, 512, 50)

    if user_groq_key:
        llm = ChatGroq(
            groq_api_key=user_groq_key,
            model_name="Llama3-8b-8192",
            temperature=temperature,
            max_tokens=max_tokens
        )
    else:
        st.error("Please enter a valid GROQ API Key to continue.")
        st.stop()

else:
    user_openai_key = st.sidebar.text_input("🔑 Enter your OpenAI API Key", type="password")
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    max_tokens = st.sidebar.slider("Max Tokens", 100, 2048, 512, 50)
    
    if user_openai_key:
      llm = ChatOpenAI(
        openai_api_key=user_openai_key,
        model_name="gpt-4o-mini",
        temperature=temperature,
        max_tokens=max_tokens
    )
    else:
      st.error("Please enter a valid OpenAI API Key to continue.")
      st.stop()



prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate respone based on the question
    <context>
    {context}
    <context>
    Question:{input}

    """

)



def hash_file(file_path):
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def create_vector_embedding():
    current_files = sorted(os.listdir("uploaded_documents"))
    file_hashes = [hash_file(os.path.join("uploaded_documents", file)) for file in current_files]

    # Create a fingerprint of current upload state
    current_fingerprint = "_".join(file_hashes)

    if "last_fingerprint" in st.session_state and st.session_state["last_fingerprint"] == current_fingerprint:
        st.info("No new documents. Using cached embeddings.")
        return

    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    st.session_state.loader = PyPDFDirectoryLoader("uploaded_documents")

    st.session_state.docs = st.session_state.loader.load()
    if not st.session_state.docs:
        st.error("No documents found or failed to load.")
        return

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

    try:
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings
        )
        st.session_state["last_fingerprint"] = current_fingerprint
        st.success("✅ Vector embeddings created.")
    except Exception as e:
        st.error(f"Embedding creation failed: {str(e)}")
 
            
st.title("✨ RAG Document Q&A With Llama3 and OpenAI ✨")

st.sidebar.title("📄 Upload Documents")

# Create folder to save uploaded documents
UPLOAD_FOLDER = "uploaded_documents"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed file types
allowed_types = ["pdf", "txt", "docx"]

# File uploader in sidebar
uploaded_files = st.sidebar.file_uploader(
    "Upload files (PDF, TXT, DOCX)", 
    type=allowed_types, 
    accept_multiple_files=True
)


# Save uploaded files


st.session_state.uploaded_docs = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        save_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if file_ext == "pdf":
            loader = PyPDFLoader(save_path)
        elif file_ext == "txt":
            loader = TextLoader(save_path)
        elif file_ext == "docx":
            loader = Docx2txtLoader(save_path)
        else:
            st.warning(f"Unsupported file type: {uploaded_file.name}")
            continue

        try:
            docs = loader.load()
            st.session_state.uploaded_docs.extend(docs)
            st.sidebar.success(f"Loaded: {uploaded_file.name}")
        except Exception as e:
            st.sidebar.error(f"Failed to load {uploaded_file.name}: {str(e)}")

          
user_prompt=st.text_input("Enter your query from the uploaded documents")

col1, col2 = st.columns([1, 1])

with col1:
    if st.button(" Find Solution : ⛓️‍💥 "):
        create_vector_embedding()
        st.success("Vector Database is ready")

with col2:
    if st.button("🆕 New Chat"):
        # Clear uploaded files
        folder = UPLOAD_FOLDER
        for file_name in os.listdir(folder):
            file_path = os.path.join(folder, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        # Reset session variables
        for key in ["vectors", "embeddings", "loader", "docs", "text_splitter", "final_documents"]:
            st.session_state.pop(key, None)
        st.rerun()
        # Refresh the app
        
st.sidebar.title("🌐 Add Web Content")
url_input = st.sidebar.text_input("Paste URL to fetch and embed")

if st.sidebar.button("Fetch and Embed from URL"):
    if url_input:
        try:
            response = requests.get(url_input, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            page_text = soup.get_text(separator=" ").strip()

            if len(page_text) < 100:
                st.sidebar.error("URL content is too short or unreadable.")
            else:
                # Split text
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                url_docs = text_splitter.create_documents([page_text])

                # Use HuggingFace embeddings (all-MiniLM-L6-v2)
                if "embeddings" not in st.session_state:
                    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

                # Vector storage
                if "vectors" not in st.session_state:
                    st.session_state.vectors = FAISS.from_documents(url_docs, st.session_state.embeddings)
                else:
                    st.session_state.vectors.add_documents(url_docs)

                st.sidebar.success("Text from URL embedded successfully!")

        except Exception as e:
            st.sidebar.error(f"Failed to fetch URL content: {str(e)}")
    else:
        st.sidebar.warning("Please enter a valid URL.")

        
        


import time

if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("⚠️ Please click the '📄 Find Solution' button before asking a question.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': user_prompt})
        st.write(response['answer'])

        with st.expander("Document similarity Search"):
            if "context" in response:
                for i, doc in enumerate(response['context']):
                    st.write(doc.page_content)
                    st.write('------------------------')
            else:
                st.info("No context documents were returned.")

        print(f"Response time: {time.process_time() - start}")





