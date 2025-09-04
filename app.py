import streamlit as st
import requests
import json

# -------------------- API Configuration --------------------
API_URL = "http://localhost:8000"

# -------------------- Session State ------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = None

# -------------------- Sidebar - Model Settings -------------
st.sidebar.title("🔧 Model Settings")
model_choice = st.sidebar.selectbox(
    "Choose LLM provider", 
    ["Groq (GPT-OSS-120B)", "OPENAI (GPT-4o-Mini)"]
)

openai_api_key = None
if model_choice == "OPENAI (GPT-4o-Mini)":
    openai_api_key = st.sidebar.text_input("🔑 Enter your OpenAI API Key", type="password")
    if not openai_api_key:
        st.error("Please enter a valid OpenAI API Key to continue.")
        st.stop()

# -------------------- Title -------------------------------
st.title("📘 Freshservice API Q&A Assistant")
st.caption("Ask questions about Freshservice API (scraped docs).")

# -------------------- Scrape Documentation -----------------
if st.sidebar.button("📥 Scrape Freshservice Docs"):
    with st.spinner("Scraping Freshservice API documentation..."):
        try:
            response = requests.post(
                f"{API_URL}/scrape_docs",
                json={"session_id": st.session_state.session_id}
            )
            if response.status_code == 200:
                data = response.json()
                st.session_state.session_id = data["session_id"]
                st.sidebar.success("✅ Docs scraped and embedded successfully.")
            else:
                st.sidebar.error(f"Error: {response.json().get('detail', 'Unknown error')}")
        except Exception as e:
            st.sidebar.error(f"❌ Failed to connect: {str(e)}")

# -------------------- Clear Session ------------------------
if st.sidebar.button("🆕 Clear All Content"):
    if st.session_state.session_id:
        try:
            response = requests.delete(f"{API_URL}/session/{st.session_state.session_id}")
            if response.status_code == 200:
                st.session_state.session_id = None
                st.rerun()
        except:
            pass

# -------------------- Session Info -------------------------
if st.session_state.session_id:
    st.sidebar.info(f"Session ID: {st.session_state.session_id[:8]}...")

# -------------------- User Query ---------------------------
user_prompt = st.chat_input("💬 Ask a question about Freshservice API")

if user_prompt:
    if not st.session_state.session_id:
        st.warning("⚠️ Please scrape the docs first.")
    else:
        with st.spinner("Processing query..."):
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json={
                        "query": user_prompt,
                        "session_id": st.session_state.session_id,
                        "model_choice": model_choice,
                        "openai_api_key": openai_api_key
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    st.write(data["answer"])

                    # Show retrieved context
                    with st.expander("📖 Relevant Documentation Sections"):
                        if data["context"]:
                            for doc in data["context"]:
                                st.markdown(f"**Source:** {doc.get('source', 'N/A')}")
                                st.write(doc["content"])
                                st.write("---")
                        else:
                            st.info("No supporting context found.")

                    st.caption(f"⚡ Response time: {data['response_time']:.2f} seconds")
                else:
                    error_data = response.json()
                    st.error(f"❌ Error: {error_data.get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"❌ Failed to connect: {str(e)}")

# -------------------- API Health Check ---------------------
with st.sidebar:
    if st.button("🔍 Check API Status"):
        try:
            response = requests.get(f"{API_URL}/health")
            if response.status_code == 200:
                data = response.json()
                st.success(f"API Status: {data['status']}")
                st.info(f"Active Sessions: {data['active_sessions']}")
            else:
                st.error("API is not responding")
        except:
            st.error("Cannot connect to API")
