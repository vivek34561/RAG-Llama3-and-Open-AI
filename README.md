# 🔍 RAG-Based Document Q&A App (Streamlit + LLaMA3/OpenAI)

This is a **Retrieval-Augmented Generation (RAG)** application built with **Streamlit** that lets you ask questions based on **uploaded documents** or **webpage content** using **LLaMA3 (via Groq)** or **OpenAI's GPT-4**.

---

## 🚀 Features

- 📄 Upload and embed **PDF, DOCX, or TXT** documents  
- 🌐 Fetch and embed **text from URLs**  
- 💬 Ask natural language questions about your uploaded or fetched documents  
- 🔁 Supports **LLaMA3 (Groq)** and **GPT-4 (OpenAI)**  
- 🧠 Intelligent context-aware responses using **LangChain RAG**  
- 🧠 Embedding caching to avoid recomputation  
- 🎯 Adjustable model parameters (temperature, max tokens)  
- 🧹 Clear uploads and reset chat with one click  

---

## 🧰 Tech Stack

- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [HuggingFace Embeddings](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [Groq LLaMA3](https://groq.com/)
- [OpenAI GPT-4](https://openai.com/)
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/)

---

## 🏁 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/rag-document-qa.git
cd rag-document-qa
````

### 2. Create and activate virtual environment

```bash
python -m venv rag_venv
source rag_venv/bin/activate    # On Windows: rag_venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root directory and add your API keys:

```
OPENAI_API_KEY=your-openai-api-key  
GROQ_API_KEY=your-groq-api-key  
```

---

## 🧪 Run the App

```bash
streamlit run app.py
```

---

## 🧠 Model Options

Choose from the sidebar:

* 🔸 **Groq (LLaMA3)**: Fast, local-like responses with Groq API.
* 🔹 **OpenAI (GPT-4)**: Best for accuracy and comprehension.

Adjust:

* **Temperature**: Controls randomness (0.0 = factual, 1.0 = creative).
* **Max Tokens**: Controls response length.

---

## 🗃️ File Structure

```
├── app.py                 # Main Streamlit app  
├── requirements.txt       # Python dependencies  
├── .env                   # Your API keys  
├── uploaded_documents/    # Folder for storing uploaded files  
```

📄 **[Download Full Project Structure PDF](https://chat.openai.com/mnt/data/ML_Project_Structure.pdf)**

---

## 📸 Screenshots

### 🧠 Streamlit App Interface


![Screenshot-2025-05-17-134217](https://github.com/user-attachments/assets/f37721dd-6d30-43a5-b249-dc569444e251)


![App UI] with 📂 File Upload & Sidebar Settings pdf 📁 
[Screenshot 2025-05-17 132104.pdf](https://github.com/user-attachments/files/20263793/Screenshot.2025-05-17.132104.pdf)


## ⚙️ Notes

* Only processes `.pdf`, `.txt`, and `.docx` files.
* Uses **MD5 hash-based caching** to avoid re-embedding unchanged documents.
* Web scraping uses `requests` + `BeautifulSoup`.

---

## 🛡️ License

MIT License. Feel free to use, modify, and share!

---

## 📧 Contact

Maintained by Vivek Kumar Gupta . Feel free to contribute or raise an issue.

For any inquiries, please reach out via email (vivekgupta3749@gmail.com) or GitHub.
