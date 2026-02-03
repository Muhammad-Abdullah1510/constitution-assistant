# Constitution Assistant ⚖️

An AI-powered assistant that answers questions about the **Pakistan Constitution** in **student-friendly language**. Built with **Streamlit**, **LangChain**, **Groq**, and **Zilliz (Milvus)**.

---

## Features

- Answer questions about Articles, Fundamental Rights, and Amendments  
- Provides **legal explanation** + **simple explanation**  
- Shows Article citations  
- Maintains chat history in the UI  

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/constitution-assistant.git
cd constitution-assistant
````

### 2. Create a `.env` file

Create a `.env` file in the project root and add your API keys:

```
GROQ_API_KEY=your_groq_key_here
ZILLIZ_ENDPOINT=your_endpoint_here
ZILLIZ_API_KEY=your_zilliz_key_here
HUGGINGFACEHUB_API_TOKEN=your_api_token_here
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Build the vector database (first time only)

```bash
python constitution_builder.py
```

This will:

* Load your `Pakistan_Constitution.pdf`
* Split it into chunks
* Upload embeddings to Milvus (Zilliz) vector database

### 5. Run the app

```bash
streamlit run constitution_app.py
```

---

## Notes

* Ensure `Pakistan_Constitution.pdf` is in the project folder
* Python 3.12 or higher is recommended
* Virtual environments (`.venv`) are **ignored** in GitHub
* Pickle/cache files (`*.pkl`) are also ignored
* Chat history is maintained **only in the session**, not in the vector database