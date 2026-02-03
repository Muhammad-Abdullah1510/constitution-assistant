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


### 2. Create a .env file
```bash
Create a .env file in the project root and add your API keys:

GROQ_API_KEY=your_groq_key_here
ZILLIZ_ENDPOINT=your_endpoint_here
ZILLIZ_API_KEY=your_zilliz_key_here
HUGGINGFACEHUB_API_TOKEN=your_Api_tokken_Here

3. Install dependencies
pip install -r requirements.txt

4. Build the vector database (first time only)
python constitution_builder.py

# This will:
# Load your Pakistan_Constitution.pdf
# Split it into chunks
# Upload embeddings to Milvus (Zilliz) vector database


5. Run the app
streamlit run constitution_app.py
