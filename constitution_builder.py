from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Milvus
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv
import os
import pickle
from pathlib import Path

load_dotenv()

# ================= CONFIG =================
PDF_PATH = "Pakistan_Constitution.pdf"
CHUNKS_CACHE_PATH = "constitution_chunks.pkl"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
COLLECTION_NAME = "pakistan_constitution"
PROCESS_PDF = False           # FIRST TIME = True
UPLOAD_TO_VECTORDB = False   # FIRST TIME = True
# ==========================================


# ===== STEP 1: LOAD & CHUNK PDF =====
def process_pdf():
    print("📘 Processing Constitution PDF...")
    loader = UnstructuredPDFLoader(PDF_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,     # Larger chunks for legal text
        chunk_overlap=250
    )

    split_docs = splitter.split_documents(documents)

    with open(CHUNKS_CACHE_PATH, "wb") as f:
        pickle.dump(split_docs, f)

    print(f"✅ {len(split_docs)} chunks created and cached.")
    return split_docs


def load_chunks():
    if not Path(CHUNKS_CACHE_PATH).exists():
        raise FileNotFoundError("Chunks not found! Set PROCESS_PDF=True")

    with open(CHUNKS_CACHE_PATH, "rb") as f:
        docs = pickle.load(f)

    print(f"📦 Loaded {len(docs)} cached chunks.")
    return docs


split_docs = process_pdf() if PROCESS_PDF else load_chunks()


# ===== STEP 2: EMBEDDINGS + VECTOR STORE =====
def setup_vector_store(docs, upload=False):
    print("🔢 Creating embeddings...")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    endpoint = os.getenv("ZILLIZ_ENDPOINT")
    token = os.getenv("ZILLIZ_API_KEY")

    if not endpoint or not token:
        raise ValueError("Missing Zilliz credentials in .env")

    if upload:
        print("⬆ Uploading to Zilliz (first time only)...")
        vector_store = Milvus.from_documents(
            documents=docs,
            embedding=embedding_model,
            connection_args={"uri": endpoint, "token": token, "secure": True},
            collection_name=COLLECTION_NAME,
            drop_old=False,
        )
        print("✅ Upload complete.")
    else:
        vector_store = Milvus(
            embedding_function=embedding_model,
            connection_args={"uri": endpoint, "token": token, "secure": True},
            collection_name=COLLECTION_NAME,
        )

    return vector_store


vector_store = setup_vector_store(split_docs, upload=UPLOAD_TO_VECTORDB)


# ===== STEP 3: RETRIEVER (MMR + COMPRESSION) =====
print("🔍 Setting up retriever...")

base_retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8, "fetch_k": 20, "lambda_mult": 0.7}
)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

compressor = LLMChainExtractor.from_llm(llm)

retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

print("✅ Retriever ready.")


# ===== STEP 4: RAG CHAIN =====
system_prompt = """You are a Pakistan Constitution assistant helping students understand the law.

Instructions:
- Use ONLY the provided context
- First give the legal explanation
- Then explain in simple student-friendly language
- Cite Article numbers when available
- If not found, say you don’t have enough information

Context:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate(input_variables=["context", "question"], template=system_prompt)

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("🤖 RAG system ready!\n")


# ===== STEP 5: TEST =====
print("🧪 Running test questions...\n")

test_questions = [
    "What is Article 25 about?",
    "Explain fundamental rights.",
    "Who can amend the constitution?"
]

for q in test_questions:
    print(f"\n❓ {q}")
    print("-" * 60)
    print(rag_chain.invoke(q))
    print("\n")