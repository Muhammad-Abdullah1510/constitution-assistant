import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Milvus
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

load_dotenv()

# ================= UI CONFIG =================
st.set_page_config(
    page_title="Constitution Assistant",
    page_icon="⚖️",
    layout="centered"
)

st.markdown("""
<style>
body {
    background-color: #f5f7fa;
}
.header {
    font-size: 2.3rem;
    font-weight: 700;
    text-align: center;
    color: #415a77;
    margin-bottom: 0.2rem;
}
.subtext {
    text-align: center;
    color: #415a77;
    margin-bottom: 2rem;
}
.stChatMessage {
    border-radius: 10px;
    padding: 10px;
}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header">⚖️ Constitution Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Ask questions about the Pakistan Constitution in simple student-friendly language.</div>', unsafe_allow_html=True)

# ================= SESSION STATE =================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "processing" not in st.session_state:
    st.session_state.processing = False

# ================= RAG INITIALIZATION =================
@st.cache_resource
def initialize_rag():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    vector_store = Milvus(
        embedding_function=embedding_model,
        connection_args={
            "uri": os.getenv("ZILLIZ_ENDPOINT"),
            "token": os.getenv("ZILLIZ_API_KEY"),
            "secure": True,
        },
        collection_name="pakistan_constitution",
    )

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
    retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)

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

    return rag_chain


rag_chain = initialize_rag()

# ================= CHAT DISPLAY =================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask about Articles, Fundamental Rights, Amendments...")

if prompt and not st.session_state.processing:
    st.session_state.processing = True
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consulting the Constitution..."):
            try:
                response = rag_chain.invoke(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(str(e))
            finally:
                st.session_state.processing = False

    st.rerun()

# ================= SIDEBAR =================
with st.sidebar:
    st.title("📚 About")
    st.write("AI assistant trained on the Pakistan Constitution.")
    st.write("Provides student-friendly legal explanations with article references.")
    st.markdown("---")
    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.caption("Built with LangChain + Groq + Zilliz")
