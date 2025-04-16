import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import os

# Streamlit Page Config
st.set_page_config(page_title="📄 AI Document Agent")

# Title and Instructions
st.title("🤖 AI Agent: Ask Anything from Your Document")
st.markdown("Upload a `.txt` or `.pdf` file and ask questions!")

# File Uploader
uploaded_file = st.file_uploader("Upload your file", type=["pdf", "txt"])

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# File Processing
if uploaded_file:
    filename = uploaded_file.name
    with open(filename, "wb") as f:
        f.write(uploaded_file.read())

    ext = os.path.splitext(filename)[1]
    if ext == ".pdf":
        loader = PyMuPDFLoader(filename)
    else:
        loader = TextLoader(filename)

    st.success("✅ File uploaded successfully!")

    # Load and split the text
    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)

    # Embeddings
    st.info("🔍 Creating embeddings...")
    embedding = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(split_docs, embedding)

    # LLM
    st.info("🤖 Loading the language model...")
    llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_length=512, truncation=True)
    llm = HuggingFacePipeline(pipeline=llm_pipeline)

    # RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    # User Query
    query = st.text_input("🧠 Ask your question from the document:")

    if query:
        with st.spinner("Thinking..."):
            result = qa_chain({"query": query})
            st.session_state.chat_history.append((query, result["result"]))

    # Display Chat History
    for i, (q, a) in enumerate(st.session_state.chat_history):
        st.markdown(f"**🧑‍💻 You:** {q}")
        st.markdown(f"**🤖 AI:** {a}")
        st.markdown("---")
