# SAP SD Knowledge Agent - Streamlit Cloud App (Stable Version)
# Features: Chat interface, SAP Help content search, source paragraph display, bookmark option

import streamlit as st
import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# Validate API key
if not openai_key:
    st.error("OPENAI_API_KEY is missing. Please set it in Streamlit Secrets.")
    st.stop()

st.set_page_config(page_title="SAP SD Knowledge Agent", layout="wide")
st.title("📘 SAP SD Knowledge Agent")

# Initialize models
embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
llm = ChatOpenAI(temperature=0.2, model_name="gpt-3.5-turbo", openai_api_key=openai_key)

# Load and split documents
@st.cache_resource
def load_data():
    sap_sd_urls = [
        "https://help.sap.com/docs/SAP_S4HANA_CLOUD/8e2d6b96f9ef4c3da8c8eac12fd412c4/8f3b6e597d6f4904b94b47a4ed4d25d7.html",
        "https://help.sap.com/docs/SAP_S4HANA_CLOUD/8e2d6b96f9ef4c3da8c8eac12fd412c4/64d1a90f52d342f5a47293b57a3d6b02.html"
    ]
    loader = WebBaseLoader(sap_sd_urls)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)
    vectordb = Chroma.from_documents(split_docs, embedding=embeddings, persist_directory="./chromadb")
    return vectordb

vectordb = load_data()
retriever = vectordb.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

# Sidebar options
st.sidebar.header("🔖 Bookmarks")
if "bookmarks" not in st.session_state:
    st.session_state.bookmarks = []

# User input
query = st.text_input("Ask a question about SAP SD:", placeholder="e.g., What is PGI in SAP?")

if query:
    with st.spinner("Thinking..."):
        result = qa_chain(query)
        st.markdown(f"**💬 Answer:** {result['result']}")

        with st.expander("📚 Source Paragraph(s)"):
            for i, doc in enumerate(result['source_documents']):
                st.markdown(f"**Source {i+1}:** {doc.page_content[:600]}...")

        if st.button("🔖 Bookmark this answer"):
            st.session_state.bookmarks.append({"question": query, "answer": result['result']})
            st.success("Bookmarked!")

# Display bookmarks
if st.sidebar.button("Refresh Bookmarks"):
    st.rerun()
for bm in st.session_state.bookmarks:
    st.sidebar.markdown(f"**Q:** {bm['question']}\n\n➡️ {bm['answer'][:150]}...")
