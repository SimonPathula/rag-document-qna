import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv
load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model = "llama-3.1-8b-instant", groq_api_key = groq_api_key)

def create_vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.loader = PyPDFDirectoryLoader("rag_document_qna/research_papers")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.splitter = RecursiveCharacterTextSplitter(chunk_size = 700, chunk_overlap = 100)
        st.session_state.final_docs = st.session_state.splitter.split_documents(st.session_state.docs)
        st.session_state.embedding = HuggingFaceEmbeddings(model_name= "all-MiniLM-L6-v2")
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs, st.session_state.embedding)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", ''' Please answer the questions that are relevant to the pdfs.
        If question is not in relevance to the context just say "I dont have enough knowledge on what
        you are asking" '''),
        ("user", ''' <context> {context} </context> question: {user_input} ''')
    ]
)

st.title("RAG Document QnA with GROQ and Llama-3")
if st.button("Document Embeddings"):
    create_vector_embeddings()
    st.write("Vector database is ready")

user_prompt = st.text_input("Enter your query from the research papers")

import time

if user_prompt:
    retriever = st.session_state.vectors.as_retriever()

    chain = (
    {
        "context": retriever,
        "user_input" : RunnablePassthrough() 
    }
    | prompt
    | llm
    |StrOutputParser()
    )
    start = time.process_time()
    response = chain.invoke(user_prompt)
    st.write(response)
    print(f"Response time: {time.process_time() - start}")

    with st.expander("Document similarity search"):
        similar_docs = retriever.invoke(user_prompt)
        for i, doc in enumerate(similar_docs):
            st.write(doc.page_content)
            st.write("-" * 30)