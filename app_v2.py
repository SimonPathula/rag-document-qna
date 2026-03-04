import streamlit as st
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

from dotenv import load_dotenv
load_dotenv()

#get the tokens and create the embedding
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
embedding = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

#title and short description of the app
st.title("RAG Chatbot with PDF uploads")
st.write("Upload the pdfs and chat with their content")

#enter the GROQ API key
groq_api_key = st.text_input("Enter your GROQ API key:", type= "password")

if groq_api_key:
    llm = ChatGroq(model_name= "openai/gpt-oss-120b", groq_api_key = groq_api_key)

    session_id = st.text_input("session_id:", value = "default_session")

    if "store" not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Choose pdf file", type= "pdf", accept_multiple_files= True)

    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temp_pdf = f"./temp.pdf"
            with open(temp_pdf, "wb") as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name

            loader = PyPDFLoader(temp_pdf)
            docs = loader.load()
            documents.extend(docs)
        
        splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
        splits = splitter.split_documents(documents)
        vectore_store = Chroma.from_documents(splits, embedding = embedding)
        retriever = vectore_store.as_retriever()

        contextualize_q_system_prompt = (
            '''
            Given a chat history and the latest user question which might reference context in 
            the chat history, formulate a standalone question which can be understood without 
            the chat history. DO NOT answer the question, just reformulate it if needed 
            otherwise return it as it is.
            '''
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("user", "{input}")
            ]
        )

        docs_related_to_standalone_q = (
            contextualize_q_prompt
            | llm
            | StrOutputParser()
            | retriever
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", '''You are helpful assistant that assists user on the uploaded documents.
                If the asked question is not in the relevant context of documents, answer "I don't 
                have enough knowledge on what you are asking, please ask relevant to the uploaded files"
                <context>{context}</context>'''),
                MessagesPlaceholder("chat_history"),
                ("user", "{input}")

            ]
        )

        answer_chain = (
            RunnablePassthrough.assign(context = docs_related_to_standalone_q)
        |  prompt
        | llm
        | StrOutputParser()  
        )

        def get_session_id(session_id):
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_chain = RunnableWithMessageHistory(
            answer_chain, 
            get_session_id,
            input_messages_key= "input",
            history_messages_key= "chat_history"
        )

        user_input = st.text_input("Enter you question:")

        if user_input:
            response = conversational_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}  
            )
            st.write(response)
