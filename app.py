import streamlit as st
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA


st.set_page_config(page_title="PDF RAG with LLaMA2", layout="wide")
st.title("PDF QA")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
user_question = st.text_input("Ask a question from the PDF:")


embeddings = OllamaEmbeddings(
    model="all-minilm"
)

llm = Ollama(
    model="llama2",
    temperature=0
)


if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    splits = text_splitter.split_documents(documents)


    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )


    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )

    st.success("PDF loaded and indexed successfully!")


    if user_question:
        with st.spinner("Thinking..."):
            answer = qa_chain.invoke({"query": user_question})

        st.subheader("Answer:")
        st.write(answer)

    os.remove(pdf_path)