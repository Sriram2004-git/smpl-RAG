from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.embeddings import Embeddings
from huggingface_hub import InferenceClient
import numpy as np

load_dotenv()

app = Flask(__name__)
CORS(app)

HF_MODEL = "flax-sentence-embeddings/all_datasets_v3_mpnet-base"
OLLAMA_MODEL = "qwen3-vl:235b-cloud"


class HFEmbeddings(Embeddings):
    def __init__(self):
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=os.environ["HF_TOKEN"],
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        result = self.client.feature_extraction(text, model=HF_MODEL)
        arr = np.array(result)
        if arr.ndim == 2:
            arr = arr.mean(axis=0)
        return arr.tolist()


@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "PDF RAG API is running"})


@app.route("/ask", methods=["POST"])
def ask():
    pdf_file = request.files.get("pdf")
    question = request.form.get("question", "").strip()

    if not pdf_file:
        return jsonify({"error": "No PDF file provided"}), 400
    if not question:
        return jsonify({"error": "No question provided"}), 400

    pdf_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            pdf_file.save(tmp)
            pdf_path = tmp.name

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(documents)

        embeddings = HFEmbeddings()
        vectordb = Chroma.from_documents(documents=splits, embedding=embeddings)

        llm = ChatOllama(
            model=OLLAMA_MODEL,
            temperature=0,
            base_url="https://api.ollama.com",
            headers={"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"},
        )

        retriever = vectordb.as_retriever(search_kwargs={"k": 3})

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the question using only the context below:\n\n{context}"),
            ("human", "{input}"),
        ])
        qa_chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))

        result = qa_chain.invoke({"input": question})
        return jsonify({"answer": result["answer"]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if pdf_path and os.path.exists(pdf_path):
            os.remove(pdf_path)


if __name__ == "__main__":
    app.run(debug=False)
