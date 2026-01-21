A fully local Retrieval-Augmented Generation (RAG)
This project uses Ollama : LLaMA 2, all-minilm and ChromaDB with a Streamlit-based UI.

FEATURES:
- Upload PDF documents
- Ask questions based on uploaded PDF content
- Semantic search using embeddings
- Context-aware answers from document data
- Fully local execution (no OpenAI or external APIs)
- Fast and privacy-preserving

ARCHITECTURE:
User Query
↓
Embeddings (all-minilm)
↓
ChromaDB (Vector Store)
↓
Relevant Context
↓
LLaMA 2 (Answer Generation)

DOWNLOAD REQUIRED MODELS:
Ensure Ollama is installed and running:

ollama pull llama2
ollama pull nomic-embed-text
