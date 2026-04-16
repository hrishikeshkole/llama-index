from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.chroma import ChromaVectorStore

import chromadb


# ----------------------------
# 1. Setup Models
# ----------------------------
def setup_models():
    Settings.llm = Ollama(model="qwen2.5:3b")
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    Settings.chunk_size = 512
    Settings.chunk_overlap = 20


# ----------------------------
# 2. Load Documents
# ----------------------------
def load_documents():
    return SimpleDirectoryReader(
        input_files=["./data/hrishi.txt"],
    ).load_data()


# ----------------------------
# 3. Create Ingestion Pipeline
# ----------------------------
def create_pipeline():
    return IngestionPipeline(
        transformations=[
            SentenceSplitter(
                chunk_size=Settings.chunk_size,
                chunk_overlap=Settings.chunk_overlap,
            ),
            Settings.embed_model,
        ]
    )


# ----------------------------
# 4. Setup ChromaDB
# ----------------------------
def setup_chroma():
    db = chromadb.Client(
        chromadb.Settings(persist_directory="./chroma_db")
    )

    collection = db.get_or_create_collection("hrishi_collection")

    vector_store = ChromaVectorStore(
        chroma_collection=collection
    )

    return vector_store


# ----------------------------
# 5. Build Index
# ----------------------------
def build_index(documents, pipeline, vector_store):
    nodes = pipeline.run(documents=documents)

    print("Nodes created:", len(nodes))  # debug

    index = VectorStoreIndex(
        nodes,
        vector_store=vector_store
    )

    return index


# ----------------------------
# 6. Query Engine
# ----------------------------
def query_index(index):
    query_engine = index.as_query_engine(
        llm=Settings.llm,
        embed_model=Settings.embed_model,
    )

    response = query_engine.query(
        "What is the profession of Hrishikesh?"
    )

    print("\nAnswer:\n", response)


# ----------------------------
# MAIN
# ----------------------------
def main():
    setup_models()

    documents = load_documents()
    pipeline = create_pipeline()
    vector_store = setup_chroma()  # ✅ fixed

    index = build_index(documents, pipeline, vector_store)

    query_index(index)


if __name__ == "__main__":
    main()