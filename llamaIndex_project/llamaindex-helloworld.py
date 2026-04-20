from dotenv import load_dotenv
import os
from llama_index.core import SimpleDirectoryReader

# core data structures -- docs and settings
from llama_index.core import Document
from llama_index.core import Settings

# Text splitters
from llama_index.core.node_parser import SentenceSplitter

# embedding models
from llama_index.embeddings.openai import OpenAIEmbedding

# Index creation - Vector Store Index
from llama_index.core import VectorStoreIndex

# LLm configuration -- OpenAI
from llama_index.llms.openai import OpenAI

from llama_index.core.ingestion import IngestionPipeline

from llama_index.core.extractors import (
    TitleExtractor,
    SummaryExtractor,
    KeywordExtractor,
)
from llama_index.core.storage.docstore import SimpleDocumentStore

# For persistent vector store
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()

Settings.llm = OpenAI(model="llama3.1:latest")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.chunk_size = 512
Settings.chunk_overlap = 50

PERSISTENCE_DIR = "./pipeline_storage"
CHROMA_DIR = "./chroma_db"


def get_transformations():
    """Return transformations - must be identical for cache to work."""
    return [
        SentenceSplitter(
            chunk_size=Settings.chunk_size,
            chunk_overlap=Settings.chunk_overlap,
        ),
        TitleExtractor(),
        # SummaryExtractor(),  # Uncomment for more metadata (slower)
        # KeywordExtractor(),  # Uncomment for more metadata (slower)
        OpenAIEmbedding(model=Settings.embed_model.model_name),
    ]


def main():
    print("Loading documents...")
    print("Files:", os.listdir("llamaindex-docs"))
    documents = SimpleDirectoryReader(
        input_dir="llamaindex-docs", required_exts=[".txt"], num_files_limit=10
    ).load_data()
    print(f"Loaded {len(documents)} documents.")

    # Create persistent Chroma vector store
    print("Setting up ChromaDB vector store...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    chroma_collection = chroma_client.get_or_create_collection("llamaindex_docs")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Check how many docs already in vector store
    existing_count = chroma_collection.count()
    print(f"ChromaDB already contains {existing_count} embeddings.")

    # If we already have embeddings, skip ingestion and go straight to querying
    if existing_count > 0:
        print("Using existing embeddings from ChromaDB (skipping ingestion).")
    else:
        # Create and run the ingestion pipeline
        print("Creating ingestion pipeline...")
        pipeline = IngestionPipeline(
            transformations=get_transformations(),
            vector_store=vector_store,
        )

        print("Running ingestion pipeline...")
        processed_nodes = pipeline.run(documents=documents, show_progress=True)
        print(f"Processed {len(processed_nodes)} nodes into ChromaDB.")

        # Show metadata from first node
        if processed_nodes:
            if processed_nodes[0].embedding:
                print(f"Embedding dimensions: {len(processed_nodes[0].embedding)}")

            first_node_metadata = processed_nodes[0].metadata
            print("First node metadata:")
            for key, value in first_node_metadata.items():
                print(f"  {key}: {value}")

    # Create index from the vector store
    print("Creating vector store index from ChromaDB...")
    vector_index = VectorStoreIndex.from_vector_store(vector_store)
    print("Vector store index created.")

    # Create query engine
    query_engine = vector_index.as_query_engine()

    # Sample query
    print("\n--- Query Test ---")
    response = query_engine.query("What is the SimpleDirectoryReader?")
    print("Query: What is the SimpleDirectoryReader?")
    print(f"Response:\n{response}")


if __name__ == "__main__":
    main()