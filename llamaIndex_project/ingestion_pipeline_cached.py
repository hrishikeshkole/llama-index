from dotenv import load_dotenv
import os
from llama_index.core import SimpleDirectoryReader

# core data structures -- docs and settings
from llama_index.core import Document
from llama_index.core import Settings

# Text splitters
from llama_index.core.node_parser import SentenceSplitter

# embedding models
from llama_index.embeddings.ollama import OllamaEmbedding

# Index creation - Vector Store Index
from llama_index.core import VectorStoreIndex

# LLm configuration -- Ollama
from llama_index.llms.ollama import Ollama

from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.storage.kvstore import SimpleKVStore

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

Settings.llm = Ollama(model="llama3.1:latest", request_timeout=360.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text:latest")
Settings.chunk_size = 512
Settings.chunk_overlap = 50

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "pipeline_cache")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db_cached")


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
        Settings.embed_model,
    ]


def main():
    print("=" * 60)
    print("Ingestion Pipeline with LlamaIndex Caching")
    print("=" * 60)

    # Load documents
    print("Loading documents...")
    docs_dir = os.path.join(BASE_DIR, "llamaindex-docs")
    print("Files:", os.listdir(docs_dir))
    documents = SimpleDirectoryReader(
        input_dir=docs_dir, required_exts=[".txt"], num_files_limit=10
    ).load_data()
    print(f"Loaded {len(documents)} documents.")

    # Create persistent Chroma vector store
    print("Setting up ChromaDB vector store...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    chroma_collection = chroma_client.get_or_create_collection("llamaindex_docs")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    print(f"ChromaDB path: {CHROMA_DIR}")
    print(f"Existing embeddings in ChromaDB: {chroma_collection.count()}")

    # Create explicit KV store for node transformation caching
    kv_store = SimpleKVStore()
    ingestion_cache = IngestionCache(cache=kv_store, collection="llama_cache")

    # Create and run the ingestion pipeline with caching
    print("Creating ingestion pipeline...")
    pipeline = IngestionPipeline(
        transformations=get_transformations(),
        vector_store=vector_store,
        docstore=SimpleDocumentStore(),  # Tracks document hashes to avoid re-reading files
        cache=ingestion_cache,           # Tracks node hashes to avoid re-running expensive LLM extractions
    )

    # Load existing cache if available
    if os.path.exists(CACHE_DIR):
        print(f"      Loading existing cache from {CACHE_DIR}...")
        pipeline.load(persist_dir=CACHE_DIR)
        print("      Cache loaded! Unchanged documents will be skipped.")
    else:
        print("      No existing cache found. Will process all documents.")

    # Run the pipeline - LlamaIndex will use cached transformations
    print("\n[4/6] Running ingestion pipeline...")
    print("      (Cached transformations will be reused - no redundant API calls)")

    import time

    start_time = time.time()
    processed_nodes = pipeline.run(
        documents=documents, show_progress=True, num_workers=4
    )
    elapsed = time.time() - start_time

    # Report results
    print(f"\n      Pipeline completed in {elapsed:.2f} seconds.")
    print(f"      Nodes returned: {len(processed_nodes)}")
    print(f"      Total embeddings in ChromaDB: {chroma_collection.count()}")

    # Show metadata from first processed node (if any)
    if processed_nodes:
        print("\n      Sample metadata from first NEW node:")
        if processed_nodes[0].embedding:
            print(
                f"        - Embedding dimensions: {len(processed_nodes[0].embedding)}"
            )
        first_node_metadata = processed_nodes[0].metadata
        for key, value in list(first_node_metadata.items())[:3]:
            print(f"        - {key}: {value}")

    # Persist cache for next run
    print(f"\n[5/6] Persisting cache to {CACHE_DIR}...")
    pipeline.persist(persist_dir=CACHE_DIR)
    print("      Cache saved! Next run will skip unchanged documents.")

    # Create index and query
    print("\n[6/6] Creating vector store index and testing query...")
    vector_index = VectorStoreIndex.from_vector_store(vector_store)
    query_engine = vector_index.as_query_engine()

    print("\n" + "=" * 60)
    print("Query Test")
    print("=" * 60)
    query = "What is LlamaIndex used for?"
    print(f"Query: {query}")
    response = query_engine.query(query)
    print(f"\nResponse:\n{response}")

if __name__ == "__main__":
    main()