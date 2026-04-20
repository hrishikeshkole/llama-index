# LlamaIndex Local Setup Project

This project demonstrates how to build a local Retrieval-Augmented Generation (RAG) pipeline using **LlamaIndex**, **Ollama**, and **ChromaDB**. It uses local models for both language generation and embeddings, ensuring privacy and cost-free execution.

## Project Structure

- `llamaindex-helloworld.py`: A basic LlamaIndex script that loads documents, processes them using local models, stores the embeddings in ChromaDB, and performs a simple query.
- `ingestion_pipeline_cached.py`: An advanced version of the pipeline that incorporates LlamaIndex caching. This prevents redundant processing and API calls for documents that have already been ingested, saving time on subsequent runs.
- `llamaindex-docs/`: Directory where your source text files should be placed.
- `chroma_db/` & `chroma_db_cached/`: Local directories where ChromaDB persists the generated embeddings.
- `pipeline_cache/`: Local directory where document processing caches are stored.
- `pyproject.toml` & `uv.lock`: Dependency management files using `uv`.

## Prerequisites

1. **Python 3.10+**: Required for ChromaDB compatibility.
2. **uv**: The modern, fast Python package installer and resolver.
3. **Ollama**: You must have [Ollama](https://ollama.com/) installed and running locally.

### Required Local Models

Before running the scripts, you need to pull the required models via Ollama. Open a terminal and run:

```bash
# Pull the LLM for text generation and metadata extraction
ollama pull llama3.1:latest

# Pull the embedding model
ollama pull nomic-embed-text:latest
```

## Setup & Installation

1. Clone or download this project.
2. Open your terminal in the project directory.
3. Use `uv` to sync and install all required dependencies:

```bash
uv sync
```

Alternatively, you can run scripts directly using `uv run` and it will manage the virtual environment for you automatically.

## Usage

Place your text documents (`.txt` files) into the `llamaindex-docs` directory. 

### 1. Basic Pipeline

Run the standard ingestion and query pipeline:

```bash
uv run python llamaindex-helloworld.py
```

### 2. Cached Pipeline (Recommended)

Run the cached version of the pipeline. On the first run, it will process all documents. On subsequent runs, it will only process new or modified documents, drastically speeding up the ingestion step.

```bash
uv run python ingestion_pipeline_cached.py
```

## Troubleshooting

- **Connection Error**: If you see an error connecting to the LLM, ensure the Ollama app is running in the background.
- **Model Not Found**: If LlamaIndex complains about an unknown model, double-check that you ran the `ollama pull` commands listed in the Prerequisites.
