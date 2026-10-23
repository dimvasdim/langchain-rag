# langchain-rag

Code from the freeCodeCamp course [Production RAG with LangChain & Vector Databases](https://www.youtube.com/watch?v=mHxLXzYjQRE), adapted to run on local [Ollama](https://ollama.com) models, with Gemini as an optional provider.

## Setup

1. Install [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com).
2. Pull the models:
   ```sh
   ollama pull qwen3.5:4b
   ollama pull qwen2.5:7b       # only used by the model-routing demo
   ollama pull embeddinggemma
   ```
3. Install dependencies:
   ```sh
   uv sync
   ```
4. Optional: copy `.env.example` to `.env` to switch to Gemini (`LLM_PROVIDER=google` + `GOOGLE_API_KEY`), override model names, or enable LangSmith tracing.

Check the setup with:

```sh
uv run python src/main.py
```

## Configuration

All lessons get their models from [src/config.py](src/config.py):

- `get_llm()`: chat model (`qwen3.5:4b`, or Gemini)
- `get_embeddings()`: embedding model (`embeddinggemma`, or Gemini embeddings at 768 dims). For embeddinggemma it adds the task prefixes the model expects.
- `CHROMA_METADATA`: every Chroma collection uses cosine distance

## Lessons

Run each lesson from the repo root, e.g. `uv run python src/rag_pipeline.py`. Most files have several demos; to pick one, comment or uncomment the calls in the `if __name__ == "__main__":` block.

On Windows, if you redirect output to a file (`> out.txt`), set `PYTHONUTF8=1` first. Otherwise the box-drawing characters and emoji in some lessons raise `UnicodeEncodeError`.

| File | Topic |
|---|---|
| `document_loaders.py` | Text, web, directory and PDF loaders |
| `text_splitters.py` | Recursive, markdown, code and PDF splitting |
| `embeddings.py` | Embeddings, cosine similarity, embedding cache |
| `vector_stores.py` | Chroma basics, scores, filters, retrievers, persistence |
| `rag_pipeline.py` | Basic RAG, sources, fallback, structured output |
| `advanced_rag.py` | Multi-query, compression, ensemble, parent-document |
| `semantic_chunking.py` | Recursive vs semantic chunking |
| `prod_hybrid_search.py` | BM25 + vector search with Reciprocal Rank Fusion |
| `final_prod_hybrid_search.py` | Reusable `HybridRetriever` class |
| `contextual_retrieval.py` | Anthropic's Contextual Retrieval |
| `late_chunking.py` | Late chunking concepts |
| `cost_optimization.py` | Model routing, semantic caching, token budgets |
| `monitoring.py` | JSON logging and metrics |
