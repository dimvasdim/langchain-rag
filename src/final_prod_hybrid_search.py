import uuid
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from config import CHROMA_METADATA, get_embeddings
from prod_hybrid_search import documents, hybrid_retrieve


class HybridRetriever:
    """Production hybrid retriever with BM25 + Vector search"""

    def __init__(self, documents: list[Document], bm25_weight: float = 0.5, k: int = 4):
        self.k = k
        self.bm25_weight = bm25_weight
        self.vector_weight = 1 - bm25_weight
        # Keep the full Documents so BM25 can be rebuilt with their metadata
        self.documents = list(documents)

        # Initialize embeddings
        self.embeddings = get_embeddings()

        # Create vector store and retriever
        self.vectorstore = Chroma.from_documents(
            self.documents,
            self.embeddings,
            collection_name=f"hybrid_search_{uuid.uuid4().hex[:8]}",
            collection_metadata=CHROMA_METADATA,
        )
        self.vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

        # Create BM25 retriever
        self.bm25_retriever = BM25Retriever.from_documents(self.documents, k=k)

    def search(self, query: str) -> list[Document]:
        """Run hybrid search using weighted RRF"""
        return hybrid_retrieve(
            query,
            retrievers=[self.bm25_retriever, self.vector_retriever],
            weights=[self.bm25_weight, self.vector_weight],
            k=self.k,
        )

    def add_documents(self, documents: list[Document]):
        """Add new documents to both retrievers"""
        # Add to vector store
        self.vectorstore.add_documents(documents)

        # Recreate BM25 (it doesn't support incremental adds)
        self.documents.extend(documents)
        self.bm25_retriever = BM25Retriever.from_documents(self.documents, k=self.k)


if __name__ == "__main__":
    retriever = HybridRetriever(documents, bm25_weight=0.5, k=4)
    results = retriever.search("SKU-7742X specifications")

    for doc in results:
        print(f"[{doc.metadata.get('type')}] {doc.page_content[:100]}")
