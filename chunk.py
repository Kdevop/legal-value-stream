import json
import os
import chromadb
from typing import Dict, Any
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

class VectorStoreManager:
    """
    Manages the lifecycle of the ChromaDB vector collection, including 
    document chunking, embedding generation, and similarity retrieval.
    """

    def __init__(self):
        """Initialize ChromaDB and embedder."""
        # Environment variables
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.endpoint = os.getenv("ENDPOINT")
        
        # Initialize ChromaDB
        self.db = chromadb.Client()  # ✅ Create database FIRST
        
        # Load embedder
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Get or create collection
        self.collection = self.db.get_or_create_collection(
            name="employment_cases",
            metadata={"hnsw:space": "cosine"}
        )
        
        print(f"✅ VectorStoreManager initialized ({self.collection.count()} documents)")

    def load_cases(self, path: str = "employment_cases.json"):
        """
        Loads case data from JSON, processes text into embeddable chunks,
        and performs batch insertion into the vector database.
        """
        if not os.path.exists(path):
            print(f"❌ Error: File {path} not found.")
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cases = data.get("cases", [])
        chunks = []

        for i, item in enumerate(cases):
            title = item.get("title", "")
            published = item.get("published", "")
            content = item.get("full_text", "")

            # Construct the text block for embedding
            text = (
                f"title: {title}\n"
                f"published: {published}\n"
                f"content: {content}"
            )

            # Generate a clean ID: case index + sanitized title snippet
            sanitized_title = title[:50].replace(' ', '_')
            chunk = {
                "id": f"case_{i}_{sanitized_title}",
                "text": text,
                "metadata": {
                    "title": title,
                    "published": published,
                    "content": content
                }
            }
            chunks.append(chunk)

        # Batch insertion logic to optimize performance and memory
        batch_size = 50
        for batch_start in range(0, len(chunks), batch_size):
            batch = chunks[batch_start : batch_start + batch_size]

            batch_ids = [c["id"] for c in batch]
            batch_texts = [c["text"] for c in batch]
            batch_metadatas = [c["metadata"] for c in batch]

            # Generate embeddings for the current batch
            batch_embeddings = self.embedder.encode(batch_texts)

            self.collection.add(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas
            )

            current_progress = min(batch_start + batch_size, len(chunks))
            print(f"Inserted {current_progress}/{len(chunks)} chunks")

    def retrieve(self, question: str, n_results: int = 3) -> Dict[str, Any]:
        """
        Converts a natural language query into an embedding and retrieves
        the most similar case precedents from the collection.
        """
        query_embedding = self.embedder.encode(question).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        # Print formatted results to console for verification
        self._print_results(results)
        
        return results

    def _print_results(self, results: Dict[str, Any]):
        """Internal helper to display retrieval matches."""
        if not results.get("documents") or not results["documents"][0]:
            print("No matches found.")
            return

        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            similarity = 1 - dist
            print(f"\n[{i+1}] Similarity: {similarity:.3f}")
            print(f"     Title : {meta.get('title', '')[:80]}")
            print(f"     Date  : {meta.get('published', '')[:10]}")
            print(f"     Text  : {doc[:300]}...")


_store: VectorStoreManager = None


def _get_store() -> VectorStoreManager:
    """Lazy-initialise the singleton VectorStoreManager."""
    global _store
    if _store is None:
        _store = VectorStoreManager()
    return _store


def load_cases(path: str = "employment_cases.json"):
    """
    Public wrapper: load case law into the vector store.
    Called once on app startup from app.py.
    """
    store = _get_store()
    if store.collection.count() == 0:
        print("📊 Collection empty — loading cases...")
        store.load_cases(path)
    else:
        print(f"✅ Collection already loaded: {store.collection.count()} documents")


def assess_risk(text: str) -> dict:
    """
    Public wrapper: run a full risk assessment on raw contract text.
    1. Extracts clauses via ClauseExtractor (contract_processor.py)
    2. Enriches high-risk clauses with case law via RiskAnalyser (risk_analyser.py)
    Returns a dict ready to be JSON-serialised by Flask.
    """
    import os
    from contract_processor import ClauseExtractor
    from risk_analyser import RiskAnalyser

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return {"error": "OPENROUTER_API_KEY not set in environment"}

    store = _get_store()

    # Wrap plain text as a single-page structure ClauseExtractor expects
    pages = [{"page": 1, "text": text}]

    # Stage 1 - extract and classify clauses
    extractor = ClauseExtractor(api_key=api_key)
    extraction = extractor.extract_clauses(pages)
    clauses = extraction.get("clauses", [])

    if not clauses:
        return {"total_clauses": 0, "high_risk_count": 0, "clauses": []}

    # Stage 2 - enrich RED clauses with case law precedents
    analyser = RiskAnalyser(
        collection=store.collection,
        embedder=store.embedder,
        api_key=api_key
    )
    return analyser.analyse_all_clauses(clauses)


def main():
    """Entry point for testing the vector store."""
    store = _get_store()

    if store.collection.count() == 0:
        print("📊 Collection empty. Loading cases...")
        store.load_cases("employment_cases.json")
    else:
        print(f"✅ Collection already loaded: {store.collection.count()} documents")

    print("\n🔍 Running test retrieval...")
    store.retrieve("unfair dismissal procedural failure")


if __name__ == "__main__":
    main()