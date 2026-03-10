import json
import os
import re
import requests
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Configuration and Constants
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ENDPOINT = os.getenv("ENDPOINT")
COLLECTION_NAME = "employment_cases"
MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL = "mistralai/ministral-3b-2512"


class CaseManager:
    """
    Handles data ingestion, embedding, and vector storage for employment cases.
    """

    def __init__(self):
        self.db = chromadb.Client()
        self.embedder = SentenceTransformer(MODEL_NAME)
        self.collection = self.db.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def load_cases(self, path="employment_cases.json"):
        """
        Loads cases from JSON, chunks them, and upserts them into ChromaDB.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cases = data["cases"]
        chunks = []

        for i, item in enumerate(cases):
            title = item.get("title", "")
            published = item.get("published", "")
            content = item.get("full_text", "")

            # Text that will be embedded
            text = (
                f"title: {title}\n"
                f"published: {published}\n"
                f"content: {content}"
            )

            chunk = {
                "id": f"case_{i}_{title[:50].replace(' ', '_')}",
                "text": text,
                "metadata": {
                    "title": title,
                    "published": published,
                    "content": content
                }
            }
            chunks.append(chunk)

        batch_size = 50
        for batch_start in range(0, len(chunks), batch_size):
            batch = chunks[batch_start : batch_start + batch_size]

            batch_ids = [c["id"] for c in batch]
            batch_texts = [c["text"] for c in batch]
            batch_metadatas = [c["metadata"] for c in batch]

            # Generate embeddings for this batch
            batch_embeddings = self.embedder.encode(batch_texts)

            self.collection.add(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas
            )

            print(f"Inserted {min(batch_start + batch_size, len(chunks))}/{len(chunks)} chunks")

    def retrieve(self, question, n_results=3):
        """
        Performs similarity search against the vector collection.
        """
        query_embedding = self.embedder.encode(question).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            print(f"\n[{i+1}] Similarity: {1-dist:.3f}")
            print(f"     Title : {meta['title'][:80]}")
            print(f"     Date  : {meta['published'][:10]}")
            print(f"     Text  : {doc[:300]}...")


class RiskAssessor:
    """
    Combines vector retrieval and LLM completion to perform legal risk assessment.
    """

    def __init__(self, case_manager: CaseManager, api_key: str):
        self.manager = case_manager
        self.llm = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    def assess_risk(self, new_case_description):
        """
        Retrieves context and generates a structured risk assessment report.
        """
        # 1. Retrieve relevant precedents from ChromaDB
        query_embedding = self.manager.embedder.encode(new_case_description).tolist()
        results = self.manager.collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["documents", "metadatas"]
        )

        # 2. Format retrieved chunks as context
        context = ""
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            context += f"CASE: {meta['title']}\n"
            context += f"DATE: {meta['published'][:10]}\n"
            context += f"EXCERPT: {doc[:600]}\n"
            context += "---\n"

        # 3. Build prompt
        prompt = f"""You are an employment law risk assessor.

Using these relevant precedent cases as context:

{context}

Assess the following case and return ONLY a JSON object:

Case to assess:
{new_case_description}

Return this exact JSON structure:
{{
"is_high_risk": true or false,
"risk_level": "low" or "medium" or "high",
"risk_reasons": ["reason 1", "reason 2"],
"similar_precedents": ["case title 1", "case title 2"],
"recommended_action": "what to do next"
}}"""

        # 4. Call the LLM
        model_resp = self.llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are an employment law risk assessor."},
                {"role": "user", "content": prompt}
            ],
        )

        raw = model_resp.choices[0].message.content

        # Clean response format and control characters
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        raw = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', raw)

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return json.loads(raw, strict=False)


def main():
    # Setup
    manager = CaseManager()
    assessor = RiskAssessor(manager, OPENROUTER_API_KEY)
    
    # Optional: load data
    # manager.load_cases()

    # Search test
    print("Testing Retrieval...")
    manager.retrieve("unfair dismissal procedural failure")

if __name__ == "__main__":
    main()