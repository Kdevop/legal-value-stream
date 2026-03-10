import json
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import os
from dotenv import load_dotenv
from openai import OpenAI
import requests
from sentence_transformers import SentenceTransformer

load_dotenv()

# environment variables
api_key=os.getenv("API_KEY")
endpoint=os.getenv("ENDPOINT")
db = chromadb.Client()
embedder = SentenceTransformer("all-MiniLM-L6-v2")
collection = db.get_or_create_collection(
            name="employment_cases",
            metadata={"hnsw:space": "cosine"}
        )


# chat client
llm = OpenAI(api_key=api_key, base_url=endpoint)

#load data
def load_cases(path = "employment_cases.json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cases = data["cases"]

        # Create a series of chunks
        chunks = []

        for i, item in enumerate(cases):
            title     = item.get("title", "")
            published = item.get("published", "")
            content   = item.get("full_text", "")

            # Text that will be embedded
            text = (
                f"title: {title}\n"
                f"published: {published}\n"
                f"content: {content}"
            )

            chunk = {
                "id": f"case_{i}_{title[:50].replace(' ', '_')}",  # fix: truncate + no spaces
                "text": text,
                "metadata": {
                    "title"    : title,      
                    "published": published,  
                    "content"  : content     
                }
            }
            chunks.append(chunk)

        BATCH_SIZE = 50  # insert in batches to avoid memory issues

        for batch_start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[batch_start : batch_start + BATCH_SIZE]

            batch_ids       = [c["id"]       for c in batch]
            batch_texts     = [c["text"]     for c in batch]
            batch_metadatas = [c["metadata"] for c in batch]

            # Generate embeddings for this batch
            batch_embeddings = embedder.encode(batch_texts)

            collection.add(
                ids        = batch_ids,
                documents  = batch_texts,
                embeddings = batch_embeddings,
                metadatas  = batch_metadatas
            )

            print(f"Inserted {min(batch_start + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks")

def retrieve(question, n_results=3):
    query_embedding = embedder.encode(question).tolist()

    results = collection.query(
        query_embeddings = [query_embedding],
        n_results        = n_results,
        include          = ["documents", "metadatas", "distances"]
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

# Test it
retrieve("unfair dismissal procedural failure")

API = os.getenv("API_KEY")
llm = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API,
)

def assess_risk(new_case_description):
    # 1. Retrieve relevant precedents from ChromaDB
    query_embedding = embedder.encode(new_case_description).tolist()
    results = collection.query(
        query_embeddings = [query_embedding],
        n_results        = 5,
        include          = ["documents", "metadatas"]
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
    model_resp = llm.chat.completions.create(
        model="mistralai/ministral-3b-2512",
        messages=[
            {"role": "system", "content": "You are an employment law risk assessor."},
            {"role": "user", "content": prompt}  # fix: use prompt variable, not hardcoded text
        ],
    )

    # fix: OpenAI response syntax, not Anthropic's
    raw = model_resp.choices[0].message.content

    # fix: strip markdown code fences if model wraps response in ```json ... ```
    
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    # Remove invalid control characters
    import re
    raw = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(raw, strict=False)

# Test it
#result = assess_risk("""
   # Employee dismissed after 3 years service with no formal warnings.
   # Employer claims redundancy but hired a replacement 2 weeks later.
    #Employee is pregnant.
#""")

#print(json.dumps(result, indent=2))
