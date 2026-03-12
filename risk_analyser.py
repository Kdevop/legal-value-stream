"""
Risk Analyser - Enriches contract clauses with case law precedents
"""

import json
from typing import List, Dict, Optional
from openai import OpenAI
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor, as_completed

class PrecedentCitation(BaseModel):
    case_title: str
    year: int
    paragraph_reference: str
    relevance_score: float = Field(description="Why this case is relevant")

class AnalysisResult(BaseModel):
    legal_risk_confirmed: bool
    precedents: List[PrecedentCitation] = Field(default_factory=list) # Add this
    recommended_action: str
    alternative_wording: Optional[str] = None

class RiskAnalyser:
    """
    Enriches extracted contract clauses with relevant case law precedents.
    For HIGH-risk clauses, searches vector DB and gets LLM legal analysis.
    """

    def __init__(self, collection, embedder, api_key: str):
        self.collection = collection
        self.embedder = embedder
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.model = "openai/gpt-4o-mini"
        print(f"✅ RiskAnalyser initialised with {self.model}")

    def analyse_clause(self, clause: dict) -> dict:
        """Analyse a single clause with case law precedents."""

        # Skip non-high-risk clauses
        if clause.get('risk_flag') not in ['HIGH', 'RED']:
            return {**clause, "precedents": [], "legal_analysis": None}

        # Query vector DB
        query_text = f"{clause.get('clause_type', clause.get('type', ''))}: {clause.get('text', '')}"
        query_embedding = self.embedder.encode(query_text).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )

        # Format precedents
        precedents = []
        context = ""

        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0], 
            results["distances"][0]
        ):
            precedent = {
                "title": meta.get('title', 'Unknown'),
                "date": meta.get('published', '')[:10],
                "relevance_score": round((2 - dist) / 2, 3),
                "excerpt": doc[:400],
                "url_html": meta.get('url_html', '')
            }
            precedents.append(precedent)

            context += f"CASE: {meta.get('title', 'Unknown')}\n"
            context += f"DATE: {precedent['date']}\n"
            context += f"EXCERPT: {doc[:500]}\n---\n"

        # Get LLM analysis
        legal_analysis = self._get_legal_analysis(clause, context)

        return {
            **clause,
            "precedents": precedents,
            "legal_analysis": legal_analysis
        }

    def _get_legal_analysis(self, clause: dict, context: str) -> dict:
        """Use LLM to analyse clause with case law context."""

        prompt = f"""You are a UK contract law expert.

Based on these relevant case law precedents:
{context}

Analyse this contract clause:
TYPE: {clause.get('type', 'Unknown')}
TEXT: {clause.get('text', '')}
CURRENT RISK: {clause.get('risk_flag', 'UNKNOWN')}
REASON: {clause.get('risk_explanation', '')}

Return ONLY valid JSON matching this schema:
{{
  "legal_risk_confirmed": boolean,
  "precedents": [
    {{"case_title": "string", "year": integer, "paragraph_reference": "string", "relevance_score": float}}
  ],
  "recommended_action": "string",
  "alternative_wording": "string"
}}
"""

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a UK contract law expert. Return only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"},
                    max_tokens=1000        
                )

                raw = response.choices[0].message.content
                data = json.loads(raw)
                validated_analysis = AnalysisResult(**data)
                
                return validated_analysis.model_dump()
            
            except Exception as e:
                print(f"⚠️ Attempt {attempt+1} failed: {e}")
                if attempt == 2:
                    return None

    def analyse_all_clauses(self, clauses: List[dict]) -> Dict:
        enriched_clauses = [None] * len(clauses)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(self.analyse_clause, clause): i 
                    for i, clause in enumerate(clauses)}
            for future in as_completed(futures):
                i = futures[future]
                enriched_clauses[i] = future.result()

        high_risk_count = sum(1 for c in enriched_clauses if c.get('risk_flag') in ['HIGH', 'RED'])
        return {
            "total_clauses": len(enriched_clauses),
            "high_risk_count": high_risk_count,
            "clauses": enriched_clauses
        }
