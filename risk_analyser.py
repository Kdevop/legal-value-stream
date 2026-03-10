"""
Risk Analyser - Enriches contract clauses with case law precedents
"""

import json
from typing import List, Dict
from openai import OpenAI


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
        if clause.get('risk_level') not in ['HIGH', 'RED']:
            return {**clause, "precedents": [], "legal_analysis": None}

        # Query vector DB
        query_text = f"{clause.get('type', '')}: {clause.get('text', '')}"
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
                "excerpt": doc[:400]
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
CURRENT RISK: {clause.get('risk_level', 'UNKNOWN')}
REASON: {clause.get('risk_explanation', '')}

Return ONLY valid JSON:
{{
  "legal_risk_confirmed": true,
  "precedent_support": ["how precedent 1 relates", "how precedent 2 relates"],
  "recommended_action": "specific action to take",
  "alternative_wording": "safer wording or null"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a UK contract law expert. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )

            raw = response.choices[0].message.content
            raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

            return json.loads(raw)

        except Exception as e:
            print(f"⚠️ LLM analysis error: {e}")
            return None

    def analyse_all_clauses(self, clauses: List[dict]) -> Dict:
        """Process all clauses from ClauseExtractor."""

        enriched_clauses = []

        for clause in clauses:
            enriched = self.analyse_clause(clause)
            enriched_clauses.append(enriched)

        high_risk_count = sum(1 for c in enriched_clauses if c.get('risk_level') in ['HIGH', 'RED'])

        return {
            "total_clauses": len(enriched_clauses),
            "high_risk_count": high_risk_count,
            "clauses": enriched_clauses
        }
