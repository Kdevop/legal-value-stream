import re
import json
from pathlib import Path
from typing import List, Set
from datetime import datetime

# For advanced grounding
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_known_facts(case_file='employment_cases.json', include_flags: List[str] = None) -> Set[str]:
    """
    Loads known facts from employment_cases.json and optionally includes risk flags.
    Returns a set of words/phrases for checking.
    """
    known_facts = set()

    # Load case law
    case_path = Path(__file__).parent.parent / case_file
    if case_path.exists():
        with open(case_path, encoding='utf-8') as f:
            data = json.load(f)
            # Handle both direct list and {cases: [...]} structure
            cases = data.get('cases', data) if isinstance(data, dict) else data
            for case in cases:
                # Try 'full_text' first (from employment_cases.json), then 'text'
                text = case.get('full_text', case.get('text', ''))
                words = set(text.lower().split())
                known_facts.update(words)

    # Add risk flags if provided
    if include_flags:
        known_facts.update(include_flags)

    return known_facts

def load_contract_reference(contract_ref_file: str = 'contract_references/contract_reference_1.json') -> List[str]:
    """
    Loads contract clauses from the contract reference file.
    Returns a list of clause texts for use as grounding context.
    This serves as the 'ground truth' for validating AI-extracted clauses.
    """
    contract_clauses = []
    
    # Try absolute path first (for utils folder context)
    ref_path = Path(contract_ref_file)
    if not ref_path.exists():
        # Try relative to parent directory
        ref_path = Path(__file__).parent.parent / contract_ref_file
    
    if ref_path.exists():
        try:
            with open(ref_path, encoding='utf-8') as f:
                contract_data = json.load(f)
                clauses = contract_data.get('clauses', [])
                # Extract clause texts for grounding context
                for clause in clauses:
                    clause_text = clause.get('text', '')
                    if clause_text:
                        contract_clauses.append(clause_text)
        except Exception as e:
            print(f"⚠️  Warning: Could not load contract reference from {ref_path}: {e}")
    else:
        print(f"⚠️  Warning: Contract reference file not found at {ref_path}")
    
    return contract_clauses

# ============================================================
# ETHICS CHECK FUNCTIONS
# ============================================================

def check_accuracy(ai_output: str, risk_flags: List[str]) -> dict:
    """
    Checks if AI outputs are accurate and consistent (e.g., risk scores match flags).
    Parses the JSON output to verify per-clause alignment.
    """
    issues = []
    try:
        # Parse AI output as JSON (assuming it's the clauses dict)
        output_data = json.loads(ai_output)
        clauses = output_data.get('clauses', [])
        
        for clause in clauses:
            flag = clause.get('risk_flag')
            score = clause.get('risk_score', 0)
            
            # Check alignment: RED should have high score (7-10), GREEN low (1-3), etc.
            if flag == 'RED' and score < 7:
                issues.append(f'Inconsistent: RED flag with low score {score}')
            elif flag == 'GREEN' and score > 6:
                issues.append(f'Inconsistent: GREEN flag with high score {score}')
            elif flag == 'YELLOW' and (score < 4 or score > 8):
                issues.append(f'Inconsistent: YELLOW flag with outlier score {score}')
    
    except json.JSONDecodeError:
        issues.append('AI output is not valid JSON')
    
    return {'accuracy_issues': issues, 'is_accurate': len(issues) == 0}

# ============================================================
# ADVANCED GROUNDING CHECK (RAG-based)
# ============================================================

# Global models (load once)
_EMBEDDING_MODEL = None
_NLI_MODEL = None

def _get_embedding_model():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = SentenceTransformer("all-mpnet-base-v2")
    return _EMBEDDING_MODEL

def _get_nli_model():
    global _NLI_MODEL
    if _NLI_MODEL is None:
        _NLI_MODEL = pipeline("text-classification", model="facebook/bart-large-mnli")
    return _NLI_MODEL

def extract_claims(text: str) -> List[str]:
    """Extracts claims (sentences) from text."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def most_similar_chunk(claim: str, context_chunks: List[str]):
    """Finds the most similar context chunk for a claim."""
    model = _get_embedding_model()
    claim_emb = model.encode(claim, convert_to_tensor=True)
    chunk_embs = model.encode(context_chunks, convert_to_tensor=True)
    scores = util.cos_sim(claim_emb, chunk_embs)[0]
    best_idx = scores.argmax().item()
    return context_chunks[best_idx], float(scores[best_idx])

def check_entailment(claim: str, evidence: str):
    """Checks entailment between claim and evidence."""
    # Truncate inputs to fit model limits (BART max ~1024 tokens)
    max_length = 512  # Conservative limit
    claim_trunc = claim[:max_length] if len(claim) > max_length else claim
    evidence_trunc = evidence[:max_length] if len(evidence) > max_length else evidence
    
    model = _get_nli_model()
    result = model(f"{claim_trunc} </s> {evidence_trunc}")[0]
    return result["label"], float(result["score"])

def check_grounding(ai_output: str, context_chunks: List[str], contract_clauses: List[str] = None) -> dict:
    """
    Advanced grounding check using RAG: similarity + entailment.
    
    Strategy:
    1. First validate against contract_clauses (ground truth source)
    2. Then validate against context_chunks (case law/precedents)
    
    context_chunks: List of text chunks (e.g., from case law or contract).
    contract_clauses: List of actual contract clauses from reference file.
    """
    claims = extract_claims(ai_output)
    results = []

    for claim in claims:
        grounded = False
        similarity = 0
        evidence = ""
        entailment_label = "NEUTRAL"
        confidence = 0
        validation_source = "unknown"
        
        # Step 1: Check against contract clauses (if provided)
        if contract_clauses:
            evidence, similarity = most_similar_chunk(claim, contract_clauses)
            label, confidence = check_entailment(claim, evidence)
            
            # Contract clauses should have high similarity threshold (strict grounding)
            if (label == "ENTAILMENT") and (similarity > 0.7):
                grounded = True
                entailment_label = label
                validation_source = "contract_reference"
        
        # Step 2: Fallback to context chunks if not grounded by contract
        if not grounded and context_chunks:
            evidence, similarity = most_similar_chunk(claim, context_chunks)
            label, confidence = check_entailment(claim, evidence)
            
            # Precedents have slightly lower threshold
            if (label == "ENTAILMENT") and (similarity > 0.65):
                grounded = True
                entailment_label = label
                validation_source = "precedents"

        results.append({
            "claim": claim,
            "best_evidence": evidence,
            "similarity": similarity,
            "entailment_label": entailment_label,
            "confidence": confidence,
            "grounded": grounded,
            "validation_source": validation_source
        })

    return {
        "grounding_risk": any(not r["grounded"] for r in results),
        "claims": results
    }

def check_bias(ai_output: str) -> dict:
    """
    Analyzes for bias using sentiment analysis (requires transformers).
    """
    try:
        classifier = pipeline('sentiment-analysis')
        result = classifier(ai_output)
        bias_score = 1 - abs(result[0]['score'] - 0.5) * 2  # Normalize to 0-1 (higher = more biased)
        return {'bias_score': bias_score, 'label': result[0]['label']}
    except Exception as e:
        # Fallback to keyword check if model fails
        biased_keywords = ['unfair', 'discriminatory', 'biased', 'prejudiced']
        bias_score = sum(1 for word in biased_keywords if word in ai_output.lower()) / len(biased_keywords)
        return {'bias_score': bias_score, 'detected_terms': [w for w in biased_keywords if w in ai_output.lower()], 'error': str(e)}

def check_explainability(ai_output: str) -> dict:
    issues = []
    explainable = True
    total_explanations = 0
    avg_length = 0

    def is_circular_explanation(explanation: str, clause_text: str) -> bool:
        """
        Check if explanation is truly circular vs. valid premise-conclusion.

        Circular examples (BAD):
        - "This is risky because it's risky"
        - "The clause poses risks because it contains risks"

        Valid examples (GOOD):
        - "No safeguards pose risk of non-compliance" (problem → consequence)
        - "Client indemnifies provider for provider's negligence" (describes issue)
        """
        exp_lower = explanation.lower()
        clause_lower = clause_text.lower()

        # Extract key terms from clause
        clause_words = set(re.findall(r'\b\w+\b', clause_lower))

        # Extract key terms from explanation
        exp_words = set(re.findall(r'\b\w+\b', exp_lower))

        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        clause_words -= stop_words
        exp_words -= stop_words

        # Circular if explanation words are subset of clause words (no new information added)
        if exp_words.issubset(clause_words) and len(exp_words) > 3:
            return True

        # Check for tautological phrases (same concept repeated)
        tautology_patterns = [
            r'\brisky\b.*\bbecause\b.*\brisky\b',
            r'\brisks?\b.*\bbecause\b.*\brisks?\b',
            r'\bunfavorable\b.*\bbecause\b.*\bunfavorable\b',
            r'\bproblematic\b.*\bbecause\b.*\bproblematic\b',
            r'\bdangerous\b.*\bbecause\b.*\bdangerous\b',
            r'\bharmful\b.*\bbecause\b.*\bharmful\b'
        ]

        for pattern in tautology_patterns:
            if re.search(pattern, exp_lower):
                return True

        return False

    try:
        data = json.loads(ai_output)
        clauses = data.get("clauses", [])

        if not clauses:
            return {"explainable": False, "issues": ["No clauses found"]}

        explanations = []
        for c in clauses:
            exp = c.get("risk_explanation", "")
            clause_text = c.get("text", "")
            if not exp:
                explainable = False
                issues.append(f"Missing explanation for clause: {clause_text[:50]}...")
                continue

            explanations.append(exp)

            # Check substance (not too short)
            if len(exp.split()) < 6:
                explainable = False
                issues.append(f"Explanation too short: '{exp}'")

            # Check for true circular arguments (not premise-conclusion statements)
            if is_circular_explanation(exp, clause_text):
                explainable = False
                issues.append(f"Explanation may be circular: '{exp}'")

            # Check relevance (mentions a keyword from the clause)
            clause_keywords = set(re.findall(r"\b\w+\b", clause_text.lower())) - {"the", "and", "of", "to", "a", "an", "is", "are", "was", "were"}
            exp_keywords = set(re.findall(r"\b\w+\b", exp.lower()))
            if clause_keywords and not any(k in exp_keywords for k in clause_keywords):
                issues.append(f"Explanation may not reference clause content: '{exp}'")

        total_explanations = len(explanations)
        if explanations:
            avg_length = sum(len(e.split()) for e in explanations) / len(explanations)

    except json.JSONDecodeError:
        return {
            "explainable": False,
            "issues": ["Output is not valid JSON"],
            "total_explanations": 0,
            "avg_explanation_length": 0
        }

    return {
        "explainable": explainable and not issues,
        "total_explanations": total_explanations,
        "avg_explanation_length": avg_length,
        "issues": issues
    }

def check_safety(ai_output: str, known_facts: Set[str]) -> dict:
    """
    Checks for safety issues, like inventing case law or unsafe advice.
    """
    issues = []
    # Flag if output mentions case law not in known_facts
    case_mentions = re.findall(r'case\s+\w+|tribunal\s+\w+', ai_output.lower())
    for mention in case_mentions:
        if mention not in ' '.join(known_facts).lower():
            issues.append(f'Potentially invented case: {mention}')
    # General safety: Avoid advice on illegal actions
    if 'illegal' in ai_output.lower() or 'bypass' in ai_output.lower():
        issues.append('Potentially unsafe advice')
    return {'safety_issues': issues, 'is_safe': len(issues) == 0}

# ============================================================
# COMPREHENSIVE ETHICS CHECK
# ============================================================

def run_ethics_checks(ai_output: str, input_data: str, risk_flags: List[str] = None, context_chunks: List[str] = None, contract_ref_file: str = 'contract_references/contract_reference_1.json') -> dict:
    """
    Runs all ethics checks and returns a summary with pass/fail status.
    
    Parameters:
    - ai_output: The AI-generated audit report
    - input_data: The original contract processor output
    - risk_flags: Optional list of risk flag keywords
    - context_chunks: Optional list of text chunks for grounding (precedents/case law)
    - contract_ref_file: Path to the contract reference file for ground truth validation
    """
    known_facts = load_known_facts(include_flags=risk_flags or [])
    
    # Load contract clauses for grounding validation (break circular dependency)
    contract_clauses = load_contract_reference(contract_ref_file)
    
    if context_chunks is None:
        # Default: Use known_facts as chunks (split into sentences or words)
        context_chunks = list(known_facts)  # Simple list; could split further

    accuracy_result = check_accuracy(ai_output, risk_flags or [])
    # Pass both contract_clauses and context_chunks to grounding check
    grounding_result = check_grounding(ai_output, context_chunks, contract_clauses=contract_clauses)
    bias_result = check_bias(ai_output)
    explainability_result = check_explainability(ai_output)
    safety_result = check_safety(ai_output, known_facts)

    # Add pass/fail status to each check
    accuracy_result['status'] = 'PASS' if accuracy_result.get('is_accurate', False) else 'FAIL'
    grounding_result['status'] = 'PASS' if not grounding_result.get('grounding_risk', True) else 'FAIL'
    bias_result['status'] = 'PASS' if bias_result.get('bias_score', 0) < 0.5 else 'FAIL'
    explainability_result['status'] = 'PASS' if explainability_result.get('explainable', False) else 'FAIL'
    safety_result['status'] = 'PASS' if safety_result.get('is_safe', False) else 'FAIL'

    report = {
        'accuracy': accuracy_result,
        'grounding': grounding_result,
        'bias': bias_result,
        'explainability': explainability_result,
        'safety': safety_result,
        'timestamp': datetime.now().isoformat()
    }

    # Overall status: PASS only if all checks pass
    report['overall_status'] = 'PASS' if all(check.get('status') == 'PASS' for check in [accuracy_result, grounding_result, bias_result, explainability_result, safety_result]) else 'FAIL'
    
    return report

# ============================================================
# REPORTING FUNCTIONS
# ============================================================

def save_ethics_report_json(report: dict, filename: str = "ethics_report.json"):
    """
    Saves the ethics report to a JSON file.
    """
    with open(filename, "w") as f:
        json.dump(report, f, indent=4)
    print(f"✅ Ethics report (JSON) saved to {filename}")

def save_ethics_report(report: dict, json_filename: str = "ethics_report.json", pdf_filename: str = "ethics_report.pdf"):
    """
    Saves the ethics report to both JSON and PDF formats.
    """
    save_ethics_report_json(report, json_filename)

