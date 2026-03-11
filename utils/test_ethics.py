#!/usr/bin/env python
"""
Test script to run ethics checks on actual contract analysis output.
Usage: python test_ethics.py [input_json] [output_dir]
"""

import json
import sys
from pathlib import Path
from ethics import run_ethics_checks, save_ethics_report

def main():
    # Default paths
    input_file = sys.argv[1] if len(sys.argv) > 1 else "../test_outputs/full_analysis.json"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "../test_outputs"
    
    # Load the actual contract analysis output
    print(f"📂 Loading analysis from: {input_file}")
    with open(input_file, 'r') as f:
        analysis_data = json.load(f)
    
    # Convert the analysis to a JSON string (as if it came from the AI)
    # Wrap in object with 'clauses' key to match AI output format
    ai_output = json.dumps({"clauses": analysis_data.get('clauses', [])})
    
    # Get contract text from first clause as sample context
    contract_text = " ".join([c.get('text', '') for c in analysis_data.get('clauses', [])])
    
    # Extract risk flags for grounding context
    risk_flags = [c.get('risk_flag') for c in analysis_data.get('clauses', []) if c.get('risk_flag')]
    
    print(f"📋 Found {len(analysis_data.get('clauses', []))} clauses to analyze")
    print(f"🚨 Risk flags: {set(risk_flags)}")
    print("\n⏳ Running ethics checks...")
    
    try:
        # Run comprehensive ethics checks.  The previous version deliberately
        # disabled grounding by passing an empty contract_ref_file (and no
        # context_chunks), which is why grounding always failed in the tests.
        # Use the default values instead so that reference clauses are loaded
        # from contract_reference_1.json and grounding can pass.
        report = run_ethics_checks(
            ai_output=ai_output,
            input_data=contract_text,
            risk_flags=risk_flags
            # context_chunks and contract_ref_file left to defaults
        )
        
        # Print summary to console
        print("\n" + "="*60)
        print("ETHICS REPORT SUMMARY")
        print("="*60)
        print(f"|Overall Status: {report['overall_status']}")
        
        checks = ['accuracy', 'grounding', 'bias', 'explainability', 'safety']
        for check in checks:
            status = report[check].get('status', 'UNKNOWN')
            symbol = "✓" if status == "PASS" else "✗"
            print(f"|{symbol} {check.upper()}: {status}")
            if status == "FAIL":
                issues = report[check].get('issues', report[check].get('accuracy_issues', []))
                for issue in issues[:2]:  # Show first 2 issues
                    print(f"|  - {issue[:80]}")
        
        print("="*60)
        
        # Save reports to files
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        json_file = output_path / "ethics_report.json"
        pdf_file = output_path / "ethics_report.pdf"
        
        save_ethics_report(
            report,
            json_filename=str(json_file),
        )
        
        print(f"\n✅ Ethics check complete!")
        print(f"📄 Reports saved to: {output_dir}/")
        
    except Exception as e:
        print(f"\n❌ Error running ethics checks: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
