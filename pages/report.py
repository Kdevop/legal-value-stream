import streamlit as st
import re

st.title("⚠️ Risk Report")

if not st.session_state.contract_text:
    st.warning("No contract uploaded yet. Please upload a document first.")
    if st.button("Go to Upload"):
        st.switch_page("pages/upload.py")
    st.stop()

# Analyze contract for risk flags
def analyze_contract(text):
    text_lower = text.lower()
    flags = {}
    
    # Liability Flag
    liability_keywords = ["unlimited", "unlimited liability", "indemnify", "indemnification"]
    liability_matches = [kw for kw in liability_keywords if kw in text_lower]
    flags["liability"] = {
        "status": "red" if liability_matches else "green",
        "title": "Unlimited Liability",
        "description": "Contract may contain unlimited liability clauses",
        "keywords": liability_matches,
        "citation": "St Albans City and District Council v International Computers Ltd [1996] 4 All ER 481"
    }
    
    # Data Training Flag
    training_keywords = ["train", "training", "machine learning", "ai training", "model training"]
    training_matches = [kw for kw in training_keywords if kw in text_lower]
    flags["data_training"] = {
        "status": "red" if training_matches else "green",
        "title": "Data Training Rights",
        "description": "Vendor may use your data for AI/ML training",
        "keywords": training_matches,
        "citation": "Data Protection Act 2018, Section 86"
    }
    
    # Termination Flag
    termination_keywords = ["immediate termination", "terminate immediately", "without notice"]
    termination_matches = [kw for kw in termination_keywords if kw in text_lower]
    flags["termination"] = {
        "status": "yellow" if termination_matches else "green",
        "title": "Termination Clauses",
        "description": "Contract contains concerning termination terms",
        "keywords": termination_matches,
        "citation": "Unfair Contract Terms Act 1977"
    }
    
    # Governing Law Flag
    law_keywords = ["new york", "delaware", "california", "foreign jurisdiction"]
    law_matches = [kw for kw in law_keywords if kw in text_lower]
    flags["governing_law"] = {
        "status": "yellow" if law_matches else "green",
        "title": "Governing Law",
        "description": "Contract governed by non-UK jurisdiction",
        "keywords": law_matches,
        "citation": "Rome I Regulation (EC) No 593/2008"
    }
    
    return flags

# Run analysis
flags = analyze_contract(st.session_state.contract_text)
st.session_state.flags = flags

# Count high-risk flags
red_flags = sum(1 for f in flags.values() if f["status"] == "red")
yellow_flags = sum(1 for f in flags.values() if f["status"] == "yellow")

# Display verdict
if red_flags > 0:
    st.error(f"🚨 {red_flags} High-Risk Flag(s) Found")
elif yellow_flags > 0:
    st.warning(f"⚠️ {yellow_flags} Medium-Risk Flag(s) Found")
else:
    st.success("✅ No Major Risk Flags Detected")

st.divider()

# Display flag cards
for flag_key, flag_data in flags.items():
    status_icon = "🔴" if flag_data["status"] == "red" else "🟡" if flag_data["status"] == "yellow" else "🟢"
    
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(f"{status_icon} {flag_data['title']}")
        with col2:
            st.caption(flag_data["status"].upper())
        
        st.write(flag_data["description"])
        
        if flag_data["keywords"]:
            with st.expander("🔍 View Details & Citations"):
                st.write("**Detected Keywords:**")
                st.write(", ".join(f"`{kw}`" for kw in flag_data["keywords"]))
                
                st.write("**Contract Snippets:**")
                # Find and display snippets
                text_lower = st.session_state.contract_text.lower()
                for keyword in flag_data["keywords"][:2]:  # Show max 2 snippets
                    idx = text_lower.find(keyword)
                    if idx != -1:
                        start = max(0, idx - 100)
                        end = min(len(st.session_state.contract_text), idx + len(keyword) + 100)
                        snippet = st.session_state.contract_text[start:end]
                        st.info(f"...{snippet}...")
                
                st.write("**Legal Citation:**")
                st.markdown(f"[{flag_data['citation']}](https://www.legislation.gov.uk/)")

st.divider()

if st.button("← Upload Another Document"):
    st.session_state.contract_text = ""
    st.session_state.analysis_done = False
    st.switch_page("pages/upload.py")
