import requests
import xmltodict
import json
import time
from lxml import etree
import os

# --- Configuration ---
BASE_ATOM_URL = "https://caselaw.nationalarchives.gov.uk/atom.xml"
BASE_XML_URL  = "https://caselaw.nationalarchives.gov.uk"


TARGET_COUNT = 100
OUTPUT_FILE  = "employment_cases.json"



#Function to search the Atom feed and retrieve metadata for cases
def fetch_atom_feed(page=1, per_page=50, query="employment"):
    """
    Fetches one page of search results from the National Archives Atom feed.
    
    Parameters:
        page     : which page of results (starts at 1)
        per_page : results per page (max 50)
        query    : full-text search keyword
    
    Returns:
        Parsed dict of the Atom XML feed entries
    """
    params = {
        "query"    : query,
        "tribunal" : "eat",      # EAT = Employment Appeal Tribunal
        "order"    : "-date",    # newest first
        "page"     : page,
        "per_page" : per_page
    }
    
    response = requests.get(BASE_ATOM_URL, params=params)
    response.raise_for_status()  # Raises error if request failed
    
    # Parse XML response into a Python dict
    parsed = xmltodict.parse(response.content)
    return parsed


#Function to extract key metatdata fields from an Atom feed entry
def extract_metadata_from_entry(entry):

    # Handle multiple 'link' elements (alternate, PDF, XML)
    links = entry.get('link', [])
    if isinstance(links, dict):
        links = [links]  # normalise single link to list
    
    url_html = None
    url_xml  = None
    url_pdf  = None
    
    for link in links:
        rel  = link.get('@rel', '')
        typ  = link.get('@type', '')
        href = link.get('@href', '')
        if rel == 'alternate' and typ == 'application/akn+xml':
            url_xml = href
        elif rel == 'alternate' and typ == 'application/pdf':
            url_pdf = href
        elif rel == 'alternate' and 'html' in typ:
            url_html = href
        elif rel == 'alternate' and not typ:
            url_html = href  # default alternate = HTML page
    
    return {
        "title"       : entry.get('title', ''),
        "published"   : entry.get('published', ''),
        "updated"     : entry.get('updated', ''),
        "summary"     : entry.get('summary', ''),
        "url_html"    : url_html,
        "url_xml"     : url_xml,
        "url_pdf"     : url_pdf,
        "court"       : entry.get('{https://caselaw.nationalarchives.gov.uk/}court', 
                                   entry.get('tna:court', '')),
        "uri"         : entry.get('{https://caselaw.nationalarchives.gov.uk/}uri', 
                                   entry.get('tna:uri', '')),
    }


#extract all text from the xml of each case
def fetch_full_text(xml_url):
    
    if not xml_url:
        return None
    
    try:
        response = requests.get(xml_url, timeout=30)
        response.raise_for_status()
        
        # Parse XML and extract all text nodes from the judgment body
        root = etree.fromstring(response.content)
        
        # LegalDocML uses namespaces — extract all text
        all_text = " ".join(root.itertext())
        
        # Clean up excess whitespace
        full_text = " ".join(all_text.split())
        return full_text
    
    except Exception as e:
        print(f"  ⚠️  Could not fetch XML for {xml_url}: {e}")
        return None



#collects data from set number of cases, including metadata and full text, and returns a list of dicts
def collect_cases(target, query="employment", tribunal="eat"):
  
    cases    = []
    page     = 1
    per_page = 50  # max allowed per page

    while len(cases) < target:
        print(f"\n📄 Fetching feed page {page}...")
        
        try:
            feed = fetch_atom_feed(page=page, per_page=per_page, query=query)
        except Exception as e:
            print(f"  ❌ Feed fetch failed: {e}")
            break
        
        entries = feed.get('feed', {}).get('entry', [])
        if not entries:
            print("  No more entries found.")
            break
        
        # Normalise: single entry comes back as dict, not list
        if isinstance(entries, dict):
            entries = [entries]
        
        for entry in entries:
            if len(cases) >= target:
                break
            
            print(f"  [{len(cases)+1}/{target}] Processing: {entry.get('title','?')[:60]}")
            
            metadata = extract_metadata_from_entry(entry)
            
            # Fetch full text (be polite  wait between requests)
            full_text = fetch_full_text(metadata.get('url_xml'))
            time.sleep(0.5)  # 0.5s pause between requests — stay within rate limits
            
            case_doc = {
                **metadata,
                "full_text"      : full_text,
                "text_length"    : len(full_text) if full_text else 0,
                "has_full_text"  : full_text is not None,
            }
            cases.append(case_doc)
        
        page += 1
        time.sleep(1)  # pause between pages
    
    print(f"\n✅ Collected {len(cases)} cases.")
    return cases

# Run it
all_cases = collect_cases(target=100)




def save_to_json(cases, filename="employment_cases.json"):
    """
    Saves the list of case dicts to a JSON file formatted for LLM grounding.
    """
    output = {
        "metadata": {
            "total_cases"  : len(cases),
            "source"       : "The National Archives - Find Case Law",
            "licence"      : "Open Justice Licence v1.0",
            "tribunal"     : "Employment Appeal Tribunal (EAT)",
            "purpose"      : "LLM grounding dataset for employment case risk assessment"
        },
        "cases": cases
    }
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Saved {len(cases)} cases to '{filename}'")
    print(f"   File size: {os.path.getsize(filename) / 1024:.1f} KB")


save_to_json(all_cases, OUTPUT_FILE)