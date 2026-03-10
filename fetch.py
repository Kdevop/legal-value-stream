import requests
import xmltodict
import json
import time
from lxml import etree
import os

# --- Configuration ---
BASE_ATOM_URL = "https://caselaw.nationalarchives.gov.uk/atom.xml"
BASE_XML_URL = "https://caselaw.nationalarchives.gov.uk"
TARGET_COUNT = 100
OUTPUT_FILE = "employment_cases.json"


class CaseFetcher:
    """
    Handles communication with the National Archives API to retrieve 
    and parse case law metadata and content.
    """

    def __init__(self, base_atom_url: str):
        self.base_atom_url = base_atom_url

    def fetch_atom_feed(self, page=1, per_page=50, query="employment"):
        params = {
            "query": query,
            "tribunal": "eat",
            "order": "-date",
            "page": page,
            "per_page": per_page
        }
        response = requests.get(self.base_atom_url, params=params)
        response.raise_for_status()
        return xmltodict.parse(response.content)

    def extract_metadata_from_entry(self, entry: dict) -> dict:
        links = entry.get('link', [])
        if isinstance(links, dict):
            links = [links]

        url_html = None
        url_xml = None
        url_pdf = None

        for link in links:
            rel = link.get('@rel', '')
            typ = link.get('@type', '')
            href = link.get('@href', '')
            if rel == 'alternate' and typ == 'application/akn+xml':
                url_xml = href
            elif rel == 'alternate' and typ == 'application/pdf':
                url_pdf = href
            elif rel == 'alternate' and 'html' in typ:
                url_html = href
            elif rel == 'alternate' and not typ:
                url_html = href

        return {
            "title": entry.get('title', ''),
            "published": entry.get('published', ''),
            "updated": entry.get('updated', ''),
            "summary": entry.get('summary', ''),
            "url_html": url_html,
            "url_xml": url_xml,
            "url_pdf": url_pdf,
            "court": entry.get('{https://caselaw.nationalarchives.gov.uk/}court', entry.get('tna:court', '')),
            "uri": entry.get('{https://caselaw.nationalarchives.gov.uk/}uri', entry.get('tna:uri', '')),
        }

    def fetch_full_text(self, xml_url: str):
        if not xml_url:
            return None
        try:
            response = requests.get(xml_url, timeout=30)
            response.raise_for_status()
            root = etree.fromstring(response.content)
            all_text = " ".join(root.itertext())
            return " ".join(all_text.split())
        except Exception as e:
            print(f"  ⚠️  Could not fetch XML for {xml_url}: {e}")
            return None


class CaseCollector:
    """
    Orchestrates the collection of case documents based on specific targets.
    """

    def __init__(self, fetcher: CaseFetcher):
        self.fetcher = fetcher

    def collect_cases(self, target, query="employment", tribunal="eat"):
        cases = []
        page = 1
        per_page = 50

        while len(cases) < target:
            print(f"\n📄 Fetching feed page {page}...")
            try:
                feed = self.fetcher.fetch_atom_feed(page=page, per_page=per_page, query=query)
            except Exception as e:
                print(f"  ❌ Feed fetch failed: {e}")
                break

            entries = feed.get('feed', {}).get('entry', [])
            if not entries:
                print("  No more entries found.")
                break

            if isinstance(entries, dict):
                entries = [entries]

            for entry in entries:
                if len(cases) >= target:
                    break

                print(f"  [{len(cases)+1}/{target}] Processing: {entry.get('title','?')[:60]}")
                metadata = self.fetcher.extract_metadata_from_entry(entry)
                full_text = self.fetcher.fetch_full_text(metadata.get('url_xml'))
                time.sleep(0.5)

                case_doc = {
                    **metadata,
                    "full_text": full_text,
                    "text_length": len(full_text) if full_text else 0,
                    "has_full_text": full_text is not None,
                }
                cases.append(case_doc)

            page += 1
            time.sleep(1)

        print(f"\n✅ Collected {len(cases)} cases.")
        return cases


class DataExporter:
    """
    Handles persistence of collected data to storage formats.
    """

    @staticmethod
    def save_to_json(cases, filename="employment_cases.json"):
        output = {
            "metadata": {
                "total_cases": len(cases),
                "source": "The National Archives - Find Case Law",
                "licence": "Open Justice Licence v1.0",
                "tribunal": "Employment Appeal Tribunal (EAT)",
                "purpose": "LLM grounding dataset for employment case risk assessment"
            },
            "cases": cases
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved {len(cases)} cases to '{filename}'")
        print(f"   File size: {os.path.getsize(filename) / 1024:.1f} KB")


def main():
    fetcher = CaseFetcher(BASE_ATOM_URL)
    collector = CaseCollector(fetcher)
    all_cases = collector.collect_cases(target=TARGET_COUNT)
    DataExporter.save_to_json(all_cases, OUTPUT_FILE)


if __name__ == "__main__":
    main()