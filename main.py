"""
Main entry for lightweight RAG + Agent for 10-K files.
Place company 10-K PDFs into ./data with filenames like:
  MSFT_2023.pdf
  GOOGL_2024.pdf
  NVDA_2022.pdf
"""

import os
import re
import json
import math
from typing import List, Tuple, Dict, Any
import pdfplumber
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from tqdm import tqdm

DATA_DIR = "./data"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_TOKEN_ESTIMATE = 200  # roughly words per chunk
CHUNK_OVERLAP = 50  # words overlap
TOP_K = 5

# ---------------------------
# Utilities: text extraction
# ---------------------------
def extract_text_from_pdf(path: str) -> List[Tuple[int, str]]:
    """
    Returns list of (page_number, text) for each page.
    Handles FontBBox errors and other PDF parsing issues.
    """
    pages = []
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                try:
                    # Try standard text extraction first
                    text = page.extract_text() or ""
                    pages.append((i + 1, text))
                except Exception as e:
                    print(f"Warning: Error extracting text from page {i+1} in {path}: {e}")
                    try:
                        # Fallback: try extracting text with different method
                        text = page.extract_text_simple() or ""
                        pages.append((i + 1, text))
                        print(f"Used fallback extraction for page {i+1}")
                    except Exception as e2:
                        print(f"Fallback extraction also failed for page {i+1}: {e2}")
                        # Last resort: try to get any text using different approach
                        try:
                            # Try extracting with layout preservation disabled
                            text = page.extract_text(layout=False) or ""
                            pages.append((i + 1, text))
                            print(f"Used layout=False extraction for page {i+1}")
                        except:
                            # If all else fails, add empty page
                            pages.append((i + 1, ""))
                            print(f"Could not extract any text from page {i+1}")
    except Exception as e:
        print(f"Error opening PDF {path}: {e}")
        # Try alternative PDF libraries if pdfplumber fails completely
        success = False
        
        # Try PyMuPDF first (often more robust)
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(path)
            for i in range(len(doc)):
                try:
                    page = doc[i]
                    text = page.get_text() or ""
                    pages.append((i + 1, text))
                except:
                    pages.append((i + 1, ""))
            doc.close()
            print(f"Used PyMuPDF as fallback for {path}")
            success = True
        except ImportError:
            print("PyMuPDF not available for fallback")
        except Exception as e2:
            print(f"PyMuPDF fallback failed for {path}: {e2}")
        
        # Try PyPDF2 if PyMuPDF failed
        if not success:
            try:
                import PyPDF2
                with open(path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for i, page in enumerate(pdf_reader.pages):
                        try:
                            text = page.extract_text() or ""
                            pages.append((i + 1, text))
                        except:
                            pages.append((i + 1, ""))
                print(f"Used PyPDF2 as fallback for {path}")
                success = True
            except ImportError:
                print("PyPDF2 not available for fallback")
            except Exception as e2:
                print(f"PyPDF2 fallback failed for {path}: {e2}")
        
        if not success:
            print(f"All PDF extraction methods failed for {path}")
    
    return pages

def chunk_text(text: str, page_num:int, chunk_size_words=CHUNK_TOKEN_ESTIMATE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size_words, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        chunks.append({"page": page_num, "text": chunk_text})
        if end == len(words):
            break
        start = end - overlap
    return chunks

# ---------------------------
# Build corpus -> embeddings
# ---------------------------
class RAGIndex:
    def __init__(self, embed_model_name=EMBED_MODEL_NAME):
        self.model = SentenceTransformer(embed_model_name)
        self.texts = []  # list of dicts {company, year, page, text, filename}
        self.embeddings = None
        self.index = None

    def ingest_folder(self, folder=DATA_DIR):
        files = [f for f in os.listdir(folder) if f.lower().endswith(".pdf")]
        for fname in files:
            path = os.path.join(folder, fname)
            meta = self._parse_filename(fname)
            pages = extract_text_from_pdf(path)
            for page_num, page_text in pages:
                page_chunks = chunk_text(page_text, page_num)
                for ch in page_chunks:
                    record = {
                        "company": meta.get("company"),
                        "year": meta.get("year"),
                        "filename": fname,
                        "page": ch["page"],
                        "text": ch["text"]
                    }
                    self.texts.append(record)
        print(f"[ingest] total chunks: {len(self.texts)}")
        self._build_index()

    def _parse_filename(self, fname: str) -> Dict[str,str]:
        # Expect something like MSFT_2023.pdf or GOOGL-2024.pdf
        base = os.path.splitext(fname)[0]
        m = re.match(r"(MSFT|GOOGL|NVDA)[-_]?(\d{4})", base, re.IGNORECASE)
        if m:
            comp, year = m.group(1).upper(), m.group(2)
            return {"company": comp, "year": year}
        # fallback: attempt to find year
        y = re.search(r"(20\d{2})", base)
        year = y.group(1) if y else "unknown"
        # attempt to map company by keywords
        comp = None
        if "goog" in base.lower() or "alphabet" in base.lower():
            comp = "GOOGL"
        elif "micro" in base.lower() or "msft" in base.lower():
            comp = "MSFT"
        elif "nvidia" in base.lower() or "nvda" in base.lower():
            comp = "NVDA"
        else:
            comp = "UNKNOWN"
        return {"company": comp, "year": year}

    def _build_index(self):
        texts = [t["text"] for t in self.texts]
        # compute embeddings in batches
        print("[index] encoding embeddings...")
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        self.embeddings = embeddings.astype("float32")
        dim = self.embeddings.shape[1]
        print(f"[index] embeddings shape: {self.embeddings.shape}")
        # faiss index
        self.index = faiss.IndexFlatIP(dim)  # use inner-product on normalized vectors
        # normalize vectors
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        print(f"[index] faiss index built with {self.index.ntotal} vectors")

    def retrieve(self, query: str, top_k=TOP_K) -> List[Dict[str,Any]]:
        q_emb = self.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_emb)
        D,I = self.index.search(q_emb, top_k)
        scores = D[0].tolist()
        idxs = I[0].tolist()
        results = []
        for score, idx in zip(scores, idxs):
            if idx < 0:
                continue
            rec = self.texts[idx].copy()
            rec["_score"] = float(score)
            results.append(rec)
        return results

# ---------------------------
# Simple extractor helpers
# ---------------------------
NUM_RE = re.compile(r"(?P<num>\$?[\d{1,3},]+\d(?:\.\d+)?)(?:\s*(million|billion|bn|m|k)?)", re.IGNORECASE)
PERCENT_RE = re.compile(r"(?P<pct>\d{1,3}\.\d+%|\d{1,3}%|\d{1,3}\.\d+ percent|\d{1,3} percent)", re.IGNORECASE)

def find_first_percentage(chunks: List[Dict[str,Any]]) -> Tuple[str, Dict[str,Any]]:
    for c in chunks:
        m = PERCENT_RE.search(c["text"])
        if m:
            return m.group(0), c
    return None, None

def find_first_number(chunks: List[Dict[str,Any]]) -> Tuple[str, Dict[str,Any]]:
    for c in chunks:
        m = NUM_RE.search(c["text"])
        if m:
            return m.group(0) + ((" " + (m.group(2) or "")) if m.group(2) else ""), c
    return None, None

# A light-weight metric extractor using keyword heuristics
def extract_metric(metric: str, company: str, year: str, index: RAGIndex) -> Dict[str,Any]:
    q = f"{company} {metric} {year}"
    chunks = index.retrieve(q, top_k=8)
    # attempt to parse percentages first for margin-like metrics
    if any(w in metric.lower() for w in ["margin", "percentage", "%", "percent", "percentage of revenue", "percentage of"]):
        val, chunk = find_first_percentage(chunks)
        if val:
            return {"value": val, "source": chunk}
    # otherwise try numeric amounts
    val, chunk = find_first_number(chunks)
    if val:
        return {"value": val, "source": chunk}
    # fallback: return top chunk as evidence
    return {"value": None, "source": chunks[0] if chunks else None, "all_chunks": chunks}

# ---------------------------
# Simple agent: detect query type and decompose
# ---------------------------
def needs_decomposition(query: str) -> bool:
    # simple heuristic: comparative words or multiple companies
    comp_words = ["compare", "compare", "which", "highest", "lowest", "vs", "versus", "across", "between"]
    years = re.findall(r"20\d{2}", query)
    return any(w in query.lower() for w in comp_words) or len(years) >= 2

def extract_companies_from_query(query: str) -> List[str]:
    comps = []
    for c in ["MSFT","GOOGL","NVDA","MICROSOFT","GOOGLE","ALPHABET","NVIDIA"]:
        if c.lower() in query.lower():
            # map to canonical
            if "msft" in c.lower() or "micro" in c.lower():
                comps.append("MSFT")
            elif "goog" in c.lower() or "alphabet" in c.lower():
                comps.append("GOOGL")
            else:
                comps.append("NVDA")
    # if none found, assume all three
    return sorted(list(set(comps))) if comps else ["MSFT","GOOGL","NVDA"]

def extract_years_from_query(query: str) -> List[str]:
    ys = re.findall(r"(20\d{2})", query)
    return ys

def decompose_query(query: str) -> List[str]:
    """
    Turn a complex query into sub-queries.
    """
    metric = None
    q_lower = query.lower()
    # detect metric keywords
    if "total revenue" in q_lower or "total revenues" in q_lower or "revenue" in q_lower:
        metric = "total revenue"
    if "operating margin" in q_lower or "operating margin" in q_lower or "operating margin" in q_lower:
        metric = "operating margin"
    if "cloud revenue" in q_lower or "cloud" in q_lower:
        metric = "cloud revenue"
    if "data center" in q_lower or "data-center" in q_lower:
        metric = "data center revenue"
    if "r&d" in q_lower or "research and development" in q_lower or "rd" in q_lower:
        metric = "r&d"
    if "ai" in q_lower or "artificial intelligence" in q_lower:
        metric = "ai investments"

    companies = extract_companies_from_query(query)
    years = extract_years_from_query(query)
    sub_queries = []

    # if it's a growth (from X to Y)
    ys = years
    if "from" in q_lower and "to" in q_lower and len(ys) >= 2:
        # request growth per company
        for c in companies:
            sub_queries.append(f"{c} {metric} {ys[0]}")
            sub_queries.append(f"{c} {metric} {ys[1]}")
        return sub_queries

    # if comparison across companies for a single year
    if len(ys) == 1:
        y = ys[0]
        for c in companies:
            sub_queries.append(f"{c} {metric} {y}")
        return sub_queries

    # if no years mentioned, produce queries for 2023 as default + 2024 where relevant
    for c in companies:
        sub_queries.append(f"{c} {metric} 2023")
    return sub_queries

# ---------------------------
# Orchestrator & synthesizer
# ---------------------------
def answer_query(query: str, index: RAGIndex) -> Dict[str,Any]:
    result = {
        "query": query,
        "answer": None,
        "reasoning": None,
        "sub_queries": [],
        "sources": []
    }
    if needs_decomposition(query):
        sub_qs = decompose_query(query)
    else:
        # single retrieval
        sub_qs = [query]

    result["sub_queries"] = sub_qs

    collected = []
    for sq in sub_qs:
        # parse company/year out of subquery pattern
        m = re.match(r"(MSFT|GOOGL|NVDA)\s+(.+?)\s+(20\d{2})", sq, re.IGNORECASE)
        if m:
            company = m.group(1).upper()
            metric = m.group(2).strip()
            year = m.group(3)
            metric_res = extract_metric(metric, company, year, index)
            collected.append({
                "sub_query": sq,
                "company": company,
                "year": year,
                "metric": metric,
                "value": metric_res.get("value"),
                "source": metric_res.get("source"),
                "raw_chunks": metric_res.get("all_chunks")
            })
        else:
            # fallback: basic retrieval
            chunks = index.retrieve(sq, top_k=TOP_K)
            collected.append({
                "sub_query": sq,
                "company": None,
                "year": None,
                "metric": None,
                "value": None,
                "source": chunks[0] if chunks else None,
                "raw_chunks": chunks
            })

    # Fill sources and synthesize a simple answer
    sources = []
    for c in collected:
        src = c["source"]
        if src:
            sources.append({
                "company": c.get("company"),
                "year": c.get("year"),
                "excerpt": (src["text"][:400] + "...") if "text" in src and src["text"] else None,
                "page": src.get("page"),
                "filename": src.get("filename")
            })
    result["sources"] = sources

    # Build answer text depending on query type (heuristic)
    if len(collected) == 1:
        c = collected[0]
        if c["value"]:
            ans = f"{c.get('company','')} {c.get('metric','').strip()} {c.get('year','')}: {c['value']}"
            result["answer"] = ans
            result["reasoning"] = f"Retrieved top chunks for the query and extracted first matching numeric/percentage value."
        else:
            # return top chunk text as answer
            src = c["source"]
            if src:
                snippet = src["text"][:800] + "..."
                result["answer"] = snippet
                result["reasoning"] = "Returned the top context chunk as the filing did not contain an easily parsed numeric value."
            else:
                result["answer"] = "No relevant information found in the filings."
                result["reasoning"] = "No chunks retrieved."
    else:
        # comparative: try to compare numeric values
        values = []
        for c in collected:
            v = c.get("value")
            # attempt to normalize percent strings to numeric (heuristic)
            if isinstance(v, str):
                p = re.search(r"(\d{1,3}\.\d+|\d{1,3})\s*%", v)
                if p:
                    values.append((c.get("company"), float(p.group(1)), c.get("year")))
                    continue
                # dollars: remove $ and commas
                num = re.search(r"\$?([\d,]+\d(?:\.\d+)?)", v)
                if num:
                    try:
                        valnum = float(num.group(1).replace(",",""))
                        values.append((c.get("company"), valnum, c.get("year")))
                        continue
                    except:
                        pass
            values.append((c.get("company"), None, c.get("year")))

        # if at least one numeric exists, pick highest
        numeric_values = [t for t in values if t[1] is not None]
        if numeric_values:
            # determine type: percent (0-100) or absolute
            is_percent = any(0 <= tv[1] <= 100 for tv in numeric_values)
            if is_percent:
                best = max(numeric_values, key=lambda x: x[1])
                result["answer"] = f"{best[0]} had the highest value: {best[1]}% ({best[2]})."
                result["reasoning"] = f"Compared extracted percentages from filings for each company."
            else:
                best = max(numeric_values, key=lambda x: x[1])
                result["answer"] = f"{best[0]} had the highest value: {best[1]} ({best[2]})."
                result["reasoning"] = f"Compared extracted numeric amounts from filings for each company."
        else:
            # no numeric: synthesize from snippets
            snippets = []
            for c in collected:
                src = c["source"]
                snippet = src["text"][:300] + "..." if src and src.get("text") else ""
                snippets.append(f"{c.get('sub_query')}: {snippet}")
            result["answer"] = "Could not extract exact numeric comparisons. Returned evidence snippets per company."
            result["reasoning"] = "No parsable numeric or percentage values were found; returned context for manual comparison."
            result["evidence_snippets"] = snippets

    result["detailed_results"] = collected
    return result

# ---------------------------
# CLI
# ---------------------------
def interactive_loop(index: RAGIndex):
    print("Simple RAG Agent CLI. Type 'quit' to exit.")
    while True:
        q = input("\nQuery> ").strip()
        if q.lower() in ("quit","exit"):
            break
        resp = answer_query(q, index)
        print(json.dumps(resp, indent=2))

# ---------------------------
# Run
# ---------------------------
def main():
    rug = RAGIndex()
    rug.ingest_folder(DATA_DIR)
    # quick demo queries (uncomment to run programmatically)
    sample_queries = [
        "What was NVIDIA's total revenue in fiscal year 2024?",
        "Which company had the highest operating margin in 2023?",
        "How did NVIDIA's data center revenue grow from 2022 to 2023?",
        "What percentage of Google's 2023 revenue came from advertising?",
        "Compare AI investments mentioned by all three companies in their 2024 10-Ks"
    ]
    print("Index built. Enter interactive mode or run sample queries.")
    print("Type 'cli' to enter interactive shell, 'sample' to run sample queries, or 'quit' to exit.")
    cmd = input("> ").strip().lower()
    if cmd == "cli":
        interactive_loop(rug)
    elif cmd == "sample":
        for q in sample_queries:
            print("\n=== QUERY ===")
            print(q)
            out = answer_query(q, rug)
            print(json.dumps(out, indent=2))
    else:
        print("exiting.")

if __name__ == "__main__":
    main()
