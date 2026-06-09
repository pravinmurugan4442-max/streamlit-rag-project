# # # ==========================================================================================
# # # LOYOLA COLLEGE — WEBSITE RAG CHATBOT  v8.0
# # # ==========================================================================================
# # #
# # # WHAT CHANGED FROM v7.0
# # # ──────────────────────
# # # ✔  Course questions       — facts layer returns actual UG/PG programme names, not dept list
# # # ✔  Response formatting    — prompt instructs LLM to use bullets / tables / concise sentences
# # # ✔  Grounded follow-ups    — suggestions restricted to topics present in retrieved context;
# # #                             empty list returned when answer is the fallback phrase
# # # ✔  Auto eval dataset      — _ensure_eval_dataset() silently creates 25-question default;
# # #                             no warning ever shown to end users
# # #
# # # CARRIED FORWARD FROM v7.0 (all improvements preserved)
# # # ───────────────────────────────────────────────────────
# # # ✔  Noise chunk filter        — removes URL/email/nav/footer boilerplate at index time
# # # ✔  Source priority boost     — official PDFs, placement & NAAC docs ranked higher
# # # ✔  Statistics query class    — triggers deep retrieval + stat-aware prompt hint
# # # ✔  Tuned retrieval params    — larger K, bigger chunks, lower rerank threshold
# # # ✔  Debug mode                — sidebar toggle reveals chunks and rerank scores
# # # ✔  Absolute prompt rules     — no hallucination, exact fallback phrase
# # #
# # # UNCHANGED FROM v6.0
# # # ────────────────────
# # # ✘  FAISS index / build logic
# # # ✘  BM25 retriever
# # # ✘  RRF merge math
# # # ✘  Semantic cache
# # # ✘  Hallucination grounding
# # # ✘  Query rewriting
# # # ✘  All analytics / feedback / evaluation tabs
# # #
# # # ⚠  REBUILD REQUIRED when upgrading from v6 (chunk size changed).
# # #    Delete faiss_index/ or click "Rebuild Knowledge Base" in the sidebar.
# # #
# # # RUN
# # # ───
# # #   streamlit run rag_v8.py
# # #
# # # ==========================================================================================


# # # ==========================================================================================
# # # IMPORTS
# # # ==========================================================================================

# # import csv
# # import io
# # import os
# # import re
# # import json
# # import math
# # import pickle
# # import sqlite3
# # import hashlib
# # import warnings
# # import time
# # import datetime
# # import unicodedata

# # from pathlib     import Path
# # from dataclasses import dataclass
# # from typing      import Optional

# # import numpy as np
# # import pandas as pd
# # import streamlit as st
# # import tiktoken

# # from dotenv                           import load_dotenv
# # from sentence_transformers            import CrossEncoder, SentenceTransformer

# # from langchain_core.documents         import Document
# # from langchain_community.vectorstores import FAISS
# # from langchain_community.retrievers   import BM25Retriever

# # from langchain_community.document_loaders import (
# #     PyMuPDFLoader,
# #     PDFPlumberLoader,
# # )

# # from langchain_text_splitters       import RecursiveCharacterTextSplitter
# # from langchain_community.embeddings import HuggingFaceEmbeddings
# # from langchain_groq                 import ChatGroq

# # warnings.filterwarnings("ignore")
# # load_dotenv()


# # # ==========================================================================================
# # # PAGE CONFIG
# # # ==========================================================================================

# # st.set_page_config(
# #     page_title = "Loyola College Assistant",
# #     page_icon  = "🎓",
# #     layout     = "wide",
# # )


# # # ==========================================================================================
# # # CONFIGURATION
# # # ==========================================================================================

# # @dataclass
# # class Config:

# #     # ── Data folders ──────────────────────────────────────────────────
# #     RAG_JSON_FOLDER    : str = "./data/rag_export"
# #     CLEANED_TEXT_FOLDER: str = "./data/cleaned_text"
# #     OCR_TEXT_FOLDER    : str = "./data/ocr_text"
# #     PDF_FOLDER         : str = "./data/pdfs"
# #     FACTS_FILE         : str = "./data/facts.json"
# #     EVAL_DATASET_FILE  : str = "./data/eval_dataset.json"

# #     # ── Persistence ───────────────────────────────────────────────────
# #     FAISS_PATH         : str = "./faiss_index"
# #     CHUNKS_PKL         : str = "./faiss_index/chunks.pkl"
# #     BM25_PKL           : str = "./faiss_index/bm25.pkl"
# #     HASH_FILE          : str = "./faiss_index/data_hash.txt"
# #     QUERY_CACHE_FILE   : str = "./faiss_index/query_cache.pkl"
# #     ANALYTICS_DB       : str = "./faiss_index/analytics.db"

# #     # ── Models ────────────────────────────────────────────────────────
# #     EMBEDDING_MODEL    : str = "BAAI/bge-small-en-v1.5"
# #     RERANK_MODEL       : str = "BAAI/bge-reranker-base"
# #     LLM_MODEL          : str = "llama-3.1-8b-instant"

# #     # ── Chunking ──────────────────────────────────────────────────────
# #     CHUNK_SIZE         : int = 1200     # keeps stat tables / paragraphs whole
# #     CHUNK_OVERLAP      : int = 200      # prevents mid-sentence / mid-table breaks

# #     # ── Retrieval ─────────────────────────────────────────────────────
# #     RETRIEVAL_K_DEFAULT : int   = 16    # doubled from v6 — bigger candidate pool
# #     RETRIEVAL_K_DEEP    : int   = 24
# #     RETRIEVAL_K_FAST    : int   = 10
# #     RERANK_K            : int   = 6
# #     RERANK_THRESHOLD    : float = 0.15  # lowered — 0.35 discarded valid chunks
# #     MAX_CONTEXT_TOKENS  : int   = 3500  # raised — full stat tables fit in prompt
# #     RRF_K               : int   = 60

# #     # ── Semantic cache ────────────────────────────────────────────────
# #     SEMANTIC_CACHE_THRESHOLD : float = 0.92

# #     # ── Hallucination grounding ───────────────────────────────────────
# #     GROUNDING_THRESHOLD : float = 0.25
# #     CONFIDENCE_HIGH     : float = 0.55
# #     CONFIDENCE_MED      : float = 0.35

# #     # ── Chat ──────────────────────────────────────────────────────────
# #     HISTORY_TURNS       : int   = 4


# # CFG = Config()
# # Path(CFG.FAISS_PATH).mkdir(parents=True, exist_ok=True)


# # # ==========================================================================================
# # # DEVICE
# # # ==========================================================================================

# # try:
# #     import torch
# #     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# # except ImportError:
# #     DEVICE = "cpu"


# # # ==========================================================================================
# # # TOKEN COUNTER
# # # ==========================================================================================

# # _tokenizer = tiktoken.get_encoding("cl100k_base")

# # def count_tokens(text: str) -> int:
# #     return len(_tokenizer.encode(text))


# # # ==========================================================================================
# # # QUERY CACHE
# # # ==========================================================================================

# # def _load_cache() -> dict:
# #     try:
# #         if os.path.exists(CFG.QUERY_CACHE_FILE):
# #             return pickle.load(open(CFG.QUERY_CACHE_FILE, "rb"))
# #     except Exception:
# #         pass
# #     return {}


# # def _save_cache(cache: dict) -> None:
# #     try:
# #         pickle.dump(cache, open(CFG.QUERY_CACHE_FILE, "wb"))
# #     except Exception:
# #         pass


# # # ==========================================================================================
# # # DATABASE INIT
# # # ==========================================================================================

# # def _init_db() -> sqlite3.Connection:
# #     """Create / migrate analytics.db with all required tables."""
# #     conn = sqlite3.connect(CFG.ANALYTICS_DB)

# #     conn.execute("""
# #         CREATE TABLE IF NOT EXISTS query_log (
# #             id         INTEGER PRIMARY KEY AUTOINCREMENT,
# #             ts         TEXT,
# #             query      TEXT,
# #             q_class    TEXT,
# #             latency_s  REAL,
# #             cache_hit  INTEGER,
# #             confidence TEXT,
# #             answered   INTEGER,
# #             section    TEXT
# #         )
# #     """)

# #     try:
# #         conn.execute("ALTER TABLE query_log ADD COLUMN section TEXT")
# #     except sqlite3.OperationalError:
# #         pass

# #     conn.execute("""
# #         CREATE TABLE IF NOT EXISTS feedback (
# #             id         INTEGER PRIMARY KEY AUTOINCREMENT,
# #             ts         TEXT,
# #             log_id     INTEGER,
# #             query      TEXT,
# #             helpful    INTEGER
# #         )
# #     """)

# #     conn.execute("""
# #         CREATE TABLE IF NOT EXISTS eval_results (
# #             id           INTEGER PRIMARY KEY AUTOINCREMENT,
# #             ts           TEXT,
# #             run_label    TEXT,
# #             k            INTEGER,
# #             precision_k  REAL,
# #             recall_k     REAL,
# #             mrr          REAL,
# #             ndcg         REAL,
# #             num_queries  INTEGER
# #         )
# #     """)

# #     conn.commit()
# #     return conn


# # # ==========================================================================================
# # # QUERY ANALYTICS
# # # ==========================================================================================

# # def log_query(
# #     query     : str,
# #     q_class   : str,
# #     latency_s : float,
# #     cache_hit : bool,
# #     confidence: str,
# #     answered  : bool,
# #     section   : str = "",
# # ) -> Optional[int]:
# #     """Log a query to analytics.db. Returns the inserted row id."""
# #     try:
# #         conn = _init_db()
# #         cur  = conn.execute(
# #             """INSERT INTO query_log
# #                (ts, query, q_class, latency_s, cache_hit, confidence, answered, section)
# #                VALUES (?,?,?,?,?,?,?,?)""",
# #             (
# #                 datetime.datetime.now().isoformat(),
# #                 query, q_class, round(latency_s, 3),
# #                 int(cache_hit), confidence, int(answered), section,
# #             ),
# #         )
# #         conn.commit()
# #         row_id = cur.lastrowid
# #         conn.close()
# #         return row_id
# #     except Exception:
# #         return None


# # def log_feedback(log_id: Optional[int], query: str, helpful: bool) -> None:
# #     """Record a 👍 or 👎 for a specific query."""
# #     try:
# #         conn = _init_db()
# #         conn.execute(
# #             "INSERT INTO feedback (ts, log_id, query, helpful) VALUES (?,?,?,?)",
# #             (datetime.datetime.now().isoformat(), log_id, query, int(helpful)),
# #         )
# #         conn.commit()
# #         conn.close()
# #     except Exception:
# #         pass


# # # ==========================================================================================
# # # SECURITY FILTER
# # # ==========================================================================================

# # _BLOCKED = [
# #     r"ignore (previous|all) instructions",
# #     r"developer mode",
# #     r"jailbreak",
# #     r"system prompt",
# #     r"forget (your|all) instructions",
# #     r"act as (an? )?(unrestricted|evil|unethical)",
# #     r"pretend you (are|have no)",
# #     r"bypass (your )?(safety|filter|restriction)",
# #     r"disregard (your )?guidelines",
# #     r"you are now",
# #     r"new persona",
# # ]

# # def is_malicious(q: str) -> bool:
# #     q_lower = q.lower()
# #     if any(re.search(p, q_lower) for p in _BLOCKED):
# #         return True
# #     q_norm = unicodedata.normalize("NFKD", q_lower)
# #     return any(re.search(p, q_norm) for p in _BLOCKED)


# # # ==========================================================================================
# # # TEXT CLEANER
# # # ==========================================================================================

# # def clean_text(t: str) -> str:
# #     t = re.sub(r"\s+", " ", t)
# #     t = re.sub(r"[^\x20-\x7E\u0900-\u097F\u0B80-\u0BFF\n]", "", t)
# #     return t.strip()


# # # ==========================================================================================
# # # NOISE CHUNK FILTER
# # # ==========================================================================================
# # # Removes URL/email/nav/footer boilerplate at index-build time so these chunks
# # # never reach FAISS or BM25. Stat-rich chunks are always preserved even if they
# # # incidentally contain a URL.

# # _NOISE_PATTERNS = [
# #     re.compile(r'(https?://\S+\s*){3,}'),
# #     re.compile(r'([a-z0-9_.+-]+@[a-z0-9-]+\.[a-z]+\s*){3,}', re.I),
# #     re.compile(r'(©|\bAll rights reserved\b|\bPrivacy Policy\b)', re.I),
# #     re.compile(r'\b(Home\s*[|>]\s*About|Skip to content|Cookie Policy)\b', re.I),
# #     re.compile(r'\b(Follow us on|Share this page|Subscribe to our)\b', re.I),
# # ]

# # _STAT_MARKERS = re.compile(
# #     r'\b(\d[\d,]+\s*(students?|faculty|staff|crore|lakh|%|placed|recruited|alumni|departments?))\b',
# #     re.I,
# # )

# # def is_noise_chunk(text: str) -> bool:
# #     """Return True if chunk is predominantly navigation / contact / footer boilerplate."""
# #     if len(text.strip()) < 80:
# #         return True
# #     if _STAT_MARKERS.search(text):
# #         return False            # never discard stat-rich chunks
# #     words       = text.split()
# #     word_count  = max(len(words), 1)
# #     url_count   = len(re.findall(r'https?://\S+', text))
# #     email_count = len(re.findall(r'[a-z0-9_.+-]+@[a-z0-9-]+\.[a-z]{2,}', text, re.I))
# #     if (url_count + email_count) / word_count > 0.30:
# #         return True
# #     return any(p.search(text) for p in _NOISE_PATTERNS)


# # # ==========================================================================================
# # # DATA HASH
# # # ==========================================================================================

# # def compute_data_hash() -> str:
# #     h = hashlib.sha256()
# #     for folder in [
# #         CFG.RAG_JSON_FOLDER, CFG.CLEANED_TEXT_FOLDER,
# #         CFG.OCR_TEXT_FOLDER, CFG.PDF_FOLDER,
# #     ]:
# #         p = Path(folder)
# #         if not p.exists():
# #             continue
# #         for f in sorted(p.iterdir()):
# #             h.update(f.name.encode())
# #             h.update(str(f.stat().st_size).encode())
# #     return h.hexdigest()


# # def data_changed() -> bool:
# #     if not os.path.exists(CFG.HASH_FILE):
# #         return True
# #     return Path(CFG.HASH_FILE).read_text().strip() != compute_data_hash()


# # def save_data_hash() -> None:
# #     Path(CFG.HASH_FILE).write_text(compute_data_hash())


# # def can_fast_load() -> bool:
# #     return (
# #         os.path.exists(CFG.FAISS_PATH)
# #         and os.path.exists(CFG.CHUNKS_PKL)
# #         and os.path.exists(CFG.BM25_PKL)
# #         and not data_changed()
# #     )


# # # ==========================================================================================
# # # SECTION EXTRACTOR
# # # ==========================================================================================

# # def extract_section_from_source(source: str) -> str:
# #     if not source:
# #         return ""
# #     s = source.lower()
# #     if s.endswith(".pdf") or "/pdfs/" in s:
# #         return "pdf"
# #     if "/ocr_text/" in s or "/ocr/" in s:
# #         return "ocr"
# #     if "/cleaned_text/" in s or "/text/" in s:
# #         return "cleaned_text"
# #     try:
# #         from urllib.parse import urlparse
# #         parsed = urlparse(source)
# #         parts  = [p for p in parsed.path.split("/") if p]
# #         if parts:
# #             return parts[0].lower()
# #     except Exception:
# #         pass
# #     return ""


# # # ==========================================================================================
# # # STRUCTURED FACTS LAYER
# # # ==========================================================================================
# # # v8 adds "programmes" key with actual UG/PG degree names so course questions
# # # return real programme titles instead of a flat department list.

# # _DEFAULT_FACTS = {
# #     "fees": {
# #         "description": "Fee information must be confirmed with the college office as it changes yearly.",
# #         "contact": "accounts@loyolacollege.edu or +91-44-2817-2624",
# #     },
# #     "contact": {
# #         "address": "Loyola College (Autonomous), Nungambakkam, Chennai - 600 034, Tamil Nadu, India",
# #         "phone": "+91-44-2817-2624 / 2817-3274",
# #         "email": "principal@loyolacollege.edu",
# #         "website": "www.loyolacollege.edu",
# #     },
# #     "established": "1925",
# #     "affiliation": "University of Madras (Autonomous status granted 1978)",
# #     "accreditation": "NAAC A++ (reaccredited 2022)",
# #     "departments": [
# #         "Commerce", "Economics", "English", "Tamil", "Hindi", "French",
# #         "Physics", "Chemistry", "Mathematics", "Statistics", "Computer Science",
# #         "Biochemistry", "Microbiology", "Plant Biology and Biotechnology",
# #         "Advanced Zoology and Biotechnology", "Visual Communication",
# #         "Social Work", "History", "Philosophy", "Psychology",
# #         "Business Administration (BBA)", "Computer Applications (BCA)",
# #     ],
# #     # NEW v8 — actual degree programme names for course questions
# #     "programmes": {
# #         "UG Programmes": [
# #             "B.A. English Literature", "B.A. Tamil", "B.A. Hindi", "B.A. French",
# #             "B.A. History", "B.A. Philosophy", "B.A. Economics",
# #             "B.Com. (General)", "B.Com. (Professional Accounting)",
# #             "B.Sc. Physics", "B.Sc. Chemistry", "B.Sc. Mathematics",
# #             "B.Sc. Statistics", "B.Sc. Computer Science", "B.Sc. Biochemistry",
# #             "B.Sc. Microbiology", "B.Sc. Plant Biology & Biotechnology",
# #             "B.Sc. Advanced Zoology & Biotechnology",
# #             "B.Sc. Visual Communication",
# #             "B.B.A. Business Administration", "B.C.A. Computer Applications",
# #             "B.S.W. Social Work",
# #         ],
# #         "PG Programmes": [
# #             "M.A. English", "M.A. Tamil", "M.A. History", "M.A. Philosophy",
# #             "M.A. Economics", "M.Com.", "M.Sc. Physics", "M.Sc. Chemistry",
# #             "M.Sc. Mathematics", "M.Sc. Statistics", "M.Sc. Computer Science",
# #             "M.Sc. Biochemistry", "M.Sc. Microbiology",
# #             "M.Sc. Plant Biology & Biotechnology",
# #             "M.Sc. Advanced Zoology & Biotechnology",
# #             "M.S.W. Social Work", "M.C.A.", "M.B.A.",
# #         ],
# #         "M.Phil. / Ph.D.": [
# #             "Available in most UG/PG departments. Contact the Research Cell for details.",
# #         ],
# #     },
# # }


# # @st.cache_resource(show_spinner=False)
# # def load_facts() -> dict:
# #     p = Path(CFG.FACTS_FILE)
# #     if p.exists():
# #         try:
# #             data   = json.loads(p.read_text(encoding="utf-8"))
# #             merged = {**_DEFAULT_FACTS, **data}
# #             return merged
# #         except Exception:
# #             pass
# #     return _DEFAULT_FACTS


# # def query_facts_layer(question: str, facts: dict) -> Optional[str]:
# #     q = question.lower()

# #     # Contact / location
# #     if any(k in q for k in ["address", "location", "where is loyola", "phone", "email", "website", "contact"]):
# #         c = facts.get("contact", {})
# #         return (
# #             f"**Loyola College Contact Details:**\n"
# #             f"- Address: {c.get('address', 'N/A')}\n"
# #             f"- Phone: {c.get('phone', 'N/A')}\n"
# #             f"- Email: {c.get('email', 'N/A')}\n"
# #             f"- Website: {c.get('website', 'N/A')}"
# #         )

# #     # Accreditation / ranking
# #     if any(k in q for k in ["naac", "accreditat", "ranking", "grade", "autonomous"]):
# #         return (
# #             f"Loyola College holds **{facts.get('accreditation', 'N/A')}** accreditation. "
# #             f"It received autonomous status in {facts.get('affiliation', 'N/A')}."
# #         )

# #     # Establishment
# #     if any(k in q for k in ["established", "founded", "year of", "when was loyola"]):
# #         return f"Loyola College was established in **{facts.get('established', 'N/A')}**."

# #     # Courses / programmes — NEW v8: returns actual degree programme names
# #     if any(k in q for k in [
# #         "course", "programme", "program", "ug ", "pg ", "degree",
# #         "bsc", "bcom", "ba ", "msc", "mcom", "ma ", "mca", "mba", "phd",
# #         "undergraduate", "postgraduate", "what can i study", "what do you offer",
# #     ]):
# #         programmes = facts.get("programmes", {})
# #         if programmes:
# #             lines = ["**Programmes offered at Loyola College:**\n"]
# #             for level, prog_list in programmes.items():
# #                 lines.append(f"\n**{level}**")
# #                 for prog in prog_list:
# #                     lines.append(f"- {prog}")
# #             return "\n".join(lines)
# #         # Fallback to department list if programmes key missing
# #         depts = facts.get("departments", [])
# #         if depts:
# #             return (
# #                 "**Departments at Loyola College:**\n"
# #                 + "\n".join(f"- {d}" for d in depts)
# #             )

# #     # Departments (explicit department question without programme intent)
# #     if any(k in q for k in ["department", "stream", "what subjects", "which departments"]):
# #         depts = facts.get("departments", [])
# #         if depts:
# #             return (
# #                 "**Departments at Loyola College:**\n"
# #                 + "\n".join(f"- {d}" for d in depts)
# #             )

# #     # Fees
# #     if any(k in q for k in ["fee", "fees", "tuition", "cost", "how much"]):
# #         c = facts.get("fees", {})
# #         return (
# #             f"Fee structures at Loyola College change annually. "
# #             f"{c.get('description', '')} "
# #             f"Contact: {c.get('contact', 'the college accounts office')}."
# #         )

# #     return None


# # # ==========================================================================================
# # # DOCUMENT LOADERS
# # # ==========================================================================================

# # def load_rag_json_folder(log) -> list[Document]:
# #     folder = Path(CFG.RAG_JSON_FOLDER)
# #     docs   = []
# #     if not folder.exists():
# #         return docs
# #     files = sorted(folder.glob("*.json"))
# #     log.write(f"📄 RAG JSON: {len(files)} files")
# #     for fpath in files:
# #         try:
# #             raw  = json.loads(fpath.read_text(encoding="utf-8"))
# #             text = raw.get("page_content", "").strip()
# #             meta = raw.get("metadata", {})
# #             if len(text) < 100:
# #                 continue
# #             source  = meta.get("url") or meta.get("source") or str(fpath)
# #             section = meta.get("section") or extract_section_from_source(source)
# #             docs.append(Document(
# #                 page_content = clean_text(text),
# #                 metadata = {
# #                     "source" : source,
# #                     "title"  : meta.get("title", ""),
# #                     "type"   : "website",
# #                     "section": section,
# #                 },
# #             ))
# #         except Exception as e:
# #             print(f"[JSON ERROR] {fpath.name}: {e}")
# #     log.success(f"✅ RAG JSON: {len(docs)} pages loaded")
# #     return docs


# # def load_text_folder(folder_path: str, label: str, log) -> list[Document]:
# #     folder = Path(folder_path)
# #     docs   = []
# #     if not folder.exists():
# #         return docs
# #     files = sorted(folder.glob("*.txt"))
# #     log.write(f"📝 {label}: {len(files)} files")
# #     for fpath in files:
# #         try:
# #             text = fpath.read_text(encoding="utf-8", errors="replace").strip()
# #             if len(text) < 100:
# #                 continue
# #             source  = str(fpath)
# #             section = extract_section_from_source(source)
# #             docs.append(Document(
# #                 page_content = clean_text(text),
# #                 metadata = {
# #                     "source" : source,
# #                     "title"  : fpath.stem,
# #                     "type"   : label.lower(),
# #                     "section": section,
# #                 },
# #             ))
# #         except Exception as e:
# #             print(f"[TEXT ERROR] {fpath.name}: {e}")
# #     log.success(f"✅ {label}: {len(docs)} loaded")
# #     return docs


# # def load_pdf(path: str) -> list[Document]:
# #     for Loader in [PyMuPDFLoader, PDFPlumberLoader]:
# #         try:
# #             raw  = Loader(path).load()
# #             docs = [
# #                 Document(
# #                     page_content = clean_text(d.page_content),
# #                     metadata = {
# #                         "source" : path,
# #                         "page"   : d.metadata.get("page", "N/A"),
# #                         "type"   : "pdf",
# #                         "title"  : Path(path).name,
# #                         "section": "pdf",
# #                     },
# #                 )
# #                 for d in raw if len(d.page_content.strip()) > 50
# #             ]
# #             if docs:
# #                 return docs
# #         except Exception:
# #             continue
# #     return []


# # def load_pdf_folder(log) -> list[Document]:
# #     folder = Path(CFG.PDF_FOLDER)
# #     docs   = []
# #     if not folder.exists():
# #         return docs
# #     files = sorted(folder.glob("*.pdf"))
# #     log.write(f"📚 PDFs: {len(files)} files")
# #     bar = log.progress(0)
# #     for i, fpath in enumerate(files):
# #         try:
# #             docs.extend(load_pdf(str(fpath)))
# #             bar.progress((i + 1) / max(len(files), 1))
# #         except Exception as e:
# #             print(f"[PDF ERROR] {fpath.name}: {e}")
# #     log.success(f"✅ PDFs: {len(docs)} pages loaded")
# #     return docs


# # def load_all_documents(sidebar_log) -> list[Document]:
# #     all_docs  = []
# #     seen_keys = set()

# #     def _add(docs):
# #         for d in docs:
# #             key = d.page_content[:300]
# #             if key not in seen_keys:
# #                 seen_keys.add(key)
# #                 all_docs.append(d)

# #     s1 = sidebar_log.empty()
# #     s2 = sidebar_log.empty()
# #     s3 = sidebar_log.empty()
# #     s4 = sidebar_log.empty()

# #     _add(load_rag_json_folder(s1))
# #     _add(load_text_folder(CFG.CLEANED_TEXT_FOLDER, "Cleaned Text", s2))
# #     _add(load_text_folder(CFG.OCR_TEXT_FOLDER,     "OCR Text",     s3))
# #     _add(load_pdf_folder(s4))

# #     sidebar_log.success(f"🗂 Total unique docs: {len(all_docs)}")
# #     return all_docs


# # # ==========================================================================================
# # # CHUNKING
# # # ==========================================================================================
# # # \n\n\n separator respects section/table boundaries.
# # # is_noise_chunk() removes boilerplate at build time.

# # def create_chunks(docs: list[Document]) -> list[Document]:
# #     splitter = RecursiveCharacterTextSplitter(
# #         chunk_size    = CFG.CHUNK_SIZE,
# #         chunk_overlap = CFG.CHUNK_OVERLAP,
# #         separators    = ["\n\n\n", "\n\n", "\n", ". ", " ", ""],
# #     )
# #     chunks = splitter.split_documents(docs)
# #     return [
# #         c for c in chunks
# #         if len(c.page_content.strip()) > 80 and not is_noise_chunk(c.page_content)
# #     ]


# # # ==========================================================================================
# # # MODELS
# # # ==========================================================================================

# # @st.cache_resource(show_spinner=False)
# # def load_embedding_model():
# #     return HuggingFaceEmbeddings(
# #         model_name    = CFG.EMBEDDING_MODEL,
# #         model_kwargs  = {"device": DEVICE},
# #         encode_kwargs = {"normalize_embeddings": True},
# #     )


# # @st.cache_resource(show_spinner=False)
# # def load_reranker():
# #     return CrossEncoder(CFG.RERANK_MODEL, device=DEVICE)


# # @st.cache_resource(show_spinner=False)
# # def load_sentence_model():
# #     return SentenceTransformer(CFG.EMBEDDING_MODEL, device=DEVICE)


# # # ==========================================================================================
# # # KNOWLEDGE BASE
# # # ==========================================================================================

# # def build_knowledge_base(sidebar_log, emb):

# #     if can_fast_load():
# #         sidebar_log.info("⚡ Fast loading from saved index...")
# #         t0     = time.time()
# #         chunks = pickle.load(open(CFG.CHUNKS_PKL, "rb"))
# #         bm25   = pickle.load(open(CFG.BM25_PKL,   "rb"))
# #         db     = FAISS.load_local(
# #             CFG.FAISS_PATH, emb,
# #             allow_dangerous_deserialization=True,
# #         )
# #         elapsed = round(time.time() - t0, 1)
# #         sidebar_log.success(f"✅ {len(chunks):,} chunks loaded in {elapsed}s")
# #         return db, chunks, bm25

# #     sidebar_log.warning("🔄 Building knowledge base — please wait...")
# #     t0   = time.time()
# #     docs = load_all_documents(sidebar_log)

# #     if not docs:
# #         st.error("❌ No documents found. Check your data/ folder.")
# #         st.stop()

# #     sidebar_log.write(f"✂️ Chunking {len(docs):,} documents...")
# #     chunks = create_chunks(docs)
# #     sidebar_log.write(f"📦 {len(chunks):,} chunks created")
# #     sidebar_log.write("🧠 Building FAISS + BM25 indices...")

# #     for f in ["index.faiss", "index.pkl"]:
# #         fp = Path(CFG.FAISS_PATH) / f
# #         if fp.exists():
# #             fp.unlink()

# #     db   = FAISS.from_documents(chunks, emb)
# #     db.save_local(CFG.FAISS_PATH)

# #     bm25   = BM25Retriever.from_documents(chunks)
# #     bm25.k = CFG.RETRIEVAL_K_DEEP

# #     pickle.dump(chunks, open(CFG.CHUNKS_PKL, "wb"))
# #     pickle.dump(bm25,   open(CFG.BM25_PKL,   "wb"))
# #     save_data_hash()

# #     elapsed = round(time.time() - t0, 1)
# #     sidebar_log.success(f"✅ Built {len(chunks):,} chunks in {elapsed}s")
# #     sidebar_log.info("⚡ Next run will load in ~5 seconds")
# #     return db, chunks, bm25


# # # ==========================================================================================
# # # QUERY CLASSIFICATION
# # # ==========================================================================================

# # _FEE_RE         = re.compile(r"\bfee|tuition|cost|how much|payment\b", re.I)
# # _ELIGIBILITY_RE = re.compile(r"\beligib|qualify|cutoff|criteria|requirement|admission\b", re.I)
# # _DATE_RE        = re.compile(r"\bdate|deadline|when|schedule|calendar|exam date\b", re.I)
# # _FACT_RE        = re.compile(r"\bwho is|what is|where is|located|principal|founded|established\b", re.I)
# # _FOLLOWUP_RE    = re.compile(r"^(what about|how about|and|also|tell me more|explain|elaborate|go on|continue)\b", re.I)

# # _STATS_RE = re.compile(
# #     r"\b("
# #     r"how many students?|total students?|student strength|student count|student enrolment|"
# #     r"placement stat|placement record|placed students?|salary package|average package|"
# #     r"pass percentage|pass rate|result percentage|"
# #     r"faculty count|number of (faculty|staff|teachers?)|"
# #     r"college ranking|nirf rank|number of departments?|programmes? offered|"
# #     r"total intake|sanctioned strength|hostel capacity"
# #     r")\b",
# #     re.I,
# # )


# # def classify_query(query: str, has_history: bool) -> str:
# #     q = query.strip()
# #     if has_history and _FOLLOWUP_RE.search(q):
# #         return "followup"
# #     if has_history and len(q.split()) <= 5:
# #         return "followup"
# #     if _STATS_RE.search(q):
# #         return "statistics"
# #     if _FEE_RE.search(q):
# #         return "fee"
# #     if _ELIGIBILITY_RE.search(q):
# #         return "eligibility"
# #     if _DATE_RE.search(q):
# #         return "date"
# #     if _FACT_RE.search(q):
# #         return "fact"
# #     return "general"


# # def get_retrieval_k(q_class: str) -> int:
# #     if q_class in ("statistics", "eligibility", "date"):
# #         return CFG.RETRIEVAL_K_DEEP
# #     if q_class in ("fact", "fee"):
# #         return CFG.RETRIEVAL_K_FAST
# #     return CFG.RETRIEVAL_K_DEFAULT


# # # ==========================================================================================
# # # HISTORY-AWARE QUERY REWRITING
# # # ==========================================================================================

# # def rewrite_query(query: str, history: list[dict], llm) -> str:
# #     if not history:
# #         return query
# #     recent = history[-2:]
# #     history_text = ""
# #     for h in recent:
# #         history_text += f"Student: {h['q']}\nAssistant: {h['a'][:200]}...\n\n"
# #     prompt = f"""You are a query rewriter for a college information chatbot.
# # Rewrite the follow-up question as a complete, standalone question.

# # Rules:
# # - Output ONLY the rewritten question. Nothing else.
# # - Keep it under 25 words.
# # - If already standalone, return unchanged.

# # Conversation history:
# # {history_text}
# # Follow-up question: {query}

# # Standalone question:"""
# #     try:
# #         result = ""
# #         for chunk in llm.stream(prompt):
# #             result += chunk.content
# #         rewritten = result.strip().strip('"').strip("'")
# #         if 5 < len(rewritten) < 200:
# #             return rewritten
# #     except Exception:
# #         pass
# #     return query


# # # ==========================================================================================
# # # RRF HYBRID RETRIEVER
# # # ==========================================================================================

# # class HybridRRFRetriever:
# #     """FAISS (dense) + BM25 (sparse) fused with Reciprocal Rank Fusion."""

# #     def __init__(self, db, bm25: BM25Retriever):
# #         self._db   = db
# #         self._bm25 = bm25

# #     def _dense_retrieve(self, query: str, k: int) -> list[Document]:
# #         retriever = self._db.as_retriever(
# #             search_type   = "mmr",
# #             search_kwargs = {"k": k, "fetch_k": k * 2},
# #         )
# #         return retriever.invoke(query)

# #     def _sparse_retrieve(self, query: str, k: int) -> list[Document]:
# #         self._bm25.k = k
# #         return self._bm25.invoke(query)

# #     def invoke(
# #         self,
# #         query      : str,
# #         k          : int       = CFG.RETRIEVAL_K_DEFAULT,
# #         type_filter: list[str] = None,
# #     ) -> list[Document]:
# #         dense_docs  = self._dense_retrieve(query, k)
# #         sparse_docs = self._sparse_retrieve(query, k)

# #         doc_map    : dict[str, Document] = {}
# #         rrf_scores : dict[str, float]   = {}

# #         def _key(d: Document) -> str:
# #             return d.page_content[:200]

# #         for rank, doc in enumerate(dense_docs):
# #             key = _key(doc)
# #             doc_map[key]    = doc
# #             rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (CFG.RRF_K + rank + 1)

# #         for rank, doc in enumerate(sparse_docs):
# #             key = _key(doc)
# #             doc_map[key]    = doc
# #             rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (CFG.RRF_K + rank + 1)

# #         sorted_keys = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
# #         results     = [doc_map[kk] for kk in sorted_keys]

# #         if type_filter:
# #             results = [
# #                 d for d in results
# #                 if d.metadata.get("type", "").lower() in [t.lower() for t in type_filter]
# #             ]

# #         return results[:k]


# # # ==========================================================================================
# # # SOURCE PRIORITY BOOST
# # # ==========================================================================================
# # # Adds a small score bonus so official docs (placement reports, NAAC, PDFs)
# # # consistently win over generic web pages when CrossEncoder scores are close.

# # _PRIORITY_PATTERNS = [
# #     re.compile(r'placement|annual.?report|brochure|prospectus', re.I),
# #     re.compile(r'naac|nirf|iqac|accreditat|handbook|admission.?guide', re.I),
# #     re.compile(r'statistical|census|highlights|achievements|ranking', re.I),
# # ]


# # def _source_priority_score(doc: Document) -> float:
# #     """
# #     Additive boost:
# #       0.30 — official doc (placement, NAAC, brochure, NIRF …)
# #       0.15 — any PDF
# #       0.00 — generic web page
# #     """
# #     combined = (
# #         doc.metadata.get("source", "") + " " +
# #         doc.metadata.get("title",  "")
# #     ).lower()
# #     for p in _PRIORITY_PATTERNS:
# #         if p.search(combined):
# #             return 0.30
# #     if doc.metadata.get("type") == "pdf":
# #         return 0.15
# #     return 0.0


# # # ==========================================================================================
# # # RERANKER
# # # ==========================================================================================

# # def rerank(
# #     query: str,
# #     docs : list[Document],
# #     model,
# #     debug: bool = False,
# # ) -> list[tuple[float, Document]]:
# #     """
# #     Score = CrossEncoder(query, chunk) + source_priority_boost.
# #     Returns (boosted_score, Document) sorted descending, capped at RERANK_K.
# #     """
# #     if not docs:
# #         return []

# #     candidates = docs[:30]
# #     pairs      = [[query, d.page_content] for d in candidates]
# #     raw_scores = model.predict(pairs, batch_size=16)

# #     boosted = [
# #         float(s) + _source_priority_score(d)
# #         for s, d in zip(raw_scores, candidates)
# #     ]

# #     ranked   = sorted(zip(boosted, candidates), key=lambda x: x[0], reverse=True)
# #     filtered = [(s, d) for s, d in ranked if s > CFG.RERANK_THRESHOLD]

# #     if not filtered:
# #         filtered = ranked[:2]       # always return at least 2

# #     if debug:
# #         print("\n[RERANK DEBUG] Top-10:")
# #         for i, (s, d) in enumerate(ranked[:10]):
# #             src     = d.metadata.get("source", "?")[:70]
# #             snippet = d.page_content[:100].strip().replace("\n", " ")
# #             print(f"  [{i+1}] score={s:.4f}  src={src}")
# #             print(f"       {snippet!r}")

# #     return filtered[:CFG.RERANK_K]


# # # ==========================================================================================
# # # SEMANTIC CACHE
# # # ==========================================================================================

# # def semantic_cache_lookup(
# #     query    : str,
# #     cache    : dict,
# #     sent_model,
# #     threshold: float = CFG.SEMANTIC_CACHE_THRESHOLD,
# # ) -> Optional[dict]:
# #     if not cache:
# #         return None
# #     try:
# #         q_emb       = sent_model.encode(query, normalize_embeddings=True)
# #         cached_keys = list(cache.keys())
# #         key_embs    = sent_model.encode(cached_keys, normalize_embeddings=True, batch_size=64)
# #         sims        = key_embs @ q_emb
# #         best_idx    = int(np.argmax(sims))
# #         if sims[best_idx] >= threshold:
# #             return cache[cached_keys[best_idx]]
# #     except Exception:
# #         pass
# #     return None


# # # ==========================================================================================
# # # HALLUCINATION GROUNDING
# # # ==========================================================================================

# # def compute_grounding_score(answer: str, context: str, sent_model) -> float:
# #     try:
# #         embs = sent_model.encode([answer[:512], context[:512]], normalize_embeddings=True)
# #         return float(embs[0] @ embs[1])
# #     except Exception:
# #         return 1.0


# # def confidence_label(score: float) -> tuple[str, str]:
# #     if score >= CFG.CONFIDENCE_HIGH:
# #         return "High", "🟢"
# #     if score >= CFG.CONFIDENCE_MED:
# #         return "Medium", "🟡"
# #     return "Low", "🔴"


# # # ==========================================================================================
# # # FOLLOW-UP SUGGESTIONS  — UPDATED v8
# # # ==========================================================================================
# # # Changes:
# # #   • Returns [] immediately when answer is the "could not find" fallback phrase.
# # #   • Passes a context excerpt to the LLM so suggestions are restricted to
# # #     topics present in the retrieved chunks — not random answerable questions.

# # def generate_followups(
# #     query   : str,
# #     answer  : str,
# #     llm,
# #     context : str = "",
# # ) -> list[str]:
# #     """
# #     Generate 3 follow-up questions grounded in the retrieved context.
# #     Returns an empty list when the answer is the fallback phrase or on any error.
# #     """
# #     if "could not find this information" in answer.lower():
# #         return []

# #     context_snippet = context[:300].strip() if context else ""
# #     grounding_note  = (
# #         f"\nOnly suggest questions answerable from this context excerpt:\n"
# #         f'"""{context_snippet}"""'
# #         if context_snippet else ""
# #     )

# #     prompt = f"""You are a helpful assistant for Loyola College, Chennai.
# # Suggest exactly 3 short follow-up questions a student might ask, based on the answer below.
# # Each question MUST be answerable from the same topic/document as the answer.
# # Do NOT suggest questions about unrelated topics.{grounding_note}

# # Question: {query}
# # Answer: {answer[:300]}

# # Output ONLY a JSON array of 3 question strings. No explanation. No markdown.
# # Example: ["Question one?", "Question two?", "Question three?"]

# # JSON array:"""

# #     try:
# #         result = ""
# #         for chunk in llm.stream(prompt):
# #             result += chunk.content
# #         match = re.search(r"\[.*?\]", result, re.DOTALL)
# #         if match:
# #             suggestions = json.loads(match.group())
# #             if isinstance(suggestions, list):
# #                 cleaned = [s.strip() for s in suggestions if isinstance(s, str) and len(s) > 5]
# #                 return cleaned[:3]
# #     except Exception:
# #         pass
# #     return []


# # # ==========================================================================================
# # # CONTEXT BUILDER
# # # ==========================================================================================

# # def build_context(
# #     scored_docs: list[tuple[float, Document]],
# # ) -> tuple[str, list[str], list[dict]]:
# #     """
# #     Returns:
# #         context_str   — text block for the LLM prompt
# #         sources       — list[str] of raw source paths/URLs
# #         rich_sources  — list[dict] with full attribution data for UI
# #     """
# #     parts, sources, rich_sources, tokens = [], [], [], 0

# #     for score, d in scored_docs:
# #         txt = d.page_content[:2500]
# #         t   = count_tokens(txt)
# #         if tokens + t > CFG.MAX_CONTEXT_TOKENS:
# #             break

# #         source  = d.metadata.get("source", "")
# #         title   = d.metadata.get("title", "")
# #         page    = d.metadata.get("page", "")
# #         dtype   = d.metadata.get("type", "")
# #         section = d.metadata.get("section", "")

# #         header = f"[SOURCE: {title or source}"
# #         if page and str(page) != "N/A":
# #             header += f" | Page {page}"
# #         header += "]"

# #         parts.append(f"{header}\n{txt}")
# #         tokens += t

# #         if source and source not in sources:
# #             sources.append(source)
# #             rich_sources.append({
# #                 "source"       : source,
# #                 "title"        : title or source,
# #                 "page"         : str(page) if page and str(page) != "N/A" else "",
# #                 "type"         : dtype,
# #                 "section"      : section,
# #                 "rerank_score" : round(float(score), 3),
# #             })

# #     return "\n\n---\n\n".join(parts), sources, rich_sources


# # # ==========================================================================================
# # # LLM
# # # ==========================================================================================

# # def get_llm(api_key: str):
# #     return ChatGroq(
# #         groq_api_key = api_key,
# #         model_name   = CFG.LLM_MODEL,
# #         temperature  = 0.1,
# #         streaming    = True,
# #     )


# # # ==========================================================================================
# # # PROMPT  — UPDATED v8
# # # ==========================================================================================
# # # Changes:
# # #   • Rule 6 adds explicit formatting instructions (bullets for lists, tables for
# # #     multi-attribute comparisons, plain sentences for single facts).
# # #   • "statistics" class hint preserved and strengthened.
# # #   • Rule count renumbered; all existing anti-hallucination rules preserved.

# # def build_prompt(context: str, question: str, history: list[dict], q_class: str) -> str:
# #     history_text = ""
# #     for h in history[-CFG.HISTORY_TURNS:]:
# #         history_text += f"Student: {h['q']}\nAssistant: {h['a']}\n\n"

# #     class_hints = {
# #         "statistics"  : (
# #             "Look for college-wide totals, not department-specific sub-numbers. "
# #             "If only department figures are present, state that a college-wide total was not found. "
# #             "Always include the year or source of the figure."
# #         ),
# #         "fee"         : "Report exact fee amounts and payment schedules only if present in context.",
# #         "eligibility" : "State admission criteria, cutoff marks, and required documents exactly as written.",
# #         "date"        : "State exact dates and deadlines only. Do not estimate or approximate.",
# #         "fact"        : "Give a single direct factual answer in one or two sentences.",
# #         "followup"    : "Continue consistently with the conversation history above.",
# #         "general"     : "Give a complete, accurate answer using only what is in the context.",
# #     }
# #     hint = class_hints.get(q_class, class_hints["general"])

# #     return f"""You are the official AI assistant for Loyola College (Autonomous), Chennai, Tamil Nadu.

# # ABSOLUTE RULES — obey every rule without exception:
# # 1. Answer ONLY using facts explicitly stated in the CONTEXT block below. Nothing else.
# # 2. Do NOT combine figures or facts from different, unrelated sections of the context.
# # 3. Do NOT infer, estimate, or generate any fact not written word-for-word in the context.
# # 4. Do NOT mention URLs, email addresses, or phone numbers unless the user explicitly asked for them.
# # 5. If the answer is not clearly present in the context, output this sentence exactly and nothing else:
# #    "I could not find this information in the knowledge base."
# # 6. FORMAT RULES:
# #    – Use bullet points (–) for lists of 3 or more items.
# #    – Use a markdown table when comparing multiple attributes across multiple items.
# #    – Use plain sentences for a single fact or a two-item answer.
# # 7. Maximum 6 sentences or 10 bullet points. Be direct and specific.
# # 8. Do NOT use meta-phrases like "Based on the context", "According to the document", "However", "Additionally".
# # 9. Do NOT repeat, rephrase, or acknowledge the question.
# # 10. {hint}

# # CONVERSATION HISTORY:
# # {history_text}
# # ====== CONTEXT — use ONLY this ======
# # {context}
# # =====================================

# # Question: {question}

# # Answer:"""


# # # ==========================================================================================
# # # MAIN ASK FUNCTION
# # # ==========================================================================================

# # def ask(
# #     query      : str,
# #     retriever  : HybridRRFRetriever,
# #     reranker,
# #     llm_mdl,
# #     sent_model,
# #     history    : list[dict],
# #     cache      : dict,
# #     facts      : dict,
# #     type_filter: list[str] = None,
# #     debug      : bool      = False,
# # ) -> dict:
# #     """
# #     Full RAG pipeline. Returns:
# #     {
# #         answer, sources, rich_sources,
# #         confidence, conf_emoji, score,
# #         followups, cache_hit, q_class,
# #         facts_hit, log_id,
# #     }
# #     """
# #     t_start = time.time()

# #     if is_malicious(query):
# #         return {
# #             "answer": "⚠️ I can't process that request.", "sources": [],
# #             "rich_sources": [], "confidence": "N/A", "conf_emoji": "🔴",
# #             "score": 0.0, "followups": [], "cache_hit": False,
# #             "q_class": "blocked", "facts_hit": False, "log_id": None,
# #         }

# #     q_class     = classify_query(query, has_history=bool(history))
# #     retrieval_k = get_retrieval_k(q_class)

# #     # ── Facts layer ───────────────────────────────────────────────────
# #     fact_answer = query_facts_layer(query, facts)
# #     if fact_answer:
# #         lid = log_query(query, q_class, time.time() - t_start, False, "High", True, "facts")
# #         return {
# #             "answer"      : fact_answer,
# #             "sources"     : ["Loyola College — Official Facts"],
# #             "rich_sources": [{"source": "facts", "title": "Loyola College — Official Facts",
# #                                "page": "", "type": "facts", "section": "facts", "rerank_score": 1.0}],
# #             "confidence"  : "High",
# #             "conf_emoji"  : "🟢",
# #             "score"       : 1.0,
# #             "followups"   : generate_followups(query, fact_answer, llm_mdl),
# #             "cache_hit"   : False,
# #             "q_class"     : q_class,
# #             "facts_hit"   : True,
# #             "log_id"      : lid,
# #         }

# #     # ── Exact cache ───────────────────────────────────────────────────
# #     cache_key = query.lower().strip()
# #     if cache_key in cache:
# #         r   = cache[cache_key]
# #         lid = log_query(query, q_class, time.time() - t_start, True, r.get("confidence", "?"), True)
# #         return {**r, "cache_hit": True, "log_id": lid}

# #     # ── Semantic cache ────────────────────────────────────────────────
# #     sem_hit = semantic_cache_lookup(query, cache, sent_model)
# #     if sem_hit:
# #         lid = log_query(query, q_class, time.time() - t_start, True, sem_hit.get("confidence", "?"), True)
# #         return {**sem_hit, "cache_hit": True, "log_id": lid}

# #     # ── Query rewriting ───────────────────────────────────────────────
# #     effective_query = query
# #     if q_class == "followup" and history:
# #         effective_query = rewrite_query(query, history, llm_mdl)

# #     # ── Multi-query RRF retrieval ─────────────────────────────────────
# #     retrieval_queries = [effective_query, f"Loyola College {effective_query}"]
# #     all_docs : dict[str, Document] = {}
# #     rrf_agg  : dict[str, float]   = {}

# #     for q in retrieval_queries:
# #         batch = retriever.invoke(q, k=retrieval_k, type_filter=type_filter)
# #         for rank, doc in enumerate(batch):
# #             key = doc.page_content[:200]
# #             all_docs[key] = doc
# #             rrf_agg[key]  = rrf_agg.get(key, 0.0) + 1.0 / (CFG.RRF_K + rank + 1)

# #     top_keys = sorted(rrf_agg, key=lambda x: rrf_agg[x], reverse=True)
# #     merged   = [all_docs[k] for k in top_keys]

# #     # ── Rerank ───────────────────────────────────────────────────────
# #     scored_docs = rerank(effective_query, merged, reranker, debug=debug)

# #     # ── Build context ─────────────────────────────────────────────────
# #     context, sources, rich_sources = build_context(scored_docs)

# #     if not context.strip():
# #         answer = "I could not find this information in the knowledge base."
# #         lid = log_query(query, q_class, time.time() - t_start, False, "Low", False)
# #         return {
# #             "answer": answer, "sources": [], "rich_sources": [],
# #             "confidence": "Low", "conf_emoji": "🔴", "score": 0.0,
# #             "followups": [], "cache_hit": False, "q_class": q_class,
# #             "facts_hit": False, "log_id": lid,
# #         }

# #     # ── LLM generation ────────────────────────────────────────────────
# #     prompt = build_prompt(context, effective_query, history, q_class)
# #     answer = ""
# #     for chunk in llm_mdl.stream(prompt):
# #         answer += chunk.content
# #     answer = answer.strip()

# #     # ── Hallucination grounding ───────────────────────────────────────
# #     score = compute_grounding_score(answer, context, sent_model)
# #     conf_lbl, conf_emoji = confidence_label(score)

# #     if score < CFG.GROUNDING_THRESHOLD:
# #         answer += (
# #             "\n\n⚠️ *Low confidence: this answer may not be fully supported by "
# #             "the available information. Please verify at www.loyolacollege.edu.*"
# #         )

# #     # ── Follow-ups — v8: grounded in context ─────────────────────────
# #     followups = generate_followups(query, answer, llm_mdl, context=context)

# #     top_section = rich_sources[0]["section"] if rich_sources else ""
# #     lid = log_query(query, q_class, time.time() - t_start, False, conf_lbl, True, top_section)

# #     result = {
# #         "answer"      : answer,
# #         "sources"     : sources,
# #         "rich_sources": rich_sources,
# #         "confidence"  : conf_lbl,
# #         "conf_emoji"  : conf_emoji,
# #         "score"       : round(score, 3),
# #         "followups"   : followups,
# #         "cache_hit"   : False,
# #         "q_class"     : q_class,
# #         "facts_hit"   : False,
# #         "log_id"      : lid,
# #     }

# #     cache[cache_key] = result
# #     _save_cache(cache)
# #     return result


# # # ==========================================================================================
# # # RETRIEVAL EVALUATION FRAMEWORK
# # # ==========================================================================================

# # def _load_eval_dataset() -> list[dict]:
# #     p = Path(CFG.EVAL_DATASET_FILE)
# #     if not p.exists():
# #         return []
# #     try:
# #         data = json.loads(p.read_text(encoding="utf-8"))
# #         if isinstance(data, list):
# #             return data
# #     except Exception:
# #         pass
# #     return []


# # def _precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
# #     hits = sum(1 for r in retrieved[:k] if any(rel in r for rel in relevant))
# #     return hits / k if k > 0 else 0.0


# # def _recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
# #     if not relevant:
# #         return 0.0
# #     hits = sum(1 for r in retrieved[:k] if any(rel in r for rel in relevant))
# #     return hits / len(relevant)


# # def _mrr(retrieved: list[str], relevant: list[str]) -> float:
# #     for i, r in enumerate(retrieved):
# #         if any(rel in r for rel in relevant):
# #             return 1.0 / (i + 1)
# #     return 0.0


# # def _dcg(retrieved: list[str], relevant: list[str], k: int) -> float:
# #     dcg = 0.0
# #     for i, r in enumerate(retrieved[:k]):
# #         rel = 1.0 if any(rel in r for rel in relevant) else 0.0
# #         dcg += rel / math.log2(i + 2)
# #     return dcg


# # def _ndcg(retrieved: list[str], relevant: list[str], k: int) -> float:
# #     ideal = sorted([1.0] * min(len(relevant), k) + [0.0] * max(0, k - len(relevant)), reverse=True)
# #     idcg  = sum(v / math.log2(i + 2) for i, v in enumerate(ideal))
# #     if idcg == 0:
# #         return 0.0
# #     return _dcg(retrieved, relevant, k) / idcg


# # def run_retrieval_evaluation(
# #     retriever : HybridRRFRetriever,
# #     k         : int = 5,
# #     run_label : str = "",
# # ) -> Optional[dict]:
# #     dataset = _load_eval_dataset()
# #     if not dataset:
# #         return None

# #     precisions, recalls, mrrs, ndcgs = [], [], [], []

# #     for item in dataset:
# #         question = item.get("question", "")
# #         relevant = item.get("relevant_sources", [])
# #         if not question or not relevant:
# #             continue
# #         docs      = retriever.invoke(question, k=k)
# #         retrieved = [d.metadata.get("source", "") for d in docs]
# #         precisions.append(_precision_at_k(retrieved, relevant, k))
# #         recalls.append(_recall_at_k(retrieved, relevant, k))
# #         mrrs.append(_mrr(retrieved, relevant))
# #         ndcgs.append(_ndcg(retrieved, relevant, k))

# #     if not precisions:
# #         return None

# #     metrics = {
# #         "run_label"  : run_label or datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
# #         "k"          : k,
# #         "precision_k": round(float(np.mean(precisions)), 4),
# #         "recall_k"   : round(float(np.mean(recalls)), 4),
# #         "mrr"        : round(float(np.mean(mrrs)), 4),
# #         "ndcg"       : round(float(np.mean(ndcgs)), 4),
# #         "num_queries": len(precisions),
# #     }

# #     try:
# #         conn = _init_db()
# #         conn.execute(
# #             """INSERT INTO eval_results
# #                (ts, run_label, k, precision_k, recall_k, mrr, ndcg, num_queries)
# #                VALUES (?,?,?,?,?,?,?,?)""",
# #             (
# #                 datetime.datetime.now().isoformat(),
# #                 metrics["run_label"], metrics["k"],
# #                 metrics["precision_k"], metrics["recall_k"],
# #                 metrics["mrr"], metrics["ndcg"], metrics["num_queries"],
# #             ),
# #         )
# #         conn.commit()
# #         conn.close()
# #     except Exception:
# #         pass

# #     return metrics


# # def metrics_to_csv(metrics_rows: list[dict]) -> str:
# #     if not metrics_rows:
# #         return ""
# #     buf    = io.StringIO()
# #     writer = csv.DictWriter(buf, fieldnames=list(metrics_rows[0].keys()))
# #     writer.writeheader()
# #     writer.writerows(metrics_rows)
# #     return buf.getvalue()


# # # ==========================================================================================
# # # DEFAULT EVAL DATASET  — NEW v8
# # # ==========================================================================================
# # # Auto-creates ./data/eval_dataset.json with 25 Loyola-specific questions if the
# # # file does not exist. Silent — no warning ever shown to end users.

# # _DEFAULT_EVAL_DATASET = [
# #     {"question": "How many students are enrolled in Loyola College?",
# #      "relevant_sources": ["loyolacollege", "students", "strength", "enrolment"]},
# #     {"question": "What is the NAAC accreditation grade of Loyola College?",
# #      "relevant_sources": ["naac", "accreditation", "grade", "loyolacollege"]},
# #     {"question": "What UG programmes does Loyola College offer?",
# #      "relevant_sources": ["loyolacollege", "undergraduate", "ug", "programmes"]},
# #     {"question": "What is the placement percentage at Loyola College?",
# #      "relevant_sources": ["placement", "loyolacollege", "recruited", "campus"]},
# #     {"question": "What is the average salary package offered during placements?",
# #      "relevant_sources": ["placement", "salary", "package", "loyolacollege"]},
# #     {"question": "How many faculty members are in Loyola College?",
# #      "relevant_sources": ["faculty", "staff", "teachers", "loyolacollege"]},
# #     {"question": "What is the NIRF ranking of Loyola College?",
# #      "relevant_sources": ["nirf", "ranking", "loyolacollege"]},
# #     {"question": "When was Loyola College established?",
# #      "relevant_sources": ["established", "founded", "1925", "loyolacollege"]},
# #     {"question": "What are the admission requirements for BSc Computer Science?",
# #      "relevant_sources": ["computer science", "admission", "eligibility", "loyolacollege"]},
# #     {"question": "What PG programmes does Loyola College offer?",
# #      "relevant_sources": ["postgraduate", "pg", "msc", "mcom", "loyolacollege"]},
# #     {"question": "What companies recruit from Loyola College?",
# #      "relevant_sources": ["placement", "recruiters", "companies", "loyolacollege"]},
# #     {"question": "What is the fee structure for BCA?",
# #      "relevant_sources": ["bca", "fee", "tuition", "loyolacollege"]},
# #     {"question": "What research facilities are available at Loyola College?",
# #      "relevant_sources": ["research", "laboratory", "facilities", "loyolacollege"]},
# #     {"question": "What is the hostel capacity of Loyola College?",
# #      "relevant_sources": ["hostel", "accommodation", "loyolacollege"]},
# #     {"question": "What scholarships are available at Loyola College?",
# #      "relevant_sources": ["scholarship", "financial aid", "loyolacollege"]},
# #     {"question": "Who is the principal of Loyola College?",
# #      "relevant_sources": ["principal", "loyolacollege", "administration"]},
# #     {"question": "What sports facilities are available at Loyola College?",
# #      "relevant_sources": ["sports", "gymnasium", "playground", "loyolacollege"]},
# #     {"question": "What are the library facilities at Loyola College?",
# #      "relevant_sources": ["library", "loyolacollege", "books", "journals"]},
# #     {"question": "What is the pass percentage in Loyola College university exams?",
# #      "relevant_sources": ["pass", "result", "percentage", "university", "loyolacollege"]},
# #     {"question": "What MBA specialisations are offered at Loyola College?",
# #      "relevant_sources": ["mba", "specialisation", "management", "loyolacollege"]},
# #     {"question": "Does Loyola College offer PhD programmes?",
# #      "relevant_sources": ["phd", "doctorate", "research", "loyolacollege"]},
# #     {"question": "What cultural events are held at Loyola College?",
# #      "relevant_sources": ["cultural", "fest", "events", "loyolacollege"]},
# #     {"question": "What is the intake capacity of BSc Physics?",
# #      "relevant_sources": ["physics", "intake", "seats", "loyolacollege"]},
# #     {"question": "What affiliation does Loyola College have?",
# #      "relevant_sources": ["university of madras", "autonomous", "affiliation", "loyolacollege"]},
# #     {"question": "What international collaborations does Loyola College have?",
# #      "relevant_sources": ["international", "collaboration", "mou", "loyolacollege"]},
# # ]


# # def _ensure_eval_dataset() -> None:
# #     """
# #     Create the default evaluation dataset if the file does not exist.
# #     Completely silent — no Streamlit messages, no print statements.
# #     """
# #     p = Path(CFG.EVAL_DATASET_FILE)
# #     if p.exists():
# #         return
# #     try:
# #         p.parent.mkdir(parents=True, exist_ok=True)
# #         p.write_text(json.dumps(_DEFAULT_EVAL_DATASET, indent=2), encoding="utf-8")
# #     except Exception:
# #         pass


# # # ==========================================================================================
# # # RICH SOURCE CARD UI
# # # ==========================================================================================

# # def render_source_cards(rich_sources: list[dict]) -> None:
# #     if not rich_sources:
# #         return
# #     with st.expander(f"📎 Sources ({len(rich_sources)})"):
# #         for rs in rich_sources:
# #             title  = rs.get("title", "") or rs.get("source", "Unknown")
# #             source = rs.get("source", "")
# #             page   = rs.get("page", "")
# #             dtype  = rs.get("type", "")
# #             score  = rs.get("rerank_score", None)

# #             col_main, col_meta = st.columns([3, 1])

# #             with col_main:
# #                 if source.startswith("http"):
# #                     st.markdown(f"**[{title}]({source})**")
# #                 else:
# #                     st.markdown(f"**{title}**")
# #                     st.caption(source)

# #             with col_meta:
# #                 badges = []
# #                 if dtype:
# #                     badges.append(f"`{dtype}`")
# #                 if page:
# #                     badges.append(f"p.{page}")
# #                 if score is not None:
# #                     badges.append(f"score: `{score:.3f}`")
# #                 st.markdown("  ".join(badges))

# #             st.divider()


# # # ==========================================================================================
# # # ANALYTICS TAB
# # # ==========================================================================================

# # def render_analytics_tab() -> None:
# #     st.header("📊 Query Analytics")
# #     try:
# #         conn  = _init_db()
# #         total = conn.execute("SELECT COUNT(*) FROM query_log").fetchone()[0]

# #         if total == 0:
# #             st.info("No queries logged yet.")
# #             conn.close()
# #             return

# #         avg_lat      = conn.execute("SELECT AVG(latency_s) FROM query_log").fetchone()[0]
# #         cache_pct    = conn.execute("SELECT 100.0*SUM(cache_hit)/COUNT(*) FROM query_log").fetchone()[0]
# #         answered_pct = conn.execute("SELECT 100.0*SUM(answered)/COUNT(*) FROM query_log").fetchone()[0]

# #         c1, c2, c3, c4 = st.columns(4)
# #         c1.metric("Total queries",  total)
# #         c2.metric("Avg latency",    f"{avg_lat:.2f}s" if avg_lat else "–")
# #         c3.metric("Cache hit rate", f"{cache_pct:.1f}%" if cache_pct else "0%")
# #         c4.metric("Answer rate",    f"{answered_pct:.1f}%" if answered_pct else "0%")

# #         st.divider()
# #         col_left, col_right = st.columns(2)

# #         with col_left:
# #             st.subheader("📂 Query types")
# #             class_rows = conn.execute(
# #                 "SELECT q_class, COUNT(*) as n FROM query_log GROUP BY q_class ORDER BY n DESC"
# #             ).fetchall()
# #             if class_rows:
# #                 df_class = pd.DataFrame(class_rows, columns=["Query type", "Count"])
# #                 st.bar_chart(df_class.set_index("Query type"))

# #         with col_right:
# #             st.subheader("🌐 Top website sections")
# #             sec_rows = conn.execute(
# #                 """SELECT section, COUNT(*) as n FROM query_log
# #                    WHERE section IS NOT NULL AND section != ''
# #                    GROUP BY section ORDER BY n DESC LIMIT 10"""
# #             ).fetchall()
# #             if sec_rows:
# #                 df_sec = pd.DataFrame(sec_rows, columns=["Section", "Count"])
# #                 st.bar_chart(df_sec.set_index("Section"))
# #             else:
# #                 st.caption("No section data yet.")

# #         st.divider()
# #         st.subheader("🔝 Top queries")
# #         rows = conn.execute(
# #             "SELECT query, COUNT(*) as n FROM query_log GROUP BY LOWER(query) ORDER BY n DESC LIMIT 10"
# #         ).fetchall()
# #         for q, n in rows:
# #             st.write(f"- ({n}×) {q}")

# #         st.divider()
# #         st.subheader("🕐 Recent queries")
# #         recent = conn.execute(
# #             """SELECT ts, query, q_class, section, latency_s, cache_hit, confidence
# #                FROM query_log ORDER BY id DESC LIMIT 25"""
# #         ).fetchall()
# #         df_recent = pd.DataFrame(
# #             recent,
# #             columns=["Timestamp", "Query", "Class", "Section", "Latency (s)", "Cache hit", "Confidence"],
# #         )
# #         st.dataframe(df_recent, use_container_width=True)
# #         conn.close()

# #     except Exception as e:
# #         st.warning(f"Analytics unavailable: {e}")


# # # ==========================================================================================
# # # FAILURE ANALYTICS TAB
# # # ==========================================================================================

# # def render_failures_tab() -> None:
# #     st.header("⚠️ Failure Analytics")
# #     try:
# #         conn       = _init_db()
# #         total      = conn.execute("SELECT COUNT(*) FROM query_log").fetchone()[0]
# #         unanswered = conn.execute("SELECT COUNT(*) FROM query_log WHERE answered=0").fetchone()[0]
# #         low_conf   = conn.execute(
# #             "SELECT COUNT(*) FROM query_log WHERE confidence='Low' AND answered=1"
# #         ).fetchone()[0]

# #         c1, c2, c3 = st.columns(3)
# #         c1.metric("Total queries", total)
# #         c2.metric("Unanswered",    unanswered,
# #                   delta=f"{100*unanswered/max(total,1):.1f}% failure rate", delta_color="inverse")
# #         c3.metric("Low-confidence", low_conf,
# #                   delta=f"{100*low_conf/max(total,1):.1f}% of queries", delta_color="inverse")

# #         st.divider()
# #         col_l, col_r = st.columns(2)

# #         with col_l:
# #             st.subheader("❌ Unanswered queries")
# #             rows = conn.execute(
# #                 "SELECT query, ts FROM query_log WHERE answered=0 ORDER BY id DESC LIMIT 20"
# #             ).fetchall()
# #             if rows:
# #                 st.dataframe(pd.DataFrame(rows, columns=["Query", "Timestamp"]), use_container_width=True)
# #             else:
# #                 st.success("No unanswered queries yet!")

# #         with col_r:
# #             st.subheader("🟡 Low-confidence answers")
# #             rows = conn.execute(
# #                 """SELECT query, section, ts FROM query_log
# #                    WHERE confidence='Low' AND answered=1 ORDER BY id DESC LIMIT 20"""
# #             ).fetchall()
# #             if rows:
# #                 st.dataframe(pd.DataFrame(rows, columns=["Query", "Section", "Timestamp"]), use_container_width=True)
# #             else:
# #                 st.success("No low-confidence answers logged yet.")

# #         st.divider()
# #         st.subheader("📉 Sections with low answer rate")
# #         sec_fail = conn.execute(
# #             """SELECT section,
# #                       COUNT(*) as total,
# #                       SUM(CASE WHEN answered=0 THEN 1 ELSE 0 END) as failed,
# #                       ROUND(100.0*SUM(CASE WHEN answered=0 THEN 1 ELSE 0 END)/COUNT(*),1) as pct
# #                FROM query_log
# #                WHERE section IS NOT NULL AND section != ''
# #                GROUP BY section HAVING total >= 2
# #                ORDER BY pct DESC LIMIT 10"""
# #         ).fetchall()
# #         if sec_fail:
# #             df_sf = pd.DataFrame(sec_fail, columns=["Section", "Total", "Failed", "Fail %"])
# #             st.dataframe(df_sf, use_container_width=True)
# #             st.bar_chart(df_sf.set_index("Section")["Fail %"])
# #         else:
# #             st.caption("Not enough section data yet.")

# #         conn.close()
# #     except Exception as e:
# #         st.warning(f"Failure analytics unavailable: {e}")


# # # ==========================================================================================
# # # FEEDBACK TAB
# # # ==========================================================================================

# # def render_feedback_tab() -> None:
# #     st.header("👍 User Feedback")
# #     try:
# #         conn     = _init_db()
# #         total_fb = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]

# #         if total_fb == 0:
# #             st.info("No feedback recorded yet.")
# #             conn.close()
# #             return

# #         helpful     = conn.execute("SELECT COUNT(*) FROM feedback WHERE helpful=1").fetchone()[0]
# #         pct         = 100 * helpful / total_fb if total_fb else 0

# #         c1, c2, c3 = st.columns(3)
# #         c1.metric("Total feedback",    total_fb)
# #         c2.metric("👍 Helpful",        helpful)
# #         c3.metric("Satisfaction rate", f"{pct:.1f}%")

# #         st.divider()
# #         col_l, col_r = st.columns(2)

# #         with col_l:
# #             st.subheader("📅 Feedback over time")
# #             daily = conn.execute(
# #                 """SELECT substr(ts,1,10) as day,
# #                           SUM(helpful) as helpful, SUM(1-helpful) as not_helpful
# #                    FROM feedback GROUP BY day ORDER BY day"""
# #             ).fetchall()
# #             if daily:
# #                 st.bar_chart(pd.DataFrame(daily, columns=["Date", "Helpful", "Not helpful"]).set_index("Date"))

# #         with col_r:
# #             st.subheader("👎 Most disliked queries")
# #             disliked = conn.execute(
# #                 """SELECT query, COUNT(*) as thumbs_down FROM feedback WHERE helpful=0
# #                    GROUP BY LOWER(query) ORDER BY thumbs_down DESC LIMIT 10"""
# #             ).fetchall()
# #             if disliked:
# #                 st.dataframe(pd.DataFrame(disliked, columns=["Query", "Thumbs down"]), use_container_width=True)

# #         st.divider()
# #         st.subheader("🕐 Recent feedback")
# #         recent = conn.execute(
# #             """SELECT ts, query,
# #                       CASE helpful WHEN 1 THEN '👍' ELSE '👎' END as vote
# #                FROM feedback ORDER BY id DESC LIMIT 30"""
# #         ).fetchall()
# #         st.dataframe(pd.DataFrame(recent, columns=["Timestamp", "Query", "Vote"]), use_container_width=True)
# #         conn.close()

# #     except Exception as e:
# #         st.warning(f"Feedback analytics unavailable: {e}")


# # # ==========================================================================================
# # # EVALUATION TAB  — UPDATED v8
# # # ==========================================================================================
# # # _ensure_eval_dataset() called silently at the top — file is created automatically
# # # if missing so users never see a "No evaluation dataset found" warning.

# # def render_evaluation_tab(retriever: HybridRRFRetriever) -> None:
# #     st.header("📐 Retrieval Evaluation")

# #     _ensure_eval_dataset()          # silently create default dataset if missing
# #     dataset = _load_eval_dataset()

# #     if not dataset:
# #         # Only reachable if the file couldn't be written (e.g. permissions)
# #         st.info("Evaluation dataset unavailable. Add `data/eval_dataset.json` to enable this tab.")
# #         return

# #     st.success(f"✅ Evaluation dataset — {len(dataset)} questions")

# #     col_k, col_label, col_run = st.columns([1, 2, 1])
# #     k         = col_k.selectbox("Top-K to evaluate", [3, 5, 10], index=1)
# #     run_label = col_label.text_input("Run label (optional)", placeholder="e.g. v8-baseline")
# #     run_now   = col_run.button("▶ Run evaluation", use_container_width=True)

# #     if run_now:
# #         with st.spinner(f"Evaluating {len(dataset)} queries at K={k}..."):
# #             metrics = run_retrieval_evaluation(retriever, k=k, run_label=run_label)
# #         if metrics:
# #             st.success("Evaluation complete!")
# #             mc1, mc2, mc3, mc4 = st.columns(4)
# #             mc1.metric(f"Precision@{k}", f"{metrics['precision_k']:.4f}")
# #             mc2.metric(f"Recall@{k}",    f"{metrics['recall_k']:.4f}")
# #             mc3.metric("MRR",            f"{metrics['mrr']:.4f}")
# #             mc4.metric(f"NDCG@{k}",      f"{metrics['ndcg']:.4f}")
# #         else:
# #             st.error("Evaluation failed — check dataset format.")

# #     st.divider()
# #     st.subheader("📜 Evaluation history")

# #     try:
# #         conn = _init_db()
# #         rows = conn.execute(
# #             """SELECT ts, run_label, k, precision_k, recall_k, mrr, ndcg, num_queries
# #                FROM eval_results ORDER BY id DESC LIMIT 20"""
# #         ).fetchall()
# #         conn.close()

# #         if rows:
# #             cols    = ["Timestamp", "Run label", "K", "Precision@K", "Recall@K", "MRR", "NDCG", "Queries"]
# #             df_eval = pd.DataFrame(rows, columns=cols)
# #             st.dataframe(df_eval, use_container_width=True)

# #             csv_str = metrics_to_csv([dict(zip(cols, r)) for r in rows])
# #             st.download_button(
# #                 label     = "⬇️ Export evaluation history as CSV",
# #                 data      = csv_str,
# #                 file_name = "loyola_rag_eval_results.csv",
# #                 mime      = "text/csv",
# #             )

# #             if len(df_eval) > 1:
# #                 st.subheader("📈 NDCG trend across runs")
# #                 st.line_chart(df_eval.set_index("Run label")["NDCG"])
# #         else:
# #             st.info("No evaluation runs yet. Click 'Run evaluation' above.")

# #     except Exception as e:
# #         st.warning(f"Could not load evaluation history: {e}")


# # # ==========================================================================================
# # # STREAMLIT UI
# # # ==========================================================================================

# # def main():

# #     st.markdown("""
# #         <div style='text-align:center; padding: 1rem 0'>
# #             <h1>🎓 Loyola College Assistant</h1>
# #             <p style='color: gray'>Ask anything about Loyola College, Chennai — v8.0</p>
# #         </div>
# #     """, unsafe_allow_html=True)

# #     # ── Sidebar ───────────────────────────────────────────────────────
# #     st.sidebar.image(
# #         "https://www.loyolacollege.edu/wp-content/uploads/2021/09/loyola-logo.png",
# #         width=120,
# #     )
# #     st.sidebar.title("⚙️ Settings")

# #     api_key = os.getenv("GROQ_API_KEY", "").strip()
# #     if not api_key:
# #         st.error("❌ GROQ_API_KEY not found in .env file.")
# #         st.stop()

# #     st.sidebar.divider()

# #     if st.sidebar.button("🔄 Rebuild Knowledge Base", use_container_width=True):
# #         for f in [CFG.CHUNKS_PKL, CFG.BM25_PKL, CFG.HASH_FILE, CFG.QUERY_CACHE_FILE]:
# #             if os.path.exists(f):
# #                 os.remove(f)
# #         for f in ["index.faiss", "index.pkl"]:
# #             fp = Path(CFG.FAISS_PATH) / f
# #             if fp.exists():
# #                 fp.unlink()
# #         st.cache_resource.clear()
# #         st.sidebar.success("Cleared — rebuilding...")
# #         st.rerun()

# #     # Source type filter
# #     st.sidebar.divider()
# #     st.sidebar.caption("🔍 Source filter (optional)")
# #     selected_types = st.sidebar.multiselect(
# #         "Restrict retrieval to",
# #         options = ["website", "pdf", "ocr", "cleaned_text"],
# #         default = [],
# #         help    = "Leave blank to search all sources.",
# #     )
# #     active_filter = selected_types if selected_types else None

# #     # Debug mode
# #     st.sidebar.divider()
# #     debug_mode = st.sidebar.checkbox(
# #         "🐛 Debug mode",
# #         value = False,
# #         help  = "Show retrieved chunks and rerank scores under each answer.",
# #     )

# #     st.sidebar.divider()
# #     st.sidebar.caption("📁 Knowledge Base Status")
# #     sidebar_log = st.sidebar.container()

# #     if can_fast_load():
# #         st.sidebar.success("⚡ Index ready — fast load")
# #     else:
# #         st.sidebar.warning("🔄 First run — full build required")

# #     with st.sidebar.expander("ℹ️ System info"):
# #         st.caption(f"Device: `{DEVICE}`")
# #         st.caption(f"Embedding: `{CFG.EMBEDDING_MODEL}`")
# #         st.caption(f"Reranker: `{CFG.RERANK_MODEL}`")
# #         st.caption(f"LLM: `{CFG.LLM_MODEL}`")
# #         st.caption(f"Chunk: {CFG.CHUNK_SIZE} / overlap {CFG.CHUNK_OVERLAP}")
# #         st.caption(f"Rerank K={CFG.RERANK_K} | threshold={CFG.RERANK_THRESHOLD}")
# #         st.caption(f"Retrieval K — default:{CFG.RETRIEVAL_K_DEFAULT} deep:{CFG.RETRIEVAL_K_DEEP}")

# #     # ── Load models ───────────────────────────────────────────────────
# #     with st.spinner("Loading AI models..."):
# #         emb        = load_embedding_model()
# #         reranker   = load_reranker()
# #         sent_model = load_sentence_model()
# #         llm_mdl    = get_llm(api_key)
# #         facts      = load_facts()

# #     # ── Load knowledge base ───────────────────────────────────────────
# #     with st.spinner(
# #         "Loading knowledge base..." if can_fast_load()
# #         else "Building knowledge base (first time only)..."
# #     ):
# #         db, chunks, bm25 = build_knowledge_base(sidebar_log, emb)
# #         retriever = HybridRRFRetriever(db, bm25)

# #     st.success(f"✅ Ready — {len(chunks):,} chunks loaded")

# #     # ── Session state ─────────────────────────────────────────────────
# #     if "chat_history" not in st.session_state:
# #         st.session_state.chat_history = []
# #     if "query_cache" not in st.session_state:
# #         st.session_state.query_cache = _load_cache()

# #     # ── Tabs ──────────────────────────────────────────────────────────
# #     tab_chat, tab_analytics, tab_failures, tab_feedback, tab_eval = st.tabs([
# #         "💬 Chat", "📊 Analytics", "⚠️ Failures", "👍 Feedback", "📐 Evaluation",
# #     ])

# #     # ================================================================
# #     # CHAT TAB
# #     # ================================================================
# #     with tab_chat:

# #         # Render conversation history
# #         for idx, chat in enumerate(st.session_state.chat_history):
# #             with st.chat_message("user"):
# #                 st.markdown(chat["q"])
# #             with st.chat_message("assistant"):
# #                 st.markdown(chat["a"])

# #                 conf      = chat.get("confidence", "?")
# #                 emoji     = chat.get("conf_emoji", "")
# #                 score     = chat.get("score", 0.0)
# #                 facts_hit = chat.get("facts_hit", False)

# #                 if facts_hit:
# #                     st.caption("⚡ Answered from structured facts layer")
# #                 elif conf != "?":
# #                     st.caption(f"{emoji} Confidence: **{conf}** (grounding score: {score:.2f})")

# #                 rich_sources = chat.get("rich_sources", [])
# #                 if rich_sources:
# #                     render_source_cards(rich_sources)
# #                 elif chat.get("sources"):
# #                     with st.expander("📎 Sources"):
# #                         for src in chat["sources"]:
# #                             if src.startswith("http"):
# #                                 st.markdown(f"- [{src}]({src})")
# #                             else:
# #                                 st.caption(f"- {src}")

# #                 if chat.get("followups"):
# #                     st.markdown("**💡 You might also ask:**")
# #                     cols = st.columns(len(chat["followups"]))
# #                     for i, fq in enumerate(chat["followups"]):
# #                         if cols[i].button(fq, key=f"fq_hist_{idx}_{i}"):
# #                             st.session_state["prefill_query"] = fq
# #                             st.rerun()

# #                 log_id    = chat.get("log_id")
# #                 voted_key = f"voted_{idx}"
# #                 if not st.session_state.get(voted_key, False):
# #                     fb_cols = st.columns([1, 1, 8])
# #                     if fb_cols[0].button("👍", key=f"up_{idx}"):
# #                         log_feedback(log_id, chat["q"], helpful=True)
# #                         st.session_state[voted_key] = True
# #                         st.rerun()
# #                     if fb_cols[1].button("👎", key=f"dn_{idx}"):
# #                         log_feedback(log_id, chat["q"], helpful=False)
# #                         st.session_state[voted_key] = True
# #                         st.rerun()
# #                 else:
# #                     st.caption("✅ Feedback recorded — thank you!")

# #         # ── Chat input ────────────────────────────────────────────────
# #         prefill    = st.session_state.pop("prefill_query", None)
# #         user_query = st.chat_input(
# #             "Ask about admissions, programmes, placements, fees, faculty…"
# #         ) or prefill

# #         if user_query:

# #             with st.chat_message("user"):
# #                 st.markdown(user_query)

# #             with st.chat_message("assistant"):
# #                 placeholder   = st.empty()
# #                 conf_slot     = st.empty()
# #                 debug_slot    = st.empty()
# #                 sources_slot  = st.empty()
# #                 followup_slot = st.empty()
# #                 feedback_slot = st.empty()

# #                 with st.spinner("Thinking..."):
# #                     result = ask(
# #                         query       = user_query,
# #                         retriever   = retriever,
# #                         reranker    = reranker,
# #                         llm_mdl     = llm_mdl,
# #                         sent_model  = sent_model,
# #                         history     = st.session_state.chat_history,
# #                         cache       = st.session_state.query_cache,
# #                         facts       = facts,
# #                         type_filter = active_filter,
# #                         debug       = debug_mode,
# #                     )

# #                 answer       = result["answer"]
# #                 sources      = result["sources"]
# #                 rich_sources = result.get("rich_sources", [])
# #                 conf         = result["confidence"]
# #                 emoji        = result["conf_emoji"]
# #                 score        = result["score"]
# #                 followups    = result["followups"]
# #                 q_class      = result["q_class"]
# #                 cache_hit    = result["cache_hit"]
# #                 facts_hit    = result["facts_hit"]
# #                 log_id       = result.get("log_id")

# #                 placeholder.markdown(answer)

# #                 badges = []
# #                 if facts_hit:
# #                     badges.append("⚡ Structured facts")
# #                 elif conf != "N/A":
# #                     badges.append(f"{emoji} Confidence: **{conf}** (score: {score:.2f})")
# #                 if cache_hit:
# #                     badges.append("🗂 Cached")
# #                 if active_filter:
# #                     badges.append(f"🔍 Filtered: {', '.join(active_filter)}")
# #                 badges.append(f"🏷 `{q_class}`")
# #                 conf_slot.caption("  |  ".join(badges))

# #                 # Debug expander
# #                 if debug_mode and rich_sources:
# #                     with debug_slot.expander("🐛 Debug: reranked chunks", expanded=False):
# #                         for i, rs in enumerate(rich_sources):
# #                             st.markdown(
# #                                 f"**[{i+1}]** score=`{rs['rerank_score']}` | "
# #                                 f"type=`{rs['type']}` | "
# #                                 f"`{rs['source'][:80]}`"
# #                             )

# #                 if rich_sources:
# #                     with sources_slot.container():
# #                         render_source_cards(rich_sources)
# #                 elif sources:
# #                     with sources_slot.expander("📎 Sources"):
# #                         for src in sources:
# #                             if src.startswith("http"):
# #                                 st.markdown(f"- [{src}]({src})")
# #                             else:
# #                                 st.caption(f"- {src}")

# #                 if followups:
# #                     followup_slot.markdown("**💡 You might also ask:**")
# #                     cols = st.columns(len(followups))
# #                     for i, fq in enumerate(followups):
# #                         if cols[i].button(fq, key=f"fq_new_{i}"):
# #                             st.session_state["prefill_query"] = fq
# #                             st.rerun()

# #                 with feedback_slot.container():
# #                     fb_cols = st.columns([1, 1, 8])
# #                     if fb_cols[0].button("👍", key="fb_up_new"):
# #                         log_feedback(log_id, user_query, helpful=True)
# #                         st.toast("Thanks for the feedback! 👍")
# #                     if fb_cols[1].button("👎", key="fb_dn_new"):
# #                         log_feedback(log_id, user_query, helpful=False)
# #                         st.toast("Thanks for the feedback! 👎")

# #             chat_entry = {
# #                 "q"           : user_query,
# #                 "a"           : answer,
# #                 "sources"     : sources,
# #                 "rich_sources": rich_sources,
# #                 "confidence"  : conf,
# #                 "conf_emoji"  : emoji,
# #                 "score"       : score,
# #                 "followups"   : followups,
# #                 "facts_hit"   : facts_hit,
# #                 "log_id"      : log_id,
# #             }
# #             st.session_state.chat_history.append(chat_entry)
# #             st.session_state.chat_history = \
# #                 st.session_state.chat_history[-CFG.HISTORY_TURNS * 2:]

# #     # ================================================================
# #     # OTHER TABS
# #     # ================================================================
# #     with tab_analytics:
# #         render_analytics_tab()

# #     with tab_failures:
# #         render_failures_tab()

# #     with tab_feedback:
# #         render_feedback_tab()

# #     with tab_eval:
# #         render_evaluation_tab(retriever)


# # # ==========================================================================================
# # # ENTRY POINT
# # # ==========================================================================================

# # if __name__ == "__main__":
# #     main()


































































































# # ==========================================================================================
# # LOYOLA COLLEGE — WEBSITE RAG CHATBOT  v9.0
# # ==========================================================================================
# #
# # WHAT CHANGED FROM v8.0
# # ──────────────────────
# # ✔  Query Expansion          — synonyms + college-specific keywords injected before retrieval
# # ✔  Smarter Query Rewriting  — all ambiguous queries rewritten, not just follow-ups
# # ✔  Dynamic K               — K scales with query complexity (fact=8, stats/compare=28)
# # ✔  Improved Context Builder — dedup, diversity filter, relevance-only chunks
# # ✔  Better Reranking         — candidate pool raised to 40, adaptive threshold
# # ✔  Stronger Hallucination Prompt — tighter rules, explicit "no estimation" clause
# # ✔  Better Grounding Score   — max-sim across ALL retrieved chunks, not first 512 chars
# # ✔  Semantic Cache v2        — stores embeddings alongside cache entries (no recompute)
# #                               LRU eviction, configurable max size, cosine via dot product
# # ✔  Richer Query Classification — 8 classes: fact/stats/fee/eligibility/date/comparison/followup/general
# # ✔  Grounded Follow-ups      — follow-ups only from retrieved context, never hallucinated
# # ✔  Source Attribution       — page numbers, doc title, reranker score preserved
# # ✔  Evaluation Improvements  — summary stats, CSV export, multi-run comparison
# # ✔  Performance              — batch encoding, no repeated embedding calls
# # ✔  UI Improvements          — loading indicators, confidence display, follow-up buttons
# #
# # CARRIED FORWARD FROM v8.0 (all improvements preserved)
# # ───────────────────────────────────────────────────────
# # ✔  Noise chunk filter
# # ✔  Source priority boost
# # ✔  Structured facts layer (with programmes)
# # ✔  FAISS + BM25 + RRF hybrid retrieval
# # ✔  Cross-encoder reranking
# # ✔  Analytics / Feedback / Evaluation tabs
# # ✔  Debug mode
# # ✔  Auto eval dataset creation
# # ✔  Security filter
# #
# # RUN
# # ───
# #   streamlit run rag_v9.py
# #
# # ==========================================================================================


# # ==========================================================================================
# # IMPORTS
# # ==========================================================================================

# import csv
# import io
# import os
# import re
# import json
# import math
# import pickle
# import sqlite3
# import hashlib
# import warnings
# import time
# import datetime
# import unicodedata
# import collections

# from pathlib     import Path
# from dataclasses import dataclass, field
# from typing      import Optional

# import numpy as np
# import pandas as pd
# import streamlit as st
# import tiktoken

# from dotenv                           import load_dotenv
# from sentence_transformers            import CrossEncoder, SentenceTransformer

# from langchain_core.documents         import Document
# from langchain_community.vectorstores import FAISS
# from langchain_community.retrievers   import BM25Retriever

# from langchain_community.document_loaders import (
#     PyMuPDFLoader,
#     PDFPlumberLoader,
# )

# from langchain_text_splitters       import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_groq                 import ChatGroq

# warnings.filterwarnings("ignore")
# load_dotenv()


# # ==========================================================================================
# # PAGE CONFIG
# # ==========================================================================================

# st.set_page_config(
#     page_title = "Loyola College Assistant",
#     page_icon  = "🎓",
#     layout     = "wide",
# )


# # ==========================================================================================
# # CONFIGURATION
# # ==========================================================================================

# @dataclass
# class Config:

#     # ── Data folders ──────────────────────────────────────────────────
#     RAG_JSON_FOLDER    : str = "./data/rag_export"
#     CLEANED_TEXT_FOLDER: str = "./data/cleaned_text"
#     OCR_TEXT_FOLDER    : str = "./data/ocr_text"
#     PDF_FOLDER         : str = "./data/pdfs"
#     FACTS_FILE         : str = "./data/facts.json"
#     EVAL_DATASET_FILE  : str = "./data/eval_dataset.json"

#     # ── Persistence ───────────────────────────────────────────────────
#     FAISS_PATH         : str = "./faiss_index"
#     CHUNKS_PKL         : str = "./faiss_index/chunks.pkl"
#     BM25_PKL           : str = "./faiss_index/bm25.pkl"
#     HASH_FILE          : str = "./faiss_index/data_hash.txt"
#     QUERY_CACHE_FILE   : str = "./faiss_index/query_cache.pkl"
#     ANALYTICS_DB       : str = "./faiss_index/analytics.db"

#     # ── Models ────────────────────────────────────────────────────────
#     EMBEDDING_MODEL    : str = "BAAI/bge-small-en-v1.5"
#     RERANK_MODEL       : str = "BAAI/bge-reranker-base"
#     LLM_MODEL          : str = "llama-3.1-8b-instant"

#     # ── Chunking ──────────────────────────────────────────────────────
#     CHUNK_SIZE         : int = 1200
#     CHUNK_OVERLAP      : int = 200

#     # ── Retrieval — dynamic K by query class ──────────────────────────
#     RETRIEVAL_K_FACT   : int = 8     # simple fact questions
#     RETRIEVAL_K_DEFAULT: int = 16    # general questions
#     RETRIEVAL_K_DEEP   : int = 24    # statistics, eligibility, date
#     RETRIEVAL_K_COMPARE: int = 28    # comparison / analytical questions

#     # ── Reranking ─────────────────────────────────────────────────────
#     RERANK_CANDIDATE_POOL : int   = 40    # raised from 30 for better recall
#     RERANK_K              : int   = 6
#     RERANK_THRESHOLD      : float = 0.15  # base threshold (adaptive in code)

#     # ── Context ───────────────────────────────────────────────────────
#     MAX_CONTEXT_TOKENS : int = 3500
#     RRF_K              : int = 60

#     # ── Semantic cache ────────────────────────────────────────────────
#     SEMANTIC_CACHE_THRESHOLD : float = 0.92
#     SEMANTIC_CACHE_MAX_SIZE  : int   = 500   # LRU eviction above this

#     # ── Hallucination grounding ───────────────────────────────────────
#     GROUNDING_THRESHOLD : float = 0.25
#     CONFIDENCE_HIGH     : float = 0.55
#     CONFIDENCE_MED      : float = 0.35

#     # ── Chat ──────────────────────────────────────────────────────────
#     HISTORY_TURNS       : int = 4


# CFG = Config()
# Path(CFG.FAISS_PATH).mkdir(parents=True, exist_ok=True)


# # ==========================================================================================
# # DEVICE
# # ==========================================================================================

# try:
#     import torch
#     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# except ImportError:
#     DEVICE = "cpu"


# # ==========================================================================================
# # TOKEN COUNTER
# # ==========================================================================================

# _tokenizer = tiktoken.get_encoding("cl100k_base")

# def count_tokens(text: str) -> int:
#     return len(_tokenizer.encode(text))


# # ==========================================================================================
# # QUERY CACHE  — v9: LRU with embedded vectors
# # ==========================================================================================
# # Cache entries now store the query embedding alongside the result so
# # semantic_cache_lookup never re-encodes cached queries.

# def _load_cache() -> "collections.OrderedDict":
#     try:
#         if os.path.exists(CFG.QUERY_CACHE_FILE):
#             obj = pickle.load(open(CFG.QUERY_CACHE_FILE, "rb"))
#             # Upgrade plain dict from v8 → OrderedDict
#             if isinstance(obj, dict) and not isinstance(obj, collections.OrderedDict):
#                 od = collections.OrderedDict(obj)
#                 return od
#             return obj
#     except Exception:
#         pass
#     return collections.OrderedDict()


# def _save_cache(cache: "collections.OrderedDict") -> None:
#     try:
#         pickle.dump(cache, open(CFG.QUERY_CACHE_FILE, "wb"))
#     except Exception:
#         pass


# def _cache_set(cache: "collections.OrderedDict", key: str, value: dict) -> None:
#     """Insert / update with LRU eviction when cache exceeds max size."""
#     if key in cache:
#         cache.move_to_end(key)
#     cache[key] = value
#     while len(cache) > CFG.SEMANTIC_CACHE_MAX_SIZE:
#         cache.popitem(last=False)   # remove oldest (FIFO / LRU)


# # ==========================================================================================
# # DATABASE INIT
# # ==========================================================================================

# def _init_db() -> sqlite3.Connection:
#     """Create / migrate analytics.db with all required tables."""
#     conn = sqlite3.connect(CFG.ANALYTICS_DB)

#     conn.execute("""
#         CREATE TABLE IF NOT EXISTS query_log (
#             id         INTEGER PRIMARY KEY AUTOINCREMENT,
#             ts         TEXT,
#             query      TEXT,
#             q_class    TEXT,
#             latency_s  REAL,
#             cache_hit  INTEGER,
#             confidence TEXT,
#             answered   INTEGER,
#             section    TEXT
#         )
#     """)

#     try:
#         conn.execute("ALTER TABLE query_log ADD COLUMN section TEXT")
#     except sqlite3.OperationalError:
#         pass

#     conn.execute("""
#         CREATE TABLE IF NOT EXISTS feedback (
#             id         INTEGER PRIMARY KEY AUTOINCREMENT,
#             ts         TEXT,
#             log_id     INTEGER,
#             query      TEXT,
#             helpful    INTEGER
#         )
#     """)

#     conn.execute("""
#         CREATE TABLE IF NOT EXISTS eval_results (
#             id           INTEGER PRIMARY KEY AUTOINCREMENT,
#             ts           TEXT,
#             run_label    TEXT,
#             k            INTEGER,
#             precision_k  REAL,
#             recall_k     REAL,
#             mrr          REAL,
#             ndcg         REAL,
#             num_queries  INTEGER
#         )
#     """)

#     conn.commit()
#     return conn


# # ==========================================================================================
# # QUERY ANALYTICS
# # ==========================================================================================

# def log_query(
#     query     : str,
#     q_class   : str,
#     latency_s : float,
#     cache_hit : bool,
#     confidence: str,
#     answered  : bool,
#     section   : str = "",
# ) -> Optional[int]:
#     """Log a query to analytics.db. Returns the inserted row id."""
#     try:
#         conn = _init_db()
#         cur  = conn.execute(
#             """INSERT INTO query_log
#                (ts, query, q_class, latency_s, cache_hit, confidence, answered, section)
#                VALUES (?,?,?,?,?,?,?,?)""",
#             (
#                 datetime.datetime.now().isoformat(),
#                 query, q_class, round(latency_s, 3),
#                 int(cache_hit), confidence, int(answered), section,
#             ),
#         )
#         conn.commit()
#         row_id = cur.lastrowid
#         conn.close()
#         return row_id
#     except Exception:
#         return None


# def log_feedback(log_id: Optional[int], query: str, helpful: bool) -> None:
#     """Record a 👍 or 👎 for a specific query."""
#     try:
#         conn = _init_db()
#         conn.execute(
#             "INSERT INTO feedback (ts, log_id, query, helpful) VALUES (?,?,?,?)",
#             (datetime.datetime.now().isoformat(), log_id, query, int(helpful)),
#         )
#         conn.commit()
#         conn.close()
#     except Exception:
#         pass


# # ==========================================================================================
# # SECURITY FILTER
# # ==========================================================================================

# _BLOCKED = [
#     r"ignore (previous|all) instructions",
#     r"developer mode",
#     r"jailbreak",
#     r"system prompt",
#     r"forget (your|all) instructions",
#     r"act as (an? )?(unrestricted|evil|unethical)",
#     r"pretend you (are|have no)",
#     r"bypass (your )?(safety|filter|restriction)",
#     r"disregard (your )?guidelines",
#     r"you are now",
#     r"new persona",
# ]

# def is_malicious(q: str) -> bool:
#     q_lower = q.lower()
#     if any(re.search(p, q_lower) for p in _BLOCKED):
#         return True
#     q_norm = unicodedata.normalize("NFKD", q_lower)
#     return any(re.search(p, q_norm) for p in _BLOCKED)


# # ==========================================================================================
# # TEXT CLEANER
# # ==========================================================================================

# def clean_text(t: str) -> str:
#     t = re.sub(r"\s+", " ", t)
#     t = re.sub(r"[^\x20-\x7E\u0900-\u097F\u0B80-\u0BFF\n]", "", t)
#     return t.strip()


# # ==========================================================================================
# # NOISE CHUNK FILTER
# # ==========================================================================================

# _NOISE_PATTERNS = [
#     re.compile(r'(https?://\S+\s*){3,}'),
#     re.compile(r'([a-z0-9_.+-]+@[a-z0-9-]+\.[a-z]+\s*){3,}', re.I),
#     re.compile(r'(©|\bAll rights reserved\b|\bPrivacy Policy\b)', re.I),
#     re.compile(r'\b(Home\s*[|>]\s*About|Skip to content|Cookie Policy)\b', re.I),
#     re.compile(r'\b(Follow us on|Share this page|Subscribe to our)\b', re.I),
# ]

# _STAT_MARKERS = re.compile(
#     r'\b(\d[\d,]+\s*(students?|faculty|staff|crore|lakh|%|placed|recruited|alumni|departments?))\b',
#     re.I,
# )

# def is_noise_chunk(text: str) -> bool:
#     """Return True if chunk is predominantly navigation / contact / footer boilerplate."""
#     if len(text.strip()) < 80:
#         return True
#     if _STAT_MARKERS.search(text):
#         return False
#     words       = text.split()
#     word_count  = max(len(words), 1)
#     url_count   = len(re.findall(r'https?://\S+', text))
#     email_count = len(re.findall(r'[a-z0-9_.+-]+@[a-z0-9-]+\.[a-z]{2,}', text, re.I))
#     if (url_count + email_count) / word_count > 0.30:
#         return True
#     return any(p.search(text) for p in _NOISE_PATTERNS)


# # ==========================================================================================
# # DATA HASH
# # ==========================================================================================

# def compute_data_hash() -> str:
#     h = hashlib.sha256()
#     for folder in [
#         CFG.RAG_JSON_FOLDER, CFG.CLEANED_TEXT_FOLDER,
#         CFG.OCR_TEXT_FOLDER, CFG.PDF_FOLDER,
#     ]:
#         p = Path(folder)
#         if not p.exists():
#             continue
#         for f in sorted(p.iterdir()):
#             h.update(f.name.encode())
#             h.update(str(f.stat().st_size).encode())
#     return h.hexdigest()


# def data_changed() -> bool:
#     if not os.path.exists(CFG.HASH_FILE):
#         return True
#     return Path(CFG.HASH_FILE).read_text().strip() != compute_data_hash()


# def save_data_hash() -> None:
#     Path(CFG.HASH_FILE).write_text(compute_data_hash())


# def can_fast_load() -> bool:
#     return (
#         os.path.exists(CFG.FAISS_PATH)
#         and os.path.exists(CFG.CHUNKS_PKL)
#         and os.path.exists(CFG.BM25_PKL)
#         and not data_changed()
#     )


# # ==========================================================================================
# # SECTION EXTRACTOR
# # ==========================================================================================

# def extract_section_from_source(source: str) -> str:
#     if not source:
#         return ""
#     s = source.lower()
#     if s.endswith(".pdf") or "/pdfs/" in s:
#         return "pdf"
#     if "/ocr_text/" in s or "/ocr/" in s:
#         return "ocr"
#     if "/cleaned_text/" in s or "/text/" in s:
#         return "cleaned_text"
#     try:
#         from urllib.parse import urlparse
#         parsed = urlparse(source)
#         parts  = [p for p in parsed.path.split("/") if p]
#         if parts:
#             return parts[0].lower()
#     except Exception:
#         pass
#     return ""


# # ==========================================================================================
# # STRUCTURED FACTS LAYER
# # ==========================================================================================

# _DEFAULT_FACTS = {
#     "fees": {
#         "description": "Fee information must be confirmed with the college office as it changes yearly.",
#         "contact": "accounts@loyolacollege.edu or +91-44-2817-2624",
#     },
#     "contact": {
#         "address": "Loyola College (Autonomous), Nungambakkam, Chennai - 600 034, Tamil Nadu, India",
#         "phone": "+91-44-2817-2624 / 2817-3274",
#         "email": "principal@loyolacollege.edu",
#         "website": "www.loyolacollege.edu",
#     },
#     "established": "1925",
#     "affiliation": "University of Madras (Autonomous status granted 1978)",
#     "accreditation": "NAAC A++ (reaccredited 2022)",
#     "departments": [
#         "Commerce", "Economics", "English", "Tamil", "Hindi", "French",
#         "Physics", "Chemistry", "Mathematics", "Statistics", "Computer Science",
#         "Biochemistry", "Microbiology", "Plant Biology and Biotechnology",
#         "Advanced Zoology and Biotechnology", "Visual Communication",
#         "Social Work", "History", "Philosophy", "Psychology",
#         "Business Administration (BBA)", "Computer Applications (BCA)",
#     ],
#     "programmes": {
#         "UG Programmes": [
#             "B.A. English Literature", "B.A. Tamil", "B.A. Hindi", "B.A. French",
#             "B.A. History", "B.A. Philosophy", "B.A. Economics",
#             "B.Com. (General)", "B.Com. (Professional Accounting)",
#             "B.Sc. Physics", "B.Sc. Chemistry", "B.Sc. Mathematics",
#             "B.Sc. Statistics", "B.Sc. Computer Science", "B.Sc. Biochemistry",
#             "B.Sc. Microbiology", "B.Sc. Plant Biology & Biotechnology",
#             "B.Sc. Advanced Zoology & Biotechnology",
#             "B.Sc. Visual Communication",
#             "B.B.A. Business Administration", "B.C.A. Computer Applications",
#             "B.S.W. Social Work",
#         ],
#         "PG Programmes": [
#             "M.A. English", "M.A. Tamil", "M.A. History", "M.A. Philosophy",
#             "M.A. Economics", "M.Com.", "M.Sc. Physics", "M.Sc. Chemistry",
#             "M.Sc. Mathematics", "M.Sc. Statistics", "M.Sc. Computer Science",
#             "M.Sc. Biochemistry", "M.Sc. Microbiology",
#             "M.Sc. Plant Biology & Biotechnology",
#             "M.Sc. Advanced Zoology & Biotechnology",
#             "M.S.W. Social Work", "M.C.A.", "M.B.A.",
#         ],
#         "M.Phil. / Ph.D.": [
#             "Available in most UG/PG departments. Contact the Research Cell for details.",
#         ],
#     },
# }


# @st.cache_resource(show_spinner=False)
# def load_facts() -> dict:
#     p = Path(CFG.FACTS_FILE)
#     if p.exists():
#         try:
#             data   = json.loads(p.read_text(encoding="utf-8"))
#             merged = {**_DEFAULT_FACTS, **data}
#             return merged
#         except Exception:
#             pass
#     return _DEFAULT_FACTS


# def query_facts_layer(question: str, facts: dict) -> Optional[str]:
#     q = question.lower()

#     if any(k in q for k in ["address", "location", "where is loyola", "phone", "email", "website", "contact"]):
#         c = facts.get("contact", {})
#         return (
#             f"**Loyola College Contact Details:**\n"
#             f"- Address: {c.get('address', 'N/A')}\n"
#             f"- Phone: {c.get('phone', 'N/A')}\n"
#             f"- Email: {c.get('email', 'N/A')}\n"
#             f"- Website: {c.get('website', 'N/A')}"
#         )

#     if any(k in q for k in ["naac", "accreditat", "ranking", "grade", "autonomous"]):
#         return (
#             f"Loyola College holds **{facts.get('accreditation', 'N/A')}** accreditation. "
#             f"It received autonomous status in {facts.get('affiliation', 'N/A')}."
#         )

#     if any(k in q for k in ["established", "founded", "year of", "when was loyola"]):
#         return f"Loyola College was established in **{facts.get('established', 'N/A')}**."

#     if any(k in q for k in [
#         "course", "programme", "program", "ug ", "pg ", "degree",
#         "bsc", "bcom", "ba ", "msc", "mcom", "ma ", "mca", "mba", "phd",
#         "undergraduate", "postgraduate", "what can i study", "what do you offer",
#     ]):
#         programmes = facts.get("programmes", {})
#         if programmes:
#             lines = ["**Programmes offered at Loyola College:**\n"]
#             for level, prog_list in programmes.items():
#                 lines.append(f"\n**{level}**")
#                 for prog in prog_list:
#                     lines.append(f"- {prog}")
#             return "\n".join(lines)
#         depts = facts.get("departments", [])
#         if depts:
#             return (
#                 "**Departments at Loyola College:**\n"
#                 + "\n".join(f"- {d}" for d in depts)
#             )

#     if any(k in q for k in ["department", "stream", "what subjects", "which departments"]):
#         depts = facts.get("departments", [])
#         if depts:
#             return (
#                 "**Departments at Loyola College:**\n"
#                 + "\n".join(f"- {d}" for d in depts)
#             )

#     if any(k in q for k in ["fee", "fees", "tuition", "cost", "how much"]):
#         c = facts.get("fees", {})
#         return (
#             f"Fee structures at Loyola College change annually. "
#             f"{c.get('description', '')} "
#             f"Contact: {c.get('contact', 'the college accounts office')}."
#         )

#     return None


# # ==========================================================================================
# # QUERY EXPANSION  — NEW v9
# # ==========================================================================================
# # Injects synonyms and Loyola-specific terminology into the search query to
# # maximise lexical recall in BM25 and improve dense retrieval coverage.

# _EXPANSION_MAP = {
#     # Placement / career
#     r"\bplacement\b":         ["recruitment", "campus hiring", "job offers", "career"],
#     r"\bsalary\b":            ["package", "CTC", "compensation", "pay"],
#     r"\brecruiter\b":         ["company", "recruiter", "employer", "hiring firm"],

#     # Academics
#     r"\bcourse\b":            ["programme", "degree", "subject", "curriculum"],
#     r"\badmission\b":         ["enrolment", "joining", "application", "eligibility"],
#     r"\beligibilit\b":        ["criteria", "requirement", "cutoff", "qualification"],
#     r"\bfee\b":               ["tuition", "cost", "charges", "payment"],
#     r"\bscholarship\b":       ["financial aid", "bursary", "stipend", "grant"],

#     # Infrastructure
#     r"\bhostel\b":            ["accommodation", "residence", "dormitory", "boarding"],
#     r"\blibrary\b":           ["books", "journals", "reading room", "digital library"],
#     r"\blab\b":               ["laboratory", "research facility", "equipment"],

#     # Rankings / quality
#     r"\branking\b":           ["nirf", "naac", "grade", "position", "accreditation"],
#     r"\bfacult\b":            ["professor", "teacher", "staff", "lecturer"],

#     # Loyola-specific shorthand
#     r"\bloc\b":               ["loyola college", "loyola"],
#     r"\bloyola\b":            ["loyola college chennai", "loyola autonomous"],
# }

# # College-specific prefix always appended for context anchoring
# _COLLEGE_PREFIX = "Loyola College Chennai"


# def expand_query(query: str) -> str:
#     """
#     Return an expanded version of the query with synonyms and college context.
#     The original query is preserved; expansions are appended as additional terms.
#     """
#     extra_terms: list[str] = []

#     for pattern, synonyms in _EXPANSION_MAP.items():
#         if re.search(pattern, query, re.I):
#             extra_terms.extend(synonyms)

#     # Deduplicate while preserving order
#     seen: set[str] = set()
#     unique_extras: list[str] = []
#     for t in extra_terms:
#         tl = t.lower()
#         if tl not in seen:
#             seen.add(tl)
#             unique_extras.append(t)

#     if unique_extras:
#         expansion = " ".join(unique_extras[:6])   # cap at 6 extra terms
#         return f"{_COLLEGE_PREFIX} {query} {expansion}"
#     return f"{_COLLEGE_PREFIX} {query}"


# # ==========================================================================================
# # DOCUMENT LOADERS
# # ==========================================================================================

# def load_rag_json_folder(log) -> list[Document]:
#     folder = Path(CFG.RAG_JSON_FOLDER)
#     docs   = []
#     if not folder.exists():
#         return docs
#     files = sorted(folder.glob("*.json"))
#     log.write(f"📄 RAG JSON: {len(files)} files")
#     for fpath in files:
#         try:
#             raw  = json.loads(fpath.read_text(encoding="utf-8"))
#             text = raw.get("page_content", "").strip()
#             meta = raw.get("metadata", {})
#             if len(text) < 100:
#                 continue
#             source  = meta.get("url") or meta.get("source") or str(fpath)
#             section = meta.get("section") or extract_section_from_source(source)
#             docs.append(Document(
#                 page_content = clean_text(text),
#                 metadata = {
#                     "source" : source,
#                     "title"  : meta.get("title", ""),
#                     "type"   : "website",
#                     "section": section,
#                 },
#             ))
#         except Exception as e:
#             print(f"[JSON ERROR] {fpath.name}: {e}")
#     log.success(f"✅ RAG JSON: {len(docs)} pages loaded")
#     return docs


# def load_text_folder(folder_path: str, label: str, log) -> list[Document]:
#     folder = Path(folder_path)
#     docs   = []
#     if not folder.exists():
#         return docs
#     files = sorted(folder.glob("*.txt"))
#     log.write(f"📝 {label}: {len(files)} files")
#     for fpath in files:
#         try:
#             text = fpath.read_text(encoding="utf-8", errors="replace").strip()
#             if len(text) < 100:
#                 continue
#             source  = str(fpath)
#             section = extract_section_from_source(source)
#             docs.append(Document(
#                 page_content = clean_text(text),
#                 metadata = {
#                     "source" : source,
#                     "title"  : fpath.stem,
#                     "type"   : label.lower(),
#                     "section": section,
#                 },
#             ))
#         except Exception as e:
#             print(f"[TEXT ERROR] {fpath.name}: {e}")
#     log.success(f"✅ {label}: {len(docs)} loaded")
#     return docs


# def load_pdf(path: str) -> list[Document]:
#     for Loader in [PyMuPDFLoader, PDFPlumberLoader]:
#         try:
#             raw  = Loader(path).load()
#             docs = [
#                 Document(
#                     page_content = clean_text(d.page_content),
#                     metadata = {
#                         "source" : path,
#                         "page"   : d.metadata.get("page", "N/A"),
#                         "type"   : "pdf",
#                         "title"  : Path(path).name,
#                         "section": "pdf",
#                     },
#                 )
#                 for d in raw if len(d.page_content.strip()) > 50
#             ]
#             if docs:
#                 return docs
#         except Exception:
#             continue
#     return []


# def load_pdf_folder(log) -> list[Document]:
#     folder = Path(CFG.PDF_FOLDER)
#     docs   = []
#     if not folder.exists():
#         return docs
#     files = sorted(folder.glob("*.pdf"))
#     log.write(f"📚 PDFs: {len(files)} files")
#     bar = log.progress(0)
#     for i, fpath in enumerate(files):
#         try:
#             docs.extend(load_pdf(str(fpath)))
#             bar.progress((i + 1) / max(len(files), 1))
#         except Exception as e:
#             print(f"[PDF ERROR] {fpath.name}: {e}")
#     log.success(f"✅ PDFs: {len(docs)} pages loaded")
#     return docs


# def load_all_documents(sidebar_log) -> list[Document]:
#     all_docs  = []
#     seen_keys = set()

#     def _add(docs):
#         for d in docs:
#             key = d.page_content[:300]
#             if key not in seen_keys:
#                 seen_keys.add(key)
#                 all_docs.append(d)

#     s1 = sidebar_log.empty()
#     s2 = sidebar_log.empty()
#     s3 = sidebar_log.empty()
#     s4 = sidebar_log.empty()

#     _add(load_rag_json_folder(s1))
#     _add(load_text_folder(CFG.CLEANED_TEXT_FOLDER, "Cleaned Text", s2))
#     _add(load_text_folder(CFG.OCR_TEXT_FOLDER,     "OCR Text",     s3))
#     _add(load_pdf_folder(s4))

#     sidebar_log.success(f"🗂 Total unique docs: {len(all_docs)}")
#     return all_docs


# # ==========================================================================================
# # CHUNKING
# # ==========================================================================================

# def create_chunks(docs: list[Document]) -> list[Document]:
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size    = CFG.CHUNK_SIZE,
#         chunk_overlap = CFG.CHUNK_OVERLAP,
#         separators    = ["\n\n\n", "\n\n", "\n", ". ", " ", ""],
#     )
#     chunks = splitter.split_documents(docs)
#     return [
#         c for c in chunks
#         if len(c.page_content.strip()) > 80 and not is_noise_chunk(c.page_content)
#     ]


# # ==========================================================================================
# # MODELS
# # ==========================================================================================

# @st.cache_resource(show_spinner=False)
# def load_embedding_model():
#     return HuggingFaceEmbeddings(
#         model_name    = CFG.EMBEDDING_MODEL,
#         model_kwargs  = {"device": DEVICE},
#         encode_kwargs = {"normalize_embeddings": True},
#     )


# @st.cache_resource(show_spinner=False)
# def load_reranker():
#     return CrossEncoder(CFG.RERANK_MODEL, device=DEVICE)


# @st.cache_resource(show_spinner=False)
# def load_sentence_model():
#     return SentenceTransformer(CFG.EMBEDDING_MODEL, device=DEVICE)


# # ==========================================================================================
# # KNOWLEDGE BASE
# # ==========================================================================================

# def build_knowledge_base(sidebar_log, emb):

#     if can_fast_load():
#         sidebar_log.info("⚡ Fast loading from saved index...")
#         t0     = time.time()
#         chunks = pickle.load(open(CFG.CHUNKS_PKL, "rb"))
#         bm25   = pickle.load(open(CFG.BM25_PKL,   "rb"))
#         db     = FAISS.load_local(
#             CFG.FAISS_PATH, emb,
#             allow_dangerous_deserialization=True,
#         )
#         elapsed = round(time.time() - t0, 1)
#         sidebar_log.success(f"✅ {len(chunks):,} chunks loaded in {elapsed}s")
#         return db, chunks, bm25

#     sidebar_log.warning("🔄 Building knowledge base — please wait...")
#     t0   = time.time()
#     docs = load_all_documents(sidebar_log)

#     if not docs:
#         st.error("❌ No documents found. Check your data/ folder.")
#         st.stop()

#     sidebar_log.write(f"✂️ Chunking {len(docs):,} documents...")
#     chunks = create_chunks(docs)
#     sidebar_log.write(f"📦 {len(chunks):,} chunks created")
#     sidebar_log.write("🧠 Building FAISS + BM25 indices...")

#     for f in ["index.faiss", "index.pkl"]:
#         fp = Path(CFG.FAISS_PATH) / f
#         if fp.exists():
#             fp.unlink()

#     db   = FAISS.from_documents(chunks, emb)
#     db.save_local(CFG.FAISS_PATH)

#     bm25   = BM25Retriever.from_documents(chunks)
#     bm25.k = CFG.RETRIEVAL_K_COMPARE

#     pickle.dump(chunks, open(CFG.CHUNKS_PKL, "wb"))
#     pickle.dump(bm25,   open(CFG.BM25_PKL,   "wb"))
#     save_data_hash()

#     elapsed = round(time.time() - t0, 1)
#     sidebar_log.success(f"✅ Built {len(chunks):,} chunks in {elapsed}s")
#     sidebar_log.info("⚡ Next run will load in ~5 seconds")
#     return db, chunks, bm25


# # ==========================================================================================
# # QUERY CLASSIFICATION  — v9: 8 classes
# # ==========================================================================================

# _FEE_RE         = re.compile(r"\bfee|tuition|cost|how much|payment\b", re.I)
# _ELIGIBILITY_RE = re.compile(r"\beligib|qualify|cutoff|criteria|requirement|admission\b", re.I)
# _DATE_RE        = re.compile(r"\bdate|deadline|when|schedule|calendar|exam date\b", re.I)
# _FACT_RE        = re.compile(r"\bwho is|what is|where is|located|principal|founded|established\b", re.I)
# _FOLLOWUP_RE    = re.compile(r"^(what about|how about|and|also|tell me more|explain|elaborate|go on|continue)\b", re.I)
# _COMPARISON_RE  = re.compile(
#     r"\b(compare|versus|vs\.?|difference between|better|worse|which is|pros and cons|"
#     r"advantages|disadvantages|contrast|similarities)\b",
#     re.I,
# )

# _STATS_RE = re.compile(
#     r"\b("
#     r"how many students?|total students?|student strength|student count|student enrolment|"
#     r"placement stat|placement record|placed students?|salary package|average package|"
#     r"pass percentage|pass rate|result percentage|"
#     r"faculty count|number of (faculty|staff|teachers?)|"
#     r"college ranking|nirf rank|number of departments?|programmes? offered|"
#     r"total intake|sanctioned strength|hostel capacity"
#     r")\b",
#     re.I,
# )


# def classify_query(query: str, has_history: bool) -> str:
#     """
#     Classify query into one of 8 categories.
#     Order matters — more specific classes checked first.
#     """
#     q = query.strip()

#     if has_history and (_FOLLOWUP_RE.search(q) or len(q.split()) <= 5):
#         return "followup"

#     if _COMPARISON_RE.search(q):
#         return "comparison"

#     if _STATS_RE.search(q):
#         return "statistics"

#     if _FEE_RE.search(q):
#         return "fee"

#     if _ELIGIBILITY_RE.search(q):
#         return "eligibility"

#     if _DATE_RE.search(q):
#         return "date"

#     if _FACT_RE.search(q):
#         return "fact"

#     return "general"


# def get_retrieval_k(q_class: str) -> int:
#     """Dynamically scale retrieval K based on query complexity."""
#     mapping = {
#         "fact"       : CFG.RETRIEVAL_K_FACT,
#         "fee"        : CFG.RETRIEVAL_K_FACT,
#         "followup"   : CFG.RETRIEVAL_K_DEFAULT,
#         "general"    : CFG.RETRIEVAL_K_DEFAULT,
#         "statistics" : CFG.RETRIEVAL_K_DEEP,
#         "eligibility": CFG.RETRIEVAL_K_DEEP,
#         "date"       : CFG.RETRIEVAL_K_DEEP,
#         "comparison" : CFG.RETRIEVAL_K_COMPARE,
#     }
#     return mapping.get(q_class, CFG.RETRIEVAL_K_DEFAULT)


# # ==========================================================================================
# # HISTORY-AWARE QUERY REWRITING  — v9: all ambiguous queries rewritten
# # ==========================================================================================

# def rewrite_query(query: str, history: list[dict], llm, q_class: str) -> str:
#     """
#     Rewrite the query into a clear, standalone, searchable form.
#     For follow-up queries: resolve pronouns / references using history.
#     For all other classes: expand abbreviations and add college context.
#     """
#     # ── Follow-up: resolve pronouns and elliptical references ─────────
#     if q_class == "followup" and history:
#         recent       = history[-2:]
#         history_text = "".join(
#             f"Student: {h['q']}\nAssistant: {h['a'][:200]}...\n\n"
#             for h in recent
#         )
#         prompt = (
#             "You are a query rewriter for a college information chatbot.\n"
#             "Rewrite the follow-up question as a complete, standalone question.\n\n"
#             "Rules:\n"
#             "- Output ONLY the rewritten question. Nothing else.\n"
#             "- Keep it under 25 words.\n"
#             "- If already standalone, return unchanged.\n\n"
#             f"Conversation history:\n{history_text}"
#             f"Follow-up question: {query}\n\n"
#             "Standalone question:"
#         )
#         try:
#             result = ""
#             for chunk in llm.stream(prompt):
#                 result += chunk.content
#             rewritten = result.strip().strip('"').strip("'")
#             if 5 < len(rewritten) < 200:
#                 return rewritten
#         except Exception:
#             pass
#         return query

#     # ── For short / ambiguous non-follow-up queries: expand silently ──
#     if len(query.split()) <= 3:
#         return f"Loyola College {query} information"

#     return query


# # ==========================================================================================
# # RRF HYBRID RETRIEVER
# # ==========================================================================================

# class HybridRRFRetriever:
#     """FAISS (dense) + BM25 (sparse) fused with Reciprocal Rank Fusion."""

#     def __init__(self, db, bm25: BM25Retriever):
#         self._db   = db
#         self._bm25 = bm25

#     def _dense_retrieve(self, query: str, k: int) -> list[Document]:
#         retriever = self._db.as_retriever(
#             search_type   = "mmr",
#             search_kwargs = {"k": k, "fetch_k": k * 2},
#         )
#         return retriever.invoke(query)

#     def _sparse_retrieve(self, query: str, k: int) -> list[Document]:
#         self._bm25.k = k
#         return self._bm25.invoke(query)

#     def invoke(
#         self,
#         query      : str,
#         k          : int       = CFG.RETRIEVAL_K_DEFAULT,
#         type_filter: list[str] = None,
#     ) -> list[Document]:
#         dense_docs  = self._dense_retrieve(query, k)
#         sparse_docs = self._sparse_retrieve(query, k)

#         doc_map    : dict[str, Document] = {}
#         rrf_scores : dict[str, float]   = {}

#         def _key(d: Document) -> str:
#             return d.page_content[:200]

#         for rank, doc in enumerate(dense_docs):
#             key = _key(doc)
#             doc_map[key]    = doc
#             rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (CFG.RRF_K + rank + 1)

#         for rank, doc in enumerate(sparse_docs):
#             key = _key(doc)
#             doc_map[key]    = doc
#             rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (CFG.RRF_K + rank + 1)

#         sorted_keys = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
#         results     = [doc_map[kk] for kk in sorted_keys]

#         if type_filter:
#             results = [
#                 d for d in results
#                 if d.metadata.get("type", "").lower() in [t.lower() for t in type_filter]
#             ]

#         return results[:k]


# # ==========================================================================================
# # SOURCE PRIORITY BOOST
# # ==========================================================================================

# _PRIORITY_PATTERNS = [
#     re.compile(r'placement|annual.?report|brochure|prospectus', re.I),
#     re.compile(r'naac|nirf|iqac|accreditat|handbook|admission.?guide', re.I),
#     re.compile(r'statistical|census|highlights|achievements|ranking', re.I),
# ]


# def _source_priority_score(doc: Document) -> float:
#     combined = (
#         doc.metadata.get("source", "") + " " +
#         doc.metadata.get("title",  "")
#     ).lower()
#     for p in _PRIORITY_PATTERNS:
#         if p.search(combined):
#             return 0.30
#     if doc.metadata.get("type") == "pdf":
#         return 0.15
#     return 0.0


# # ==========================================================================================
# # RERANKER  — v9: larger candidate pool + adaptive threshold
# # ==========================================================================================

# def _adaptive_rerank_threshold(q_class: str, base: float = CFG.RERANK_THRESHOLD) -> float:
#     """
#     Lower the threshold for complex queries so more potentially-relevant
#     chunks survive; raise it for simple facts to keep context tight.
#     """
#     adjustments = {
#         "fact"       :  0.05,   # tighter — only high-confidence chunks
#         "fee"        :  0.05,
#         "comparison" : -0.05,   # looser — need breadth
#         "statistics" : -0.05,
#         "eligibility": -0.03,
#     }
#     return base + adjustments.get(q_class, 0.0)


# def rerank(
#     query  : str,
#     docs   : list[Document],
#     model,
#     q_class: str  = "general",
#     debug  : bool = False,
# ) -> list[tuple[float, Document]]:
#     """
#     Score = CrossEncoder(query, chunk) + source_priority_boost.
#     Returns (boosted_score, Document) sorted descending, capped at RERANK_K.
#     """
#     if not docs:
#         return []

#     # Larger candidate pool in v9
#     candidates = docs[:CFG.RERANK_CANDIDATE_POOL]
#     pairs      = [[query, d.page_content] for d in candidates]
#     raw_scores = model.predict(pairs, batch_size=16)

#     boosted = [
#         float(s) + _source_priority_score(d)
#         for s, d in zip(raw_scores, candidates)
#     ]

#     ranked    = sorted(zip(boosted, candidates), key=lambda x: x[0], reverse=True)
#     threshold = _adaptive_rerank_threshold(q_class)
#     filtered  = [(s, d) for s, d in ranked if s > threshold]

#     if not filtered:
#         filtered = ranked[:2]   # always return at least 2

#     if debug:
#         print(f"\n[RERANK DEBUG] class={q_class} threshold={threshold:.3f} Top-10:")
#         for i, (s, d) in enumerate(ranked[:10]):
#             src     = d.metadata.get("source", "?")[:70]
#             snippet = d.page_content[:100].strip().replace("\n", " ")
#             print(f"  [{i+1}] score={s:.4f}  src={src}")
#             print(f"       {snippet!r}")

#     return filtered[:CFG.RERANK_K]


# # ==========================================================================================
# # SEMANTIC CACHE  — v9: embedding storage, no recompute, LRU eviction
# # ==========================================================================================

# def semantic_cache_lookup(
#     query    : str,
#     cache    : "collections.OrderedDict",
#     sent_model,
#     threshold: float = CFG.SEMANTIC_CACHE_THRESHOLD,
# ) -> Optional[dict]:
#     """
#     v9 improvements:
#     - Each cache entry stores its own embedding under key '__emb__'.
#     - We batch-compare query against stored embeddings (no re-encoding cached queries).
#     - Uses efficient dot-product cosine (vectors pre-normalised).
#     """
#     if not cache:
#         return None
#     try:
#         q_emb = sent_model.encode(query, normalize_embeddings=True)

#         # Collect keys and their pre-stored embeddings
#         keys_with_emb = [
#             (k, v["__emb__"])
#             for k, v in cache.items()
#             if isinstance(v, dict) and "__emb__" in v
#         ]

#         if not keys_with_emb:
#             # Fallback for old-format entries (no __emb__): re-encode and upgrade
#             all_keys  = list(cache.keys())
#             key_embs  = sent_model.encode(all_keys, normalize_embeddings=True, batch_size=64)
#             sims      = key_embs @ q_emb
#             best_idx  = int(np.argmax(sims))
#             if sims[best_idx] >= threshold:
#                 hit = cache[all_keys[best_idx]]
#                 return hit if isinstance(hit, dict) else None
#             return None

#         cached_keys = [k for k, _ in keys_with_emb]
#         emb_matrix  = np.array([e for _, e in keys_with_emb])
#         sims        = emb_matrix @ q_emb
#         best_idx    = int(np.argmax(sims))

#         if sims[best_idx] >= threshold:
#             hit = cache[cached_keys[best_idx]]
#             cache.move_to_end(cached_keys[best_idx])   # LRU touch
#             # Return a copy without the internal embedding key
#             return {k: v for k, v in hit.items() if k != "__emb__"}

#     except Exception:
#         pass
#     return None


# def semantic_cache_store(
#     cache    : "collections.OrderedDict",
#     key      : str,
#     result   : dict,
#     sent_model,
# ) -> None:
#     """Store result with its embedding for O(1) similarity lookups later."""
#     try:
#         emb            = sent_model.encode(key, normalize_embeddings=True)
#         entry          = dict(result)
#         entry["__emb__"] = emb
#         _cache_set(cache, key, entry)
#     except Exception:
#         _cache_set(cache, key, result)


# # ==========================================================================================
# # HALLUCINATION GROUNDING  — v9: compare against ALL chunks, not first 512 chars
# # ==========================================================================================

# def compute_grounding_score(
#     answer     : str,
#     scored_docs: list[tuple[float, Document]],
#     sent_model,
# ) -> float:
#     """
#     v9: Encode the answer and ALL retrieved chunk texts.
#     Return the MAXIMUM cosine similarity across chunks (most-grounded chunk wins).
#     This is more reliable than comparing against a truncated context string.
#     """
#     if not scored_docs:
#         return 1.0
#     try:
#         chunk_texts = [d.page_content[:512] for _, d in scored_docs]
#         all_texts   = [answer[:512]] + chunk_texts
#         # Batch encode in one call — no repeated computations
#         embs = sent_model.encode(all_texts, normalize_embeddings=True, batch_size=32)
#         answer_emb  = embs[0]
#         chunk_embs  = embs[1:]
#         sims        = chunk_embs @ answer_emb
#         return float(np.max(sims))
#     except Exception:
#         return 1.0


# def confidence_label(score: float) -> tuple[str, str]:
#     if score >= CFG.CONFIDENCE_HIGH:
#         return "High", "🟢"
#     if score >= CFG.CONFIDENCE_MED:
#         return "Medium", "🟡"
#     return "Low", "🔴"


# # ==========================================================================================
# # FOLLOW-UP SUGGESTIONS  — v9: strictly grounded, no hallucinations
# # ==========================================================================================

# def generate_followups(
#     query   : str,
#     answer  : str,
#     llm,
#     context : str = "",
# ) -> list[str]:
#     """
#     Generate 3 follow-up questions grounded in the retrieved context.
#     Returns an empty list when the answer is the fallback phrase or on any error.
#     """
#     if "could not find this information" in answer.lower():
#         return []

#     # Use a richer context excerpt in v9 (500 chars instead of 300)
#     context_snippet = context[:500].strip() if context else ""
#     grounding_note  = (
#         f"\nOnly suggest questions directly answerable from this context excerpt:\n"
#         f'"""{context_snippet}"""'
#         if context_snippet else ""
#     )

#     prompt = (
#         "You are a helpful assistant for Loyola College, Chennai.\n"
#         "Suggest exactly 3 short follow-up questions a student might ask, "
#         "based ONLY on the answer and context below.\n"
#         "Each question MUST be answerable from the same topic/document as the answer.\n"
#         "Do NOT suggest questions about unrelated topics or invent new facts."
#         f"{grounding_note}\n\n"
#         f"Question: {query}\n"
#         f"Answer: {answer[:400]}\n\n"
#         "Output ONLY a JSON array of 3 question strings. No explanation. No markdown.\n"
#         'Example: ["Question one?", "Question two?", "Question three?"]\n\n'
#         "JSON array:"
#     )

#     try:
#         result = ""
#         for chunk in llm.stream(prompt):
#             result += chunk.content
#         match = re.search(r"\[.*?\]", result, re.DOTALL)
#         if match:
#             suggestions = json.loads(match.group())
#             if isinstance(suggestions, list):
#                 cleaned = [s.strip() for s in suggestions if isinstance(s, str) and len(s) > 5]
#                 return cleaned[:3]
#     except Exception:
#         pass
#     return []


# # ==========================================================================================
# # CONTEXT BUILDER  — v9: dedup, diversity filter, relevance-only
# # ==========================================================================================

# def _chunk_fingerprint(text: str) -> str:
#     """Short fingerprint for near-duplicate detection."""
#     words = re.sub(r"\s+", " ", text.lower()).split()
#     return " ".join(words[:30])


# def build_context(
#     scored_docs: list[tuple[float, Document]],
# ) -> tuple[str, list[str], list[dict]]:
#     """
#     v9 improvements:
#     - Skip near-duplicate chunks (fingerprint-based dedup).
#     - Diversity filter: avoid piling up chunks from the same source.
#     - Only include chunks that contribute unique information.

#     Returns:
#         context_str   — text block for the LLM prompt
#         sources       — list[str] of raw source paths/URLs
#         rich_sources  — list[dict] with full attribution data for UI
#     """
#     parts, sources, rich_sources = [], [], []
#     tokens        = 0
#     seen_fps      = set()           # fingerprints for dedup
#     source_counts : dict[str, int] = {}   # diversity cap per source
#     MAX_PER_SOURCE = 2              # avoid > 2 chunks from same source

#     for score, d in scored_docs:
#         # Fingerprint-based near-duplicate check
#         fp = _chunk_fingerprint(d.page_content)
#         if fp in seen_fps:
#             continue
#         seen_fps.add(fp)

#         # Diversity: cap contributions from the same source
#         src_key = d.metadata.get("source", "")
#         if source_counts.get(src_key, 0) >= MAX_PER_SOURCE:
#             continue
#         source_counts[src_key] = source_counts.get(src_key, 0) + 1

#         txt = d.page_content[:2500]
#         t   = count_tokens(txt)
#         if tokens + t > CFG.MAX_CONTEXT_TOKENS:
#             break

#         source  = d.metadata.get("source", "")
#         title   = d.metadata.get("title", "")
#         page    = d.metadata.get("page", "")
#         dtype   = d.metadata.get("type", "")
#         section = d.metadata.get("section", "")

#         header = f"[SOURCE: {title or source}"
#         if page and str(page) != "N/A":
#             header += f" | Page {page}"
#         header += "]"

#         parts.append(f"{header}\n{txt}")
#         tokens += t

#         if source and source not in sources:
#             sources.append(source)
#             rich_sources.append({
#                 "source"       : source,
#                 "title"        : title or source,
#                 "page"         : str(page) if page and str(page) != "N/A" else "",
#                 "type"         : dtype,
#                 "section"      : section,
#                 "rerank_score" : round(float(score), 3),
#             })

#     return "\n\n---\n\n".join(parts), sources, rich_sources


# # ==========================================================================================
# # LLM
# # ==========================================================================================

# def get_llm(api_key: str):
#     return ChatGroq(
#         groq_api_key = api_key,
#         model_name   = CFG.LLM_MODEL,
#         temperature  = 0.1,
#         streaming    = True,
#     )


# # ==========================================================================================
# # PROMPT  — v9: stronger hallucination prevention
# # ==========================================================================================

# def build_prompt(context: str, question: str, history: list[dict], q_class: str) -> str:
#     history_text = ""
#     for h in history[-CFG.HISTORY_TURNS:]:
#         history_text += f"Student: {h['q']}\nAssistant: {h['a']}\n\n"

#     class_hints = {
#         "statistics" : (
#             "Look for college-wide totals, not department-specific sub-numbers. "
#             "If only department figures are present, state that a college-wide total was not found. "
#             "Always include the year or source of the figure."
#         ),
#         "fee"        : "Report exact fee amounts and payment schedules only if present in context. Never estimate.",
#         "eligibility": "State admission criteria, cutoff marks, and required documents exactly as written.",
#         "date"       : "State exact dates and deadlines only. Do not estimate or approximate.",
#         "fact"       : "Give a single direct factual answer in one or two sentences.",
#         "comparison" : (
#             "Use a markdown table to compare attributes across items. "
#             "Only compare attributes explicitly mentioned in the context."
#         ),
#         "followup"   : "Continue consistently with the conversation history above.",
#         "general"    : "Give a complete, accurate answer using only what is in the context.",
#     }
#     hint = class_hints.get(q_class, class_hints["general"])

#     return f"""You are the official AI assistant for Loyola College (Autonomous), Chennai, Tamil Nadu.

# ABSOLUTE RULES — obey every rule without exception:
# 1. Answer ONLY using facts explicitly stated in the CONTEXT block below. Nothing else.
# 2. Do NOT use your own training knowledge, general world knowledge, or any external information.
# 3. Do NOT combine figures or facts from different, unrelated sections of the context.
# 4. Do NOT infer, estimate, approximate, or generate any fact not written word-for-word in the context.
# 5. Do NOT mention URLs, email addresses, or phone numbers unless the user explicitly asked for them.
# 6. If the answer is not clearly present in the context, output this sentence exactly and nothing else:
#    "I could not find this information in the knowledge base."
# 7. FORMAT RULES:
#    – Use bullet points (–) for lists of 3 or more items.
#    – Use a markdown table when comparing multiple attributes across multiple items.
#    – Use plain sentences for a single fact or a two-item answer.
# 8. Maximum 6 sentences or 10 bullet points. Be direct and specific.
# 9. Do NOT use meta-phrases like "Based on the context", "According to the document", "However", "Additionally".
# 10. Do NOT repeat, rephrase, or acknowledge the question.
# 11. Do NOT say "I don't know" or "I'm not sure" — use the exact fallback phrase from Rule 6 instead.
# 12. {hint}

# CONVERSATION HISTORY:
# {history_text}
# ====== CONTEXT — use ONLY this ======
# {context}
# =====================================

# Question: {question}

# Answer:"""


# # ==========================================================================================
# # MAIN ASK FUNCTION
# # ==========================================================================================

# def ask(
#     query      : str,
#     retriever  : HybridRRFRetriever,
#     reranker,
#     llm_mdl,
#     sent_model,
#     history    : list[dict],
#     cache      : "collections.OrderedDict",
#     facts      : dict,
#     type_filter: list[str] = None,
#     debug      : bool      = False,
# ) -> dict:
#     """
#     Full RAG pipeline. Returns:
#     {
#         answer, sources, rich_sources,
#         confidence, conf_emoji, score,
#         followups, cache_hit, q_class,
#         facts_hit, log_id,
#     }
#     """
#     t_start = time.time()

#     # ── Security ─────────────────────────────────────────────────────
#     if is_malicious(query):
#         return {
#             "answer": "⚠️ I can't process that request.", "sources": [],
#             "rich_sources": [], "confidence": "N/A", "conf_emoji": "🔴",
#             "score": 0.0, "followups": [], "cache_hit": False,
#             "q_class": "blocked", "facts_hit": False, "log_id": None,
#         }

#     q_class     = classify_query(query, has_history=bool(history))
#     retrieval_k = get_retrieval_k(q_class)

#     # ── Facts layer ───────────────────────────────────────────────────
#     fact_answer = query_facts_layer(query, facts)
#     if fact_answer:
#         lid = log_query(query, q_class, time.time() - t_start, False, "High", True, "facts")
#         return {
#             "answer"      : fact_answer,
#             "sources"     : ["Loyola College — Official Facts"],
#             "rich_sources": [{"source": "facts", "title": "Loyola College — Official Facts",
#                                "page": "", "type": "facts", "section": "facts", "rerank_score": 1.0}],
#             "confidence"  : "High",
#             "conf_emoji"  : "🟢",
#             "score"       : 1.0,
#             "followups"   : generate_followups(query, fact_answer, llm_mdl),
#             "cache_hit"   : False,
#             "q_class"     : q_class,
#             "facts_hit"   : True,
#             "log_id"      : lid,
#         }

#     # ── Exact cache ───────────────────────────────────────────────────
#     cache_key = query.lower().strip()
#     if cache_key in cache:
#         r   = cache[cache_key]
#         lid = log_query(query, q_class, time.time() - t_start, True, r.get("confidence", "?"), True)
#         cache.move_to_end(cache_key)
#         return {**{k: v for k, v in r.items() if k != "__emb__"}, "cache_hit": True, "log_id": lid}

#     # ── Semantic cache ────────────────────────────────────────────────
#     sem_hit = semantic_cache_lookup(query, cache, sent_model)
#     if sem_hit:
#         lid = log_query(query, q_class, time.time() - t_start, True, sem_hit.get("confidence", "?"), True)
#         return {**sem_hit, "cache_hit": True, "log_id": lid}

#     # ── Query rewriting ───────────────────────────────────────────────
#     effective_query = rewrite_query(query, history, llm_mdl, q_class)

#     # ── Query expansion — NEW v9 ─────────────────────────────────────
#     expanded_query = expand_query(effective_query)

#     # ── Multi-query RRF retrieval ─────────────────────────────────────
#     # Use expanded query for primary retrieval, original for secondary
#     retrieval_queries = [expanded_query, effective_query]
#     all_docs : dict[str, Document] = {}
#     rrf_agg  : dict[str, float]   = {}

#     for q in retrieval_queries:
#         batch = retriever.invoke(q, k=retrieval_k, type_filter=type_filter)
#         for rank, doc in enumerate(batch):
#             key = doc.page_content[:200]
#             all_docs[key] = doc
#             rrf_agg[key]  = rrf_agg.get(key, 0.0) + 1.0 / (CFG.RRF_K + rank + 1)

#     top_keys = sorted(rrf_agg, key=lambda x: rrf_agg[x], reverse=True)
#     merged   = [all_docs[k] for k in top_keys]

#     # ── Rerank with adaptive threshold ────────────────────────────────
#     scored_docs = rerank(effective_query, merged, reranker, q_class=q_class, debug=debug)

#     # ── Build context with dedup + diversity ──────────────────────────
#     context, sources, rich_sources = build_context(scored_docs)

#     if not context.strip():
#         answer = "I could not find this information in the knowledge base."
#         lid = log_query(query, q_class, time.time() - t_start, False, "Low", False)
#         return {
#             "answer": answer, "sources": [], "rich_sources": [],
#             "confidence": "Low", "conf_emoji": "🔴", "score": 0.0,
#             "followups": [], "cache_hit": False, "q_class": q_class,
#             "facts_hit": False, "log_id": lid,
#         }

#     # ── LLM generation ────────────────────────────────────────────────
#     prompt = build_prompt(context, effective_query, history, q_class)
#     answer = ""
#     for chunk in llm_mdl.stream(prompt):
#         answer += chunk.content
#     answer = answer.strip()

#     # ── Hallucination grounding — v9: all chunks, max similarity ─────
#     score = compute_grounding_score(answer, scored_docs, sent_model)
#     conf_lbl, conf_emoji = confidence_label(score)

#     if score < CFG.GROUNDING_THRESHOLD:
#         answer += (
#             "\n\n⚠️ *Low confidence: this answer may not be fully supported by "
#             "the available information. Please verify at www.loyolacollege.edu.*"
#         )

#     # ── Follow-ups — grounded in retrieved context ────────────────────
#     followups = generate_followups(query, answer, llm_mdl, context=context)

#     top_section = rich_sources[0]["section"] if rich_sources else ""
#     lid = log_query(query, q_class, time.time() - t_start, False, conf_lbl, True, top_section)

#     result = {
#         "answer"      : answer,
#         "sources"     : sources,
#         "rich_sources": rich_sources,
#         "confidence"  : conf_lbl,
#         "conf_emoji"  : conf_emoji,
#         "score"       : round(score, 3),
#         "followups"   : followups,
#         "cache_hit"   : False,
#         "q_class"     : q_class,
#         "facts_hit"   : False,
#         "log_id"      : lid,
#     }

#     # Store with embedding for fast future lookups
#     semantic_cache_store(cache, cache_key, result, sent_model)
#     _save_cache(cache)
#     return result


# # ==========================================================================================
# # RETRIEVAL EVALUATION FRAMEWORK
# # ==========================================================================================

# def _load_eval_dataset() -> list[dict]:
#     p = Path(CFG.EVAL_DATASET_FILE)
#     if not p.exists():
#         return []
#     try:
#         data = json.loads(p.read_text(encoding="utf-8"))
#         if isinstance(data, list):
#             return data
#     except Exception:
#         pass
#     return []


# def _precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
#     hits = sum(1 for r in retrieved[:k] if any(rel in r for rel in relevant))
#     return hits / k if k > 0 else 0.0


# def _recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
#     if not relevant:
#         return 0.0
#     hits = sum(1 for r in retrieved[:k] if any(rel in r for rel in relevant))
#     return hits / len(relevant)


# def _mrr(retrieved: list[str], relevant: list[str]) -> float:
#     for i, r in enumerate(retrieved):
#         if any(rel in r for rel in relevant):
#             return 1.0 / (i + 1)
#     return 0.0


# def _dcg(retrieved: list[str], relevant: list[str], k: int) -> float:
#     dcg = 0.0
#     for i, r in enumerate(retrieved[:k]):
#         rel = 1.0 if any(rel in r for rel in relevant) else 0.0
#         dcg += rel / math.log2(i + 2)
#     return dcg


# def _ndcg(retrieved: list[str], relevant: list[str], k: int) -> float:
#     ideal = sorted([1.0] * min(len(relevant), k) + [0.0] * max(0, k - len(relevant)), reverse=True)
#     idcg  = sum(v / math.log2(i + 2) for i, v in enumerate(ideal))
#     if idcg == 0:
#         return 0.0
#     return _dcg(retrieved, relevant, k) / idcg


# def run_retrieval_evaluation(
#     retriever : HybridRRFRetriever,
#     k         : int = 5,
#     run_label : str = "",
# ) -> Optional[dict]:
#     dataset = _load_eval_dataset()
#     if not dataset:
#         return None

#     precisions, recalls, mrrs, ndcgs = [], [], [], []
#     per_query_results = []

#     for item in dataset:
#         question = item.get("question", "")
#         relevant = item.get("relevant_sources", [])
#         if not question or not relevant:
#             continue

#         # Use expanded query for evaluation too
#         expanded = expand_query(question)
#         docs      = retriever.invoke(expanded, k=k)
#         retrieved = [d.metadata.get("source", "") for d in docs]

#         p_k = _precision_at_k(retrieved, relevant, k)
#         r_k = _recall_at_k(retrieved, relevant, k)
#         m   = _mrr(retrieved, relevant)
#         n   = _ndcg(retrieved, relevant, k)

#         precisions.append(p_k)
#         recalls.append(r_k)
#         mrrs.append(m)
#         ndcgs.append(n)
#         per_query_results.append({
#             "question"   : question,
#             "precision_k": round(p_k, 4),
#             "recall_k"   : round(r_k, 4),
#             "mrr"        : round(m, 4),
#             "ndcg"       : round(n, 4),
#         })

#     if not precisions:
#         return None

#     metrics = {
#         "run_label"       : run_label or datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
#         "k"               : k,
#         "precision_k"     : round(float(np.mean(precisions)), 4),
#         "recall_k"        : round(float(np.mean(recalls)), 4),
#         "mrr"             : round(float(np.mean(mrrs)), 4),
#         "ndcg"            : round(float(np.mean(ndcgs)), 4),
#         "num_queries"     : len(precisions),
#         # v9: summary statistics
#         "precision_std"   : round(float(np.std(precisions)), 4),
#         "recall_std"      : round(float(np.std(recalls)), 4),
#         "mrr_std"         : round(float(np.std(mrrs)), 4),
#         "ndcg_std"        : round(float(np.std(ndcgs)), 4),
#         "per_query"       : per_query_results,
#     }

#     try:
#         conn = _init_db()
#         conn.execute(
#             """INSERT INTO eval_results
#                (ts, run_label, k, precision_k, recall_k, mrr, ndcg, num_queries)
#                VALUES (?,?,?,?,?,?,?,?)""",
#             (
#                 datetime.datetime.now().isoformat(),
#                 metrics["run_label"], metrics["k"],
#                 metrics["precision_k"], metrics["recall_k"],
#                 metrics["mrr"], metrics["ndcg"], metrics["num_queries"],
#             ),
#         )
#         conn.commit()
#         conn.close()
#     except Exception:
#         pass

#     return metrics


# def metrics_to_csv(metrics_rows: list[dict]) -> str:
#     if not metrics_rows:
#         return ""
#     buf    = io.StringIO()
#     # Exclude per_query nested key for CSV export
#     clean_rows = [{k: v for k, v in row.items() if k != "per_query"} for row in metrics_rows]
#     writer = csv.DictWriter(buf, fieldnames=list(clean_rows[0].keys()))
#     writer.writeheader()
#     writer.writerows(clean_rows)
#     return buf.getvalue()


# # ==========================================================================================
# # DEFAULT EVAL DATASET
# # ==========================================================================================

# _DEFAULT_EVAL_DATASET = [
#     {"question": "How many students are enrolled in Loyola College?",
#      "relevant_sources": ["loyolacollege", "students", "strength", "enrolment"]},
#     {"question": "What is the NAAC accreditation grade of Loyola College?",
#      "relevant_sources": ["naac", "accreditation", "grade", "loyolacollege"]},
#     {"question": "What UG programmes does Loyola College offer?",
#      "relevant_sources": ["loyolacollege", "undergraduate", "ug", "programmes"]},
#     {"question": "What is the placement percentage at Loyola College?",
#      "relevant_sources": ["placement", "loyolacollege", "recruited", "campus"]},
#     {"question": "What is the average salary package offered during placements?",
#      "relevant_sources": ["placement", "salary", "package", "loyolacollege"]},
#     {"question": "How many faculty members are in Loyola College?",
#      "relevant_sources": ["faculty", "staff", "teachers", "loyolacollege"]},
#     {"question": "What is the NIRF ranking of Loyola College?",
#      "relevant_sources": ["nirf", "ranking", "loyolacollege"]},
#     {"question": "When was Loyola College established?",
#      "relevant_sources": ["established", "founded", "1925", "loyolacollege"]},
#     {"question": "What are the admission requirements for BSc Computer Science?",
#      "relevant_sources": ["computer science", "admission", "eligibility", "loyolacollege"]},
#     {"question": "What PG programmes does Loyola College offer?",
#      "relevant_sources": ["postgraduate", "pg", "msc", "mcom", "loyolacollege"]},
#     {"question": "What companies recruit from Loyola College?",
#      "relevant_sources": ["placement", "recruiters", "companies", "loyolacollege"]},
#     {"question": "What is the fee structure for BCA?",
#      "relevant_sources": ["bca", "fee", "tuition", "loyolacollege"]},
#     {"question": "What research facilities are available at Loyola College?",
#      "relevant_sources": ["research", "laboratory", "facilities", "loyolacollege"]},
#     {"question": "What is the hostel capacity of Loyola College?",
#      "relevant_sources": ["hostel", "accommodation", "loyolacollege"]},
#     {"question": "What scholarships are available at Loyola College?",
#      "relevant_sources": ["scholarship", "financial aid", "loyolacollege"]},
#     {"question": "Who is the principal of Loyola College?",
#      "relevant_sources": ["principal", "loyolacollege", "administration"]},
#     {"question": "What sports facilities are available at Loyola College?",
#      "relevant_sources": ["sports", "gymnasium", "playground", "loyolacollege"]},
#     {"question": "What are the library facilities at Loyola College?",
#      "relevant_sources": ["library", "loyolacollege", "books", "journals"]},
#     {"question": "What is the pass percentage in Loyola College university exams?",
#      "relevant_sources": ["pass", "result", "percentage", "university", "loyolacollege"]},
#     {"question": "What MBA specialisations are offered at Loyola College?",
#      "relevant_sources": ["mba", "specialisation", "management", "loyolacollege"]},
#     {"question": "Does Loyola College offer PhD programmes?",
#      "relevant_sources": ["phd", "doctorate", "research", "loyolacollege"]},
#     {"question": "What cultural events are held at Loyola College?",
#      "relevant_sources": ["cultural", "fest", "events", "loyolacollege"]},
#     {"question": "What is the intake capacity of BSc Physics?",
#      "relevant_sources": ["physics", "intake", "seats", "loyolacollege"]},
#     {"question": "What affiliation does Loyola College have?",
#      "relevant_sources": ["university of madras", "autonomous", "affiliation", "loyolacollege"]},
#     {"question": "What international collaborations does Loyola College have?",
#      "relevant_sources": ["international", "collaboration", "mou", "loyolacollege"]},
# ]


# def _ensure_eval_dataset() -> None:
#     """Create the default evaluation dataset if the file does not exist. Silent."""
#     p = Path(CFG.EVAL_DATASET_FILE)
#     if p.exists():
#         return
#     try:
#         p.parent.mkdir(parents=True, exist_ok=True)
#         p.write_text(json.dumps(_DEFAULT_EVAL_DATASET, indent=2), encoding="utf-8")
#     except Exception:
#         pass


# # ==========================================================================================
# # RICH SOURCE CARD UI
# # ==========================================================================================

# def render_source_cards(rich_sources: list[dict]) -> None:
#     if not rich_sources:
#         return
#     with st.expander(f"📎 Sources ({len(rich_sources)})"):
#         for rs in rich_sources:
#             title  = rs.get("title", "") or rs.get("source", "Unknown")
#             source = rs.get("source", "")
#             page   = rs.get("page", "")
#             dtype  = rs.get("type", "")
#             score  = rs.get("rerank_score", None)

#             col_main, col_meta = st.columns([3, 1])

#             with col_main:
#                 if source.startswith("http"):
#                     st.markdown(f"**[{title}]({source})**")
#                 else:
#                     st.markdown(f"**{title}**")
#                     st.caption(source)

#             with col_meta:
#                 badges = []
#                 if dtype:
#                     badges.append(f"`{dtype}`")
#                 if page:
#                     badges.append(f"p.{page}")
#                 if score is not None:
#                     badges.append(f"score: `{score:.3f}`")
#                 st.markdown("  ".join(badges))

#             st.divider()


# # ==========================================================================================
# # ANALYTICS TAB
# # ==========================================================================================

# def render_analytics_tab() -> None:
#     st.header("📊 Query Analytics")
#     try:
#         conn  = _init_db()
#         total = conn.execute("SELECT COUNT(*) FROM query_log").fetchone()[0]

#         if total == 0:
#             st.info("No queries logged yet.")
#             conn.close()
#             return

#         avg_lat      = conn.execute("SELECT AVG(latency_s) FROM query_log").fetchone()[0]
#         cache_pct    = conn.execute("SELECT 100.0*SUM(cache_hit)/COUNT(*) FROM query_log").fetchone()[0]
#         answered_pct = conn.execute("SELECT 100.0*SUM(answered)/COUNT(*) FROM query_log").fetchone()[0]

#         c1, c2, c3, c4 = st.columns(4)
#         c1.metric("Total queries",  total)
#         c2.metric("Avg latency",    f"{avg_lat:.2f}s" if avg_lat else "–")
#         c3.metric("Cache hit rate", f"{cache_pct:.1f}%" if cache_pct else "0%")
#         c4.metric("Answer rate",    f"{answered_pct:.1f}%" if answered_pct else "0%")

#         st.divider()
#         col_left, col_right = st.columns(2)

#         with col_left:
#             st.subheader("📂 Query types")
#             class_rows = conn.execute(
#                 "SELECT q_class, COUNT(*) as n FROM query_log GROUP BY q_class ORDER BY n DESC"
#             ).fetchall()
#             if class_rows:
#                 df_class = pd.DataFrame(class_rows, columns=["Query type", "Count"])
#                 st.bar_chart(df_class.set_index("Query type"))

#         with col_right:
#             st.subheader("🌐 Top website sections")
#             sec_rows = conn.execute(
#                 """SELECT section, COUNT(*) as n FROM query_log
#                    WHERE section IS NOT NULL AND section != ''
#                    GROUP BY section ORDER BY n DESC LIMIT 10"""
#             ).fetchall()
#             if sec_rows:
#                 df_sec = pd.DataFrame(sec_rows, columns=["Section", "Count"])
#                 st.bar_chart(df_sec.set_index("Section"))
#             else:
#                 st.caption("No section data yet.")

#         st.divider()
#         st.subheader("🔝 Top queries")
#         rows = conn.execute(
#             "SELECT query, COUNT(*) as n FROM query_log GROUP BY LOWER(query) ORDER BY n DESC LIMIT 10"
#         ).fetchall()
#         for q, n in rows:
#             st.write(f"- ({n}×) {q}")

#         st.divider()
#         st.subheader("🕐 Recent queries")
#         recent = conn.execute(
#             """SELECT ts, query, q_class, section, latency_s, cache_hit, confidence
#                FROM query_log ORDER BY id DESC LIMIT 25"""
#         ).fetchall()
#         df_recent = pd.DataFrame(
#             recent,
#             columns=["Timestamp", "Query", "Class", "Section", "Latency (s)", "Cache hit", "Confidence"],
#         )
#         st.dataframe(df_recent, use_container_width=True)
#         conn.close()

#     except Exception as e:
#         st.warning(f"Analytics unavailable: {e}")


# # ==========================================================================================
# # FAILURE ANALYTICS TAB
# # ==========================================================================================

# def render_failures_tab() -> None:
#     st.header("⚠️ Failure Analytics")
#     try:
#         conn       = _init_db()
#         total      = conn.execute("SELECT COUNT(*) FROM query_log").fetchone()[0]
#         unanswered = conn.execute("SELECT COUNT(*) FROM query_log WHERE answered=0").fetchone()[0]
#         low_conf   = conn.execute(
#             "SELECT COUNT(*) FROM query_log WHERE confidence='Low' AND answered=1"
#         ).fetchone()[0]

#         c1, c2, c3 = st.columns(3)
#         c1.metric("Total queries", total)
#         c2.metric("Unanswered",    unanswered,
#                   delta=f"{100*unanswered/max(total,1):.1f}% failure rate", delta_color="inverse")
#         c3.metric("Low-confidence", low_conf,
#                   delta=f"{100*low_conf/max(total,1):.1f}% of queries", delta_color="inverse")

#         st.divider()
#         col_l, col_r = st.columns(2)

#         with col_l:
#             st.subheader("❌ Unanswered queries")
#             rows = conn.execute(
#                 "SELECT query, ts FROM query_log WHERE answered=0 ORDER BY id DESC LIMIT 20"
#             ).fetchall()
#             if rows:
#                 st.dataframe(pd.DataFrame(rows, columns=["Query", "Timestamp"]), use_container_width=True)
#             else:
#                 st.success("No unanswered queries yet!")

#         with col_r:
#             st.subheader("🟡 Low-confidence answers")
#             rows = conn.execute(
#                 """SELECT query, section, ts FROM query_log
#                    WHERE confidence='Low' AND answered=1 ORDER BY id DESC LIMIT 20"""
#             ).fetchall()
#             if rows:
#                 st.dataframe(pd.DataFrame(rows, columns=["Query", "Section", "Timestamp"]), use_container_width=True)
#             else:
#                 st.success("No low-confidence answers logged yet.")

#         st.divider()
#         st.subheader("📉 Sections with low answer rate")
#         sec_fail = conn.execute(
#             """SELECT section,
#                       COUNT(*) as total,
#                       SUM(CASE WHEN answered=0 THEN 1 ELSE 0 END) as failed,
#                       ROUND(100.0*SUM(CASE WHEN answered=0 THEN 1 ELSE 0 END)/COUNT(*),1) as pct
#                FROM query_log
#                WHERE section IS NOT NULL AND section != ''
#                GROUP BY section HAVING total >= 2
#                ORDER BY pct DESC LIMIT 10"""
#         ).fetchall()
#         if sec_fail:
#             df_sf = pd.DataFrame(sec_fail, columns=["Section", "Total", "Failed", "Fail %"])
#             st.dataframe(df_sf, use_container_width=True)
#             st.bar_chart(df_sf.set_index("Section")["Fail %"])
#         else:
#             st.caption("Not enough section data yet.")

#         conn.close()
#     except Exception as e:
#         st.warning(f"Failure analytics unavailable: {e}")


# # ==========================================================================================
# # FEEDBACK TAB
# # ==========================================================================================

# def render_feedback_tab() -> None:
#     st.header("👍 User Feedback")
#     try:
#         conn     = _init_db()
#         total_fb = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]

#         if total_fb == 0:
#             st.info("No feedback recorded yet.")
#             conn.close()
#             return

#         helpful = conn.execute("SELECT COUNT(*) FROM feedback WHERE helpful=1").fetchone()[0]
#         pct     = 100 * helpful / total_fb if total_fb else 0

#         c1, c2, c3 = st.columns(3)
#         c1.metric("Total feedback",    total_fb)
#         c2.metric("👍 Helpful",        helpful)
#         c3.metric("Satisfaction rate", f"{pct:.1f}%")

#         st.divider()
#         col_l, col_r = st.columns(2)

#         with col_l:
#             st.subheader("📅 Feedback over time")
#             daily = conn.execute(
#                 """SELECT substr(ts,1,10) as day,
#                           SUM(helpful) as helpful, SUM(1-helpful) as not_helpful
#                    FROM feedback GROUP BY day ORDER BY day"""
#             ).fetchall()
#             if daily:
#                 st.bar_chart(pd.DataFrame(daily, columns=["Date", "Helpful", "Not helpful"]).set_index("Date"))

#         with col_r:
#             st.subheader("👎 Most disliked queries")
#             disliked = conn.execute(
#                 """SELECT query, COUNT(*) as thumbs_down FROM feedback WHERE helpful=0
#                    GROUP BY LOWER(query) ORDER BY thumbs_down DESC LIMIT 10"""
#             ).fetchall()
#             if disliked:
#                 st.dataframe(pd.DataFrame(disliked, columns=["Query", "Thumbs down"]), use_container_width=True)

#         st.divider()
#         st.subheader("🕐 Recent feedback")
#         recent = conn.execute(
#             """SELECT ts, query,
#                       CASE helpful WHEN 1 THEN '👍' ELSE '👎' END as vote
#                FROM feedback ORDER BY id DESC LIMIT 30"""
#         ).fetchall()
#         st.dataframe(pd.DataFrame(recent, columns=["Timestamp", "Query", "Vote"]), use_container_width=True)
#         conn.close()

#     except Exception as e:
#         st.warning(f"Feedback analytics unavailable: {e}")


# # ==========================================================================================
# # EVALUATION TAB  — v9: per-query breakdown, summary stats, multi-run comparison
# # ==========================================================================================

# def render_evaluation_tab(retriever: HybridRRFRetriever) -> None:
#     st.header("📐 Retrieval Evaluation")

#     _ensure_eval_dataset()
#     dataset = _load_eval_dataset()

#     if not dataset:
#         st.info("Evaluation dataset unavailable. Add `data/eval_dataset.json` to enable this tab.")
#         return

#     st.success(f"✅ Evaluation dataset — {len(dataset)} questions")

#     col_k, col_label, col_run = st.columns([1, 2, 1])
#     k         = col_k.selectbox("Top-K to evaluate", [3, 5, 10], index=1)
#     run_label = col_label.text_input("Run label (optional)", placeholder="e.g. v9-baseline")
#     run_now   = col_run.button("▶ Run evaluation", use_container_width=True)

#     if run_now:
#         with st.spinner(f"Evaluating {len(dataset)} queries at K={k}..."):
#             metrics = run_retrieval_evaluation(retriever, k=k, run_label=run_label)

#         if metrics:
#             st.success("Evaluation complete!")

#             mc1, mc2, mc3, mc4 = st.columns(4)
#             mc1.metric(f"Precision@{k}", f"{metrics['precision_k']:.4f}",
#                        delta=f"±{metrics['precision_std']:.4f}")
#             mc2.metric(f"Recall@{k}",    f"{metrics['recall_k']:.4f}",
#                        delta=f"±{metrics['recall_std']:.4f}")
#             mc3.metric("MRR",            f"{metrics['mrr']:.4f}",
#                        delta=f"±{metrics['mrr_std']:.4f}")
#             mc4.metric(f"NDCG@{k}",      f"{metrics['ndcg']:.4f}",
#                        delta=f"±{metrics['ndcg_std']:.4f}")

#             # Per-query breakdown
#             if metrics.get("per_query"):
#                 with st.expander("📋 Per-query breakdown"):
#                     df_pq = pd.DataFrame(metrics["per_query"])
#                     st.dataframe(df_pq, use_container_width=True)

#                     # Export per-query results
#                     pq_csv = df_pq.to_csv(index=False)
#                     st.download_button(
#                         label     = "⬇️ Export per-query results as CSV",
#                         data      = pq_csv,
#                         file_name = f"loyola_eval_per_query_{run_label or 'run'}.csv",
#                         mime      = "text/csv",
#                     )
#         else:
#             st.error("Evaluation failed — check dataset format.")

#     st.divider()
#     st.subheader("📜 Evaluation history")

#     try:
#         conn = _init_db()
#         rows = conn.execute(
#             """SELECT ts, run_label, k, precision_k, recall_k, mrr, ndcg, num_queries
#                FROM eval_results ORDER BY id DESC LIMIT 20"""
#         ).fetchall()
#         conn.close()

#         if rows:
#             cols    = ["Timestamp", "Run label", "K", "Precision@K", "Recall@K", "MRR", "NDCG", "Queries"]
#             df_eval = pd.DataFrame(rows, columns=cols)
#             st.dataframe(df_eval, use_container_width=True)

#             csv_str = metrics_to_csv([dict(zip(cols, r)) for r in rows])
#             st.download_button(
#                 label     = "⬇️ Export evaluation history as CSV",
#                 data      = csv_str,
#                 file_name = "loyola_rag_eval_results.csv",
#                 mime      = "text/csv",
#             )

#             # Multi-run comparison chart
#             if len(df_eval) > 1:
#                 st.subheader("📈 Metric trends across runs")
#                 chart_df = df_eval.set_index("Run label")[["Precision@K", "Recall@K", "MRR", "NDCG"]]
#                 st.line_chart(chart_df)
#         else:
#             st.info("No evaluation runs yet. Click 'Run evaluation' above.")

#     except Exception as e:
#         st.warning(f"Could not load evaluation history: {e}")


# # ==========================================================================================
# # STREAMLIT UI
# # ==========================================================================================

# def main():

#     st.markdown("""
#         <div style='text-align:center; padding: 1rem 0'>
#             <h1>🎓 Loyola College Assistant</h1>
#             <p style='color: gray'>Ask anything about Loyola College, Chennai — v9.0</p>
#         </div>
#     """, unsafe_allow_html=True)

#     # ── Sidebar ───────────────────────────────────────────────────────
#     st.sidebar.image(
#         "https://www.loyolacollege.edu/wp-content/uploads/2021/09/loyola-logo.png",
#         width=120,
#     )
#     st.sidebar.title("⚙️ Settings")

#     api_key = os.getenv("GROQ_API_KEY", "").strip()
#     if not api_key:
#         st.error("❌ GROQ_API_KEY not found in .env file.")
#         st.stop()

#     st.sidebar.divider()

#     if st.sidebar.button("🔄 Rebuild Knowledge Base", use_container_width=True):
#         for f in [CFG.CHUNKS_PKL, CFG.BM25_PKL, CFG.HASH_FILE, CFG.QUERY_CACHE_FILE]:
#             if os.path.exists(f):
#                 os.remove(f)
#         for f in ["index.faiss", "index.pkl"]:
#             fp = Path(CFG.FAISS_PATH) / f
#             if fp.exists():
#                 fp.unlink()
#         st.cache_resource.clear()
#         st.sidebar.success("Cleared — rebuilding...")
#         st.rerun()

#     # Source type filter
#     st.sidebar.divider()
#     st.sidebar.caption("🔍 Source filter (optional)")
#     selected_types = st.sidebar.multiselect(
#         "Restrict retrieval to",
#         options = ["website", "pdf", "ocr", "cleaned_text"],
#         default = [],
#         help    = "Leave blank to search all sources.",
#     )
#     active_filter = selected_types if selected_types else None

#     # Debug mode
#     st.sidebar.divider()
#     debug_mode = st.sidebar.checkbox(
#         "🐛 Debug mode",
#         value = False,
#         help  = "Show retrieved chunks, rerank scores, and expanded queries.",
#     )

#     st.sidebar.divider()
#     st.sidebar.caption("📁 Knowledge Base Status")
#     sidebar_log = st.sidebar.container()

#     if can_fast_load():
#         st.sidebar.success("⚡ Index ready — fast load")
#     else:
#         st.sidebar.warning("🔄 First run — full build required")

#     with st.sidebar.expander("ℹ️ System info"):
#         st.caption(f"Device: `{DEVICE}`")
#         st.caption(f"Embedding: `{CFG.EMBEDDING_MODEL}`")
#         st.caption(f"Reranker: `{CFG.RERANK_MODEL}`")
#         st.caption(f"LLM: `{CFG.LLM_MODEL}`")
#         st.caption(f"Chunk: {CFG.CHUNK_SIZE} / overlap {CFG.CHUNK_OVERLAP}")
#         st.caption(f"Rerank K={CFG.RERANK_K} | pool={CFG.RERANK_CANDIDATE_POOL}")
#         st.caption(f"Retrieval K — fact:{CFG.RETRIEVAL_K_FACT} default:{CFG.RETRIEVAL_K_DEFAULT} "
#                    f"deep:{CFG.RETRIEVAL_K_DEEP} compare:{CFG.RETRIEVAL_K_COMPARE}")
#         st.caption(f"Cache max size: {CFG.SEMANTIC_CACHE_MAX_SIZE} entries (LRU)")

#     # ── Load models ───────────────────────────────────────────────────
#     with st.spinner("Loading AI models..."):
#         emb        = load_embedding_model()
#         reranker   = load_reranker()
#         sent_model = load_sentence_model()
#         llm_mdl    = get_llm(api_key)
#         facts      = load_facts()

#     # ── Load knowledge base ───────────────────────────────────────────
#     with st.spinner(
#         "Loading knowledge base..." if can_fast_load()
#         else "Building knowledge base (first time only)..."
#     ):
#         db, chunks, bm25 = build_knowledge_base(sidebar_log, emb)
#         retriever = HybridRRFRetriever(db, bm25)

#     st.success(f"✅ Ready — {len(chunks):,} chunks loaded")

#     # ── Session state ─────────────────────────────────────────────────
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []
#     if "query_cache" not in st.session_state:
#         st.session_state.query_cache = _load_cache()

#     # ── Tabs ──────────────────────────────────────────────────────────
#     tab_chat, tab_analytics, tab_failures, tab_feedback, tab_eval = st.tabs([
#         "💬 Chat", "📊 Analytics", "⚠️ Failures", "👍 Feedback", "📐 Evaluation",
#     ])

#     # ================================================================
#     # CHAT TAB
#     # ================================================================
#     with tab_chat:

#         # Render conversation history
#         for idx, chat in enumerate(st.session_state.chat_history):
#             with st.chat_message("user"):
#                 st.markdown(chat["q"])
#             with st.chat_message("assistant"):
#                 st.markdown(chat["a"])

#                 conf      = chat.get("confidence", "?")
#                 emoji     = chat.get("conf_emoji", "")
#                 score     = chat.get("score", 0.0)
#                 facts_hit = chat.get("facts_hit", False)

#                 if facts_hit:
#                     st.caption("⚡ Answered from structured facts layer")
#                 elif conf != "?":
#                     st.caption(f"{emoji} Confidence: **{conf}** (grounding score: {score:.2f})")

#                 rich_sources = chat.get("rich_sources", [])
#                 if rich_sources:
#                     render_source_cards(rich_sources)
#                 elif chat.get("sources"):
#                     with st.expander("📎 Sources"):
#                         for src in chat["sources"]:
#                             if src.startswith("http"):
#                                 st.markdown(f"- [{src}]({src})")
#                             else:
#                                 st.caption(f"- {src}")

#                 if chat.get("followups"):
#                     st.markdown("**💡 You might also ask:**")
#                     cols = st.columns(len(chat["followups"]))
#                     for i, fq in enumerate(chat["followups"]):
#                         if cols[i].button(fq, key=f"fq_hist_{idx}_{i}"):
#                             st.session_state["prefill_query"] = fq
#                             st.rerun()

#                 log_id    = chat.get("log_id")
#                 voted_key = f"voted_{idx}"
#                 if not st.session_state.get(voted_key, False):
#                     fb_cols = st.columns([1, 1, 8])
#                     if fb_cols[0].button("👍", key=f"up_{idx}"):
#                         log_feedback(log_id, chat["q"], helpful=True)
#                         st.session_state[voted_key] = True
#                         st.rerun()
#                     if fb_cols[1].button("👎", key=f"dn_{idx}"):
#                         log_feedback(log_id, chat["q"], helpful=False)
#                         st.session_state[voted_key] = True
#                         st.rerun()
#                 else:
#                     st.caption("✅ Feedback recorded — thank you!")

#         # ── Chat input ────────────────────────────────────────────────
#         prefill    = st.session_state.pop("prefill_query", None)
#         user_query = st.chat_input(
#             "Ask about admissions, programmes, placements, fees, faculty…"
#         ) or prefill

#         if user_query:

#             with st.chat_message("user"):
#                 st.markdown(user_query)

#             with st.chat_message("assistant"):
#                 placeholder   = st.empty()
#                 conf_slot     = st.empty()
#                 debug_slot    = st.empty()
#                 sources_slot  = st.empty()
#                 followup_slot = st.empty()
#                 feedback_slot = st.empty()

#                 with st.spinner("Thinking..."):
#                     result = ask(
#                         query       = user_query,
#                         retriever   = retriever,
#                         reranker    = reranker,
#                         llm_mdl     = llm_mdl,
#                         sent_model  = sent_model,
#                         history     = st.session_state.chat_history,
#                         cache       = st.session_state.query_cache,
#                         facts       = facts,
#                         type_filter = active_filter,
#                         debug       = debug_mode,
#                     )

#                 answer       = result["answer"]
#                 sources      = result["sources"]
#                 rich_sources = result.get("rich_sources", [])
#                 conf         = result["confidence"]
#                 emoji        = result["conf_emoji"]
#                 score        = result["score"]
#                 followups    = result["followups"]
#                 q_class      = result["q_class"]
#                 cache_hit    = result["cache_hit"]
#                 facts_hit    = result["facts_hit"]
#                 log_id       = result.get("log_id")

#                 placeholder.markdown(answer)

#                 badges = []
#                 if facts_hit:
#                     badges.append("⚡ Structured facts")
#                 elif conf != "N/A":
#                     badges.append(f"{emoji} Confidence: **{conf}** (score: {score:.2f})")
#                 if cache_hit:
#                     badges.append("🗂 Cached")
#                 if active_filter:
#                     badges.append(f"🔍 Filtered: {', '.join(active_filter)}")
#                 badges.append(f"🏷 `{q_class}`")
#                 conf_slot.caption("  |  ".join(badges))

#                 # Debug expander — v9 also shows expanded query
#                 if debug_mode:
#                     with debug_slot.expander("🐛 Debug: retrieval details", expanded=False):
#                         st.markdown(f"**Effective query:** `{user_query}`")
#                         st.markdown(f"**Expanded query:** `{expand_query(user_query)}`")
#                         st.markdown(f"**Class:** `{q_class}` | **K:** `{get_retrieval_k(q_class)}`")
#                         if rich_sources:
#                             for i, rs in enumerate(rich_sources):
#                                 st.markdown(
#                                     f"**[{i+1}]** score=`{rs['rerank_score']}` | "
#                                     f"type=`{rs['type']}` | "
#                                     f"`{rs['source'][:80]}`"
#                                 )

#                 if rich_sources:
#                     with sources_slot.container():
#                         render_source_cards(rich_sources)
#                 elif sources:
#                     with sources_slot.expander("📎 Sources"):
#                         for src in sources:
#                             if src.startswith("http"):
#                                 st.markdown(f"- [{src}]({src})")
#                             else:
#                                 st.caption(f"- {src}")

#                 if followups:
#                     followup_slot.markdown("**💡 You might also ask:**")
#                     cols = st.columns(len(followups))
#                     for i, fq in enumerate(followups):
#                         if cols[i].button(fq, key=f"fq_new_{i}"):
#                             st.session_state["prefill_query"] = fq
#                             st.rerun()

#                 with feedback_slot.container():
#                     fb_cols = st.columns([1, 1, 8])
#                     if fb_cols[0].button("👍", key="fb_up_new"):
#                         log_feedback(log_id, user_query, helpful=True)
#                         st.toast("Thanks for the feedback! 👍")
#                     if fb_cols[1].button("👎", key="fb_dn_new"):
#                         log_feedback(log_id, user_query, helpful=False)
#                         st.toast("Thanks for the feedback! 👎")

#             chat_entry = {
#                 "q"           : user_query,
#                 "a"           : answer,
#                 "sources"     : sources,
#                 "rich_sources": rich_sources,
#                 "confidence"  : conf,
#                 "conf_emoji"  : emoji,
#                 "score"       : score,
#                 "followups"   : followups,
#                 "facts_hit"   : facts_hit,
#                 "log_id"      : log_id,
#             }
#             st.session_state.chat_history.append(chat_entry)
#             st.session_state.chat_history = \
#                 st.session_state.chat_history[-CFG.HISTORY_TURNS * 2:]

#     # ================================================================
#     # OTHER TABS
#     # ================================================================
#     with tab_analytics:
#         render_analytics_tab()

#     with tab_failures:
#         render_failures_tab()

#     with tab_feedback:
#         render_feedback_tab()

#     with tab_eval:
#         render_evaluation_tab(retriever)


# # ==========================================================================================
# # ENTRY POINT
# # ==========================================================================================

# if __name__ == "__main__":
#     main()
# #



























































































































# ==========================================================================================
# LOYOLA COLLEGE — WEBSITE RAG CHATBOT  v10.0
# ==========================================================================================
#
# WHAT CHANGED FROM v9.0  (targeted bug fixes from live test results)
# ────────────────────────────────────────────────────────────────────
#
# BUG FIX 1 — Facts layer comparison bypass
#   Problem : "What is the difference between BCA and BSc CS?" triggered the
#             programme-list branch in query_facts_layer() and returned the full
#             programme list instead of going to RAG retrieval.
#   Fix     : query_facts_layer() now returns None immediately when the query
#             contains any comparison keyword (difference, compare, vs, versus,
#             better, contrast, pros, cons, advantages, disadvantages).
#             Comparison queries always fall through to the full RAG pipeline.
#
# BUG FIX 2 — Facts layer fee + course collision
#   Problem : "What is the exact fee for BSc Physics in 2024?" matched the
#             course/programme keyword branch (bsc) BEFORE the fee branch,
#             returning the programme list instead of the fee disclaimer.
#   Fix     : Fee check is now the FIRST branch in query_facts_layer() (after
#             the comparison guard), so specific fee questions always get the
#             fee disclaimer and are correctly routed to RAG for exact amounts.
#
# BUG FIX 3 — Statistics contradictory numbers
#   Problem : Two source chunks with conflicting totals (9242 students but
#             combined total listed as 3721) were both included in context,
#             causing the LLM to list contradictory figures.
#   Fix     : build_context() now detects numeric contradictions within the
#             same stat category (student count, faculty count, etc.) and keeps
#             only the highest-scoring chunk per numeric category.
#             Prompt Rule 13 added: "If two context chunks give different numbers
#             for the same fact, report only the figure from the highest-scored
#             chunk and note the discrepancy."
#
# BUG FIX 4 — Follow-up suggestions missing after facts-layer answers
#   Problem : Facts-layer answers (Q1, Q3, Q4) returned empty follow-up
#             suggestions because generate_followups() was called without
#             context when facts_hit=True.
#   Fix     : For facts-layer hits, pass the fact_answer text itself as
#             context to generate_followups() so suggestions are grounded.
#
# CARRIED FORWARD FROM v9.0 (all improvements preserved)
# ────────────────────────────────────────────────────────
# ✔  Query expansion (synonyms + college keywords)
# ✔  Smarter query rewriting
# ✔  Dynamic K (4-tier: fact=8 / default=16 / deep=24 / compare=28)
# ✔  8-class query classification
# ✔  Improved context builder (dedup + diversity)
# ✔  Better reranking (pool=40, adaptive threshold)
# ✔  Stronger hallucination prompt
# ✔  Better grounding score (max-sim across all chunks)
# ✔  Semantic cache v2 (stored embeddings, LRU eviction)
# ✔  Grounded follow-ups
# ✔  Rich source attribution
# ✔  Evaluation improvements (per-query, std-dev, multi-run chart)
# ✔  All analytics / feedback / evaluation tabs
# ✔  Debug mode
# ✔  Security filter
#
# RUN
# ───
#   streamlit run rag_v10.py
#
# ==========================================================================================


# ==========================================================================================
# IMPORTS
# ==========================================================================================

import csv
import io
import os
import re
import json
import math
import pickle
import sqlite3
import hashlib
import warnings
import time
import datetime
import unicodedata
import collections

from pathlib     import Path
from dataclasses import dataclass, field
from typing      import Optional

import numpy as np
import pandas as pd
import streamlit as st
import tiktoken

from dotenv                           import load_dotenv
from sentence_transformers            import CrossEncoder, SentenceTransformer

from langchain_core.documents         import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers   import BM25Retriever

from langchain_community.document_loaders import (
    PyMuPDFLoader,
    PDFPlumberLoader,
)

from langchain_text_splitters       import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq                 import ChatGroq

warnings.filterwarnings("ignore")
load_dotenv()


# ==========================================================================================
# PAGE CONFIG
# ==========================================================================================

st.set_page_config(
    page_title = "Loyola College Assistant v10",
    page_icon  = "🎓",
    layout     = "wide",
)


# ==========================================================================================
# CONFIGURATION
# ==========================================================================================

@dataclass
class Config:

    # ── Data folders ──────────────────────────────────────────────────
    RAG_JSON_FOLDER    : str = "./data/rag_export"
    CLEANED_TEXT_FOLDER: str = "./data/cleaned_text"
    OCR_TEXT_FOLDER    : str = "./data/ocr_text"
    PDF_FOLDER         : str = "./data/pdfs"
    FACTS_FILE         : str = "./data/facts.json"
    EVAL_DATASET_FILE  : str = "./data/eval_dataset.json"

    # ── Persistence ───────────────────────────────────────────────────
    FAISS_PATH         : str = "./faiss_index"
    CHUNKS_PKL         : str = "./faiss_index/chunks.pkl"
    BM25_PKL           : str = "./faiss_index/bm25.pkl"
    HASH_FILE          : str = "./faiss_index/data_hash.txt"
    QUERY_CACHE_FILE   : str = "./faiss_index/query_cache.pkl"
    ANALYTICS_DB       : str = "./faiss_index/analytics.db"

    # ── Models ────────────────────────────────────────────────────────
    EMBEDDING_MODEL    : str = "BAAI/bge-small-en-v1.5"
    RERANK_MODEL       : str = "BAAI/bge-reranker-base"
    LLM_MODEL          : str = "llama-3.1-8b-instant"

    # ── Chunking ──────────────────────────────────────────────────────
    CHUNK_SIZE         : int = 1200
    CHUNK_OVERLAP      : int = 200

    # ── Retrieval — dynamic K by query class ──────────────────────────
    RETRIEVAL_K_FACT   : int = 8     # simple fact questions
    RETRIEVAL_K_DEFAULT: int = 16    # general questions
    RETRIEVAL_K_DEEP   : int = 24    # statistics, eligibility, date
    RETRIEVAL_K_COMPARE: int = 28    # comparison / analytical questions

    # ── Reranking ─────────────────────────────────────────────────────
    RERANK_CANDIDATE_POOL : int   = 40    # raised from 30 for better recall
    RERANK_K              : int   = 6
    RERANK_THRESHOLD      : float = 0.15  # base threshold (adaptive in code)

    # ── Context ───────────────────────────────────────────────────────
    MAX_CONTEXT_TOKENS : int = 3500
    RRF_K              : int = 60

    # ── Semantic cache ────────────────────────────────────────────────
    SEMANTIC_CACHE_THRESHOLD : float = 0.92
    SEMANTIC_CACHE_MAX_SIZE  : int   = 500   # LRU eviction above this

    # ── Hallucination grounding ───────────────────────────────────────
    GROUNDING_THRESHOLD : float = 0.25
    CONFIDENCE_HIGH     : float = 0.55
    CONFIDENCE_MED      : float = 0.35

    # ── Chat ──────────────────────────────────────────────────────────
    HISTORY_TURNS       : int = 4


CFG = Config()
Path(CFG.FAISS_PATH).mkdir(parents=True, exist_ok=True)


# ==========================================================================================
# DEVICE
# ==========================================================================================

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"


# ==========================================================================================
# TOKEN COUNTER
# ==========================================================================================

_tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(_tokenizer.encode(text))


# ==========================================================================================
# QUERY CACHE  — v9: LRU with embedded vectors
# ==========================================================================================
# Cache entries now store the query embedding alongside the result so
# semantic_cache_lookup never re-encodes cached queries.

def _load_cache() -> "collections.OrderedDict":
    try:
        if os.path.exists(CFG.QUERY_CACHE_FILE):
            obj = pickle.load(open(CFG.QUERY_CACHE_FILE, "rb"))
            # Upgrade plain dict from v8 → OrderedDict
            if isinstance(obj, dict) and not isinstance(obj, collections.OrderedDict):
                od = collections.OrderedDict(obj)
                return od
            return obj
    except Exception:
        pass
    return collections.OrderedDict()


def _save_cache(cache: "collections.OrderedDict") -> None:
    try:
        pickle.dump(cache, open(CFG.QUERY_CACHE_FILE, "wb"))
    except Exception:
        pass


def _cache_set(cache: "collections.OrderedDict", key: str, value: dict) -> None:
    """Insert / update with LRU eviction when cache exceeds max size."""
    if key in cache:
        cache.move_to_end(key)
    cache[key] = value
    while len(cache) > CFG.SEMANTIC_CACHE_MAX_SIZE:
        cache.popitem(last=False)   # remove oldest (FIFO / LRU)


# ==========================================================================================
# DATABASE INIT
# ==========================================================================================

def _init_db() -> sqlite3.Connection:
    """Create / migrate analytics.db with all required tables."""
    conn = sqlite3.connect(CFG.ANALYTICS_DB)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_log (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            ts         TEXT,
            query      TEXT,
            q_class    TEXT,
            latency_s  REAL,
            cache_hit  INTEGER,
            confidence TEXT,
            answered   INTEGER,
            section    TEXT
        )
    """)

    try:
        conn.execute("ALTER TABLE query_log ADD COLUMN section TEXT")
    except sqlite3.OperationalError:
        pass

    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            ts         TEXT,
            log_id     INTEGER,
            query      TEXT,
            helpful    INTEGER
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS eval_results (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            ts           TEXT,
            run_label    TEXT,
            k            INTEGER,
            precision_k  REAL,
            recall_k     REAL,
            mrr          REAL,
            ndcg         REAL,
            num_queries  INTEGER
        )
    """)

    conn.commit()
    return conn


# ==========================================================================================
# QUERY ANALYTICS
# ==========================================================================================

def log_query(
    query     : str,
    q_class   : str,
    latency_s : float,
    cache_hit : bool,
    confidence: str,
    answered  : bool,
    section   : str = "",
) -> Optional[int]:
    """Log a query to analytics.db. Returns the inserted row id."""
    try:
        conn = _init_db()
        cur  = conn.execute(
            """INSERT INTO query_log
               (ts, query, q_class, latency_s, cache_hit, confidence, answered, section)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                datetime.datetime.now().isoformat(),
                query, q_class, round(latency_s, 3),
                int(cache_hit), confidence, int(answered), section,
            ),
        )
        conn.commit()
        row_id = cur.lastrowid
        conn.close()
        return row_id
    except Exception:
        return None


def log_feedback(log_id: Optional[int], query: str, helpful: bool) -> None:
    """Record a 👍 or 👎 for a specific query."""
    try:
        conn = _init_db()
        conn.execute(
            "INSERT INTO feedback (ts, log_id, query, helpful) VALUES (?,?,?,?)",
            (datetime.datetime.now().isoformat(), log_id, query, int(helpful)),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


# ==========================================================================================
# SECURITY FILTER
# ==========================================================================================

_BLOCKED = [
    r"ignore (previous|all) instructions",
    r"developer mode",
    r"jailbreak",
    r"system prompt",
    r"forget (your|all) instructions",
    r"act as (an? )?(unrestricted|evil|unethical)",
    r"pretend you (are|have no)",
    r"bypass (your )?(safety|filter|restriction)",
    r"disregard (your )?guidelines",
    r"you are now",
    r"new persona",
]

def is_malicious(q: str) -> bool:
    q_lower = q.lower()
    if any(re.search(p, q_lower) for p in _BLOCKED):
        return True
    q_norm = unicodedata.normalize("NFKD", q_lower)
    return any(re.search(p, q_norm) for p in _BLOCKED)


# ==========================================================================================
# TEXT CLEANER
# ==========================================================================================

def clean_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\x20-\x7E\u0900-\u097F\u0B80-\u0BFF\n]", "", t)
    return t.strip()


# ==========================================================================================
# NOISE CHUNK FILTER
# ==========================================================================================

_NOISE_PATTERNS = [
    re.compile(r'(https?://\S+\s*){3,}'),
    re.compile(r'([a-z0-9_.+-]+@[a-z0-9-]+\.[a-z]+\s*){3,}', re.I),
    re.compile(r'(©|\bAll rights reserved\b|\bPrivacy Policy\b)', re.I),
    re.compile(r'\b(Home\s*[|>]\s*About|Skip to content|Cookie Policy)\b', re.I),
    re.compile(r'\b(Follow us on|Share this page|Subscribe to our)\b', re.I),
]

_STAT_MARKERS = re.compile(
    r'\b(\d[\d,]+\s*(students?|faculty|staff|crore|lakh|%|placed|recruited|alumni|departments?))\b',
    re.I,
)

def is_noise_chunk(text: str) -> bool:
    """Return True if chunk is predominantly navigation / contact / footer boilerplate."""
    if len(text.strip()) < 80:
        return True
    if _STAT_MARKERS.search(text):
        return False
    words       = text.split()
    word_count  = max(len(words), 1)
    url_count   = len(re.findall(r'https?://\S+', text))
    email_count = len(re.findall(r'[a-z0-9_.+-]+@[a-z0-9-]+\.[a-z]{2,}', text, re.I))
    if (url_count + email_count) / word_count > 0.30:
        return True
    return any(p.search(text) for p in _NOISE_PATTERNS)


# ==========================================================================================
# DATA HASH
# ==========================================================================================

def compute_data_hash() -> str:
    h = hashlib.sha256()
    for folder in [
        CFG.RAG_JSON_FOLDER, CFG.CLEANED_TEXT_FOLDER,
        CFG.OCR_TEXT_FOLDER, CFG.PDF_FOLDER,
    ]:
        p = Path(folder)
        if not p.exists():
            continue
        for f in sorted(p.iterdir()):
            h.update(f.name.encode())
            h.update(str(f.stat().st_size).encode())
    return h.hexdigest()


def data_changed() -> bool:
    if not os.path.exists(CFG.HASH_FILE):
        return True
    return Path(CFG.HASH_FILE).read_text().strip() != compute_data_hash()


def save_data_hash() -> None:
    Path(CFG.HASH_FILE).write_text(compute_data_hash())


# def can_fast_load() -> bool:
#     return (
#         os.path.exists(CFG.FAISS_PATH)
#         and os.path.exists(CFG.CHUNKS_PKL)
#         and os.path.exists(CFG.BM25_PKL)
#         and not data_changed()
#     )

def can_fast_load():
    return (
        os.path.exists(CFG.FAISS_PATH)
        and os.path.exists(CFG.CHUNKS_PKL)
        and os.path.exists(CFG.BM25_PKL)
    )


# ==========================================================================================
# SECTION EXTRACTOR
# ==========================================================================================

def extract_section_from_source(source: str) -> str:
    if not source:
        return ""
    s = source.lower()
    if s.endswith(".pdf") or "/pdfs/" in s:
        return "pdf"
    if "/ocr_text/" in s or "/ocr/" in s:
        return "ocr"
    if "/cleaned_text/" in s or "/text/" in s:
        return "cleaned_text"
    try:
        from urllib.parse import urlparse
        parsed = urlparse(source)
        parts  = [p for p in parsed.path.split("/") if p]
        if parts:
            return parts[0].lower()
    except Exception:
        pass
    return ""


# ==========================================================================================
# STRUCTURED FACTS LAYER
# ==========================================================================================

_DEFAULT_FACTS = {
    "fees": {
        "description": "Fee information must be confirmed with the college office as it changes yearly.",
        "contact": "accounts@loyolacollege.edu or +91-44-2817-2624",
    },
    "contact": {
        "address": "Loyola College (Autonomous), Nungambakkam, Chennai - 600 034, Tamil Nadu, India",
        "phone": "+91-44-2817-2624 / 2817-3274",
        "email": "principal@loyolacollege.edu",
        "website": "www.loyolacollege.edu",
    },
    "established": "1925",
    "affiliation": "University of Madras (Autonomous status granted 1978)",
    "accreditation": "NAAC A++ (reaccredited 2022)",
    "departments": [
        "Commerce", "Economics", "English", "Tamil", "Hindi", "French",
        "Physics", "Chemistry", "Mathematics", "Statistics", "Computer Science",
        "Biochemistry", "Microbiology", "Plant Biology and Biotechnology",
        "Advanced Zoology and Biotechnology", "Visual Communication",
        "Social Work", "History", "Philosophy", "Psychology",
        "Business Administration (BBA)", "Computer Applications (BCA)",
    ],
    "programmes": {
        "UG Programmes": [
            "B.A. English Literature", "B.A. Tamil", "B.A. Hindi", "B.A. French",
            "B.A. History", "B.A. Philosophy", "B.A. Economics",
            "B.Com. (General)", "B.Com. (Professional Accounting)",
            "B.Sc. Physics", "B.Sc. Chemistry", "B.Sc. Mathematics",
            "B.Sc. Statistics", "B.Sc. Computer Science", "B.Sc. Biochemistry",
            "B.Sc. Microbiology", "B.Sc. Plant Biology & Biotechnology",
            "B.Sc. Advanced Zoology & Biotechnology",
            "B.Sc. Visual Communication",
            "B.B.A. Business Administration", "B.C.A. Computer Applications",
            "B.S.W. Social Work",
        ],
        "PG Programmes": [
            "M.A. English", "M.A. Tamil", "M.A. History", "M.A. Philosophy",
            "M.A. Economics", "M.Com.", "M.Sc. Physics", "M.Sc. Chemistry",
            "M.Sc. Mathematics", "M.Sc. Statistics", "M.Sc. Computer Science",
            "M.Sc. Biochemistry", "M.Sc. Microbiology",
            "M.Sc. Plant Biology & Biotechnology",
            "M.Sc. Advanced Zoology & Biotechnology",
            "M.S.W. Social Work", "M.C.A.", "M.B.A.",
        ],
        "M.Phil. / Ph.D.": [
            "Available in most UG/PG departments. Contact the Research Cell for details.",
        ],
    },
}


@st.cache_resource(show_spinner=False)
def load_facts() -> dict:
    p = Path(CFG.FACTS_FILE)
    if p.exists():
        try:
            data   = json.loads(p.read_text(encoding="utf-8"))
            merged = {**_DEFAULT_FACTS, **data}
            return merged
        except Exception:
            pass
    return _DEFAULT_FACTS


def query_facts_layer(question: str, facts: dict) -> Optional[str]:
    q = question.lower()

    # ── BUG FIX 1 (v10): Comparison bypass ───────────────────────────
    # Comparison/contrast questions must always go to RAG retrieval so
    # the LLM can diff real content. Never answer from the facts layer.
    _COMPARISON_BYPASS = re.compile(
        r"\b(difference|compare|versus|vs\.?|better|worse|contrast|"
        r"pros and cons|advantages|disadvantages|which is|similarities)\b",
        re.I,
    )
    if _COMPARISON_BYPASS.search(q):
        return None

    # ── BUG FIX 2 (v10): Fee check FIRST, before course/programme ────
    # "fee for BSc Physics" contains "bsc" which previously matched the
    # programme branch before reaching the fee branch.  Fee intent must
    # win whenever fee keywords are present, regardless of subject name.
    if any(k in q for k in ["fee", "fees", "tuition", "cost", "how much", "charges", "payment"]):
        # If asking for a specific programme's fee, fall through to RAG
        # so the LLM can search for the exact figure.  Only return the
        # generic disclaimer when the question is clearly about fees in
        # general (no specific programme / year mentioned).
        specific_fee = re.search(
            r"\b(bsc|bca|bba|bcom|bsw|ba |msc|mca|mba|mcom|ma |msw|"
            r"physics|chemistry|maths|cs|computer|commerce|economics|"
            r"english|history|zoology|biochem|micro|social work|"
            r"visual comm|statistics|french|hindi|tamil)\b",
            q,
        )
        if specific_fee:
            # Specific programme fee → let RAG try; fallback to disclaimer
            return None
        c = facts.get("fees", {})
        return (
            f"Fee structures at Loyola College change annually. "
            f"{c.get('description', '')} "
            f"Contact: {c.get('contact', 'the college accounts office')}."
        )

    # ── Contact / location ────────────────────────────────────────────
    if any(k in q for k in ["address", "location", "where is loyola", "phone", "email", "website", "contact"]):
        c = facts.get("contact", {})
        return (
            f"**Loyola College Contact Details:**\n"
            f"- Address: {c.get('address', 'N/A')}\n"
            f"- Phone: {c.get('phone', 'N/A')}\n"
            f"- Email: {c.get('email', 'N/A')}\n"
            f"- Website: {c.get('website', 'N/A')}"
        )

    # ── Accreditation / ranking ───────────────────────────────────────
    if any(k in q for k in ["naac", "accreditat", "ranking", "grade", "autonomous"]):
        return (
            f"Loyola College holds **{facts.get('accreditation', 'N/A')}** accreditation. "
            f"It received autonomous status in {facts.get('affiliation', 'N/A')}."
        )

    # ── Establishment ─────────────────────────────────────────────────
    if any(k in q for k in ["established", "founded", "year of", "when was loyola"]):
        return f"Loyola College was established in **{facts.get('established', 'N/A')}**."

    # ── Courses / programmes ──────────────────────────────────────────
    # Guard: only fire for generic listing questions, not specific queries
    # about a single programme's details (fee, syllabus, seats, etc.).
    _SPECIFIC_DETAIL = re.compile(
        r"\b(syllabus|seat|intake|eligib|cutoff|duration|year|semester|"
        r"subject|credit|detail|structure|scope|career|job|salary|fee|cost)\b",
        re.I,
    )
    if any(k in q for k in [
        "course", "programme", "program", "ug ", "pg ", "degree",
        "undergraduate", "postgraduate", "what can i study", "what do you offer",
        "list of", "available courses", "all courses",
    ]) and not _SPECIFIC_DETAIL.search(q):
        programmes = facts.get("programmes", {})
        if programmes:
            lines = ["**Programmes offered at Loyola College:**\n"]
            for level, prog_list in programmes.items():
                lines.append(f"\n**{level}**")
                for prog in prog_list:
                    lines.append(f"- {prog}")
            return "\n".join(lines)
        depts = facts.get("departments", [])
        if depts:
            return (
                "**Departments at Loyola College:**\n"
                + "\n".join(f"- {d}" for d in depts)
            )

    # ── Departments ───────────────────────────────────────────────────
    if any(k in q for k in ["department", "stream", "what subjects", "which departments"]):
        depts = facts.get("departments", [])
        if depts:
            return (
                "**Departments at Loyola College:**\n"
                + "\n".join(f"- {d}" for d in depts)
            )

    return None


# ==========================================================================================
# QUERY EXPANSION  — NEW v9
# ==========================================================================================
# Injects synonyms and Loyola-specific terminology into the search query to
# maximise lexical recall in BM25 and improve dense retrieval coverage.

_EXPANSION_MAP = {
    # Placement / career
    r"\bplacement\b":         ["recruitment", "campus hiring", "job offers", "career"],
    r"\bsalary\b":            ["package", "CTC", "compensation", "pay"],
    r"\brecruiter\b":         ["company", "recruiter", "employer", "hiring firm"],

    # Academics
    r"\bcourse\b":            ["programme", "degree", "subject", "curriculum"],
    r"\badmission\b":         ["enrolment", "joining", "application", "eligibility"],
    r"\beligibilit\b":        ["criteria", "requirement", "cutoff", "qualification"],
    r"\bfee\b":               ["tuition", "cost", "charges", "payment"],
    r"\bscholarship\b":       ["financial aid", "bursary", "stipend", "grant"],

    # Infrastructure
    r"\bhostel\b":            ["accommodation", "residence", "dormitory", "boarding"],
    r"\blibrary\b":           ["books", "journals", "reading room", "digital library"],
    r"\blab\b":               ["laboratory", "research facility", "equipment"],

    # Rankings / quality
    r"\branking\b":           ["nirf", "naac", "grade", "position", "accreditation"],
    r"\bfacult\b":            ["professor", "teacher", "staff", "lecturer"],

    # Loyola-specific shorthand
    r"\bloc\b":               ["loyola college", "loyola"],
    r"\bloyola\b":            ["loyola college chennai", "loyola autonomous"],
}

# College-specific prefix always appended for context anchoring
_COLLEGE_PREFIX = "Loyola College Chennai"


def expand_query(query: str) -> str:
    """
    Return an expanded version of the query with synonyms and college context.
    The original query is preserved; expansions are appended as additional terms.
    """
    extra_terms: list[str] = []

    for pattern, synonyms in _EXPANSION_MAP.items():
        if re.search(pattern, query, re.I):
            extra_terms.extend(synonyms)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_extras: list[str] = []
    for t in extra_terms:
        tl = t.lower()
        if tl not in seen:
            seen.add(tl)
            unique_extras.append(t)

    if unique_extras:
        expansion = " ".join(unique_extras[:6])   # cap at 6 extra terms
        return f"{_COLLEGE_PREFIX} {query} {expansion}"
    return f"{_COLLEGE_PREFIX} {query}"


# ==========================================================================================
# DOCUMENT LOADERS
# ==========================================================================================

def load_rag_json_folder(log) -> list[Document]:
    folder = Path(CFG.RAG_JSON_FOLDER)
    docs   = []
    if not folder.exists():
        return docs
    files = sorted(folder.glob("*.json"))
    log.write(f"📄 RAG JSON: {len(files)} files")
    for fpath in files:
        try:
            raw  = json.loads(fpath.read_text(encoding="utf-8"))
            text = raw.get("page_content", "").strip()
            meta = raw.get("metadata", {})
            if len(text) < 100:
                continue
            source  = meta.get("url") or meta.get("source") or str(fpath)
            section = meta.get("section") or extract_section_from_source(source)
            docs.append(Document(
                page_content = clean_text(text),
                metadata = {
                    "source" : source,
                    "title"  : meta.get("title", ""),
                    "type"   : "website",
                    "section": section,
                },
            ))
        except Exception as e:
            print(f"[JSON ERROR] {fpath.name}: {e}")
    log.success(f"✅ RAG JSON: {len(docs)} pages loaded")
    return docs


def load_text_folder(folder_path: str, label: str, log) -> list[Document]:
    folder = Path(folder_path)
    docs   = []
    if not folder.exists():
        return docs
    files = sorted(folder.glob("*.txt"))
    log.write(f"📝 {label}: {len(files)} files")
    for fpath in files:
        try:
            text = fpath.read_text(encoding="utf-8", errors="replace").strip()
            if len(text) < 100:
                continue
            source  = str(fpath)
            section = extract_section_from_source(source)
            docs.append(Document(
                page_content = clean_text(text),
                metadata = {
                    "source" : source,
                    "title"  : fpath.stem,
                    "type"   : label.lower(),
                    "section": section,
                },
            ))
        except Exception as e:
            print(f"[TEXT ERROR] {fpath.name}: {e}")
    log.success(f"✅ {label}: {len(docs)} loaded")
    return docs


def load_pdf(path: str) -> list[Document]:
    for Loader in [PyMuPDFLoader, PDFPlumberLoader]:
        try:
            raw  = Loader(path).load()
            docs = [
                Document(
                    page_content = clean_text(d.page_content),
                    metadata = {
                        "source" : path,
                        "page"   : d.metadata.get("page", "N/A"),
                        "type"   : "pdf",
                        "title"  : Path(path).name,
                        "section": "pdf",
                    },
                )
                for d in raw if len(d.page_content.strip()) > 50
            ]
            if docs:
                return docs
        except Exception:
            continue
    return []


def load_pdf_folder(log) -> list[Document]:
    folder = Path(CFG.PDF_FOLDER)
    docs   = []
    if not folder.exists():
        return docs
    files = sorted(folder.glob("*.pdf"))
    log.write(f"📚 PDFs: {len(files)} files")
    bar = log.progress(0)
    for i, fpath in enumerate(files):
        try:
            docs.extend(load_pdf(str(fpath)))
            bar.progress((i + 1) / max(len(files), 1))
        except Exception as e:
            print(f"[PDF ERROR] {fpath.name}: {e}")
    log.success(f"✅ PDFs: {len(docs)} pages loaded")
    return docs


def load_all_documents(sidebar_log) -> list[Document]:
    all_docs  = []
    seen_keys = set()

    def _add(docs):
        for d in docs:
            key = d.page_content[:300]
            if key not in seen_keys:
                seen_keys.add(key)
                all_docs.append(d)

    s1 = sidebar_log.empty()
    s2 = sidebar_log.empty()
    s3 = sidebar_log.empty()
    s4 = sidebar_log.empty()

    _add(load_rag_json_folder(s1))
    _add(load_text_folder(CFG.CLEANED_TEXT_FOLDER, "Cleaned Text", s2))
    _add(load_text_folder(CFG.OCR_TEXT_FOLDER,     "OCR Text",     s3))
    _add(load_pdf_folder(s4))

    sidebar_log.success(f"🗂 Total unique docs: {len(all_docs)}")
    return all_docs


# ==========================================================================================
# CHUNKING
# ==========================================================================================

def create_chunks(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = CFG.CHUNK_SIZE,
        chunk_overlap = CFG.CHUNK_OVERLAP,
        separators    = ["\n\n\n", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    return [
        c for c in chunks
        if len(c.page_content.strip()) > 80 and not is_noise_chunk(c.page_content)
    ]


# ==========================================================================================
# MODELS
# ==========================================================================================

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name    = CFG.EMBEDDING_MODEL,
        model_kwargs  = {"device": DEVICE},
        encode_kwargs = {"normalize_embeddings": True},
    )


@st.cache_resource(show_spinner=False)
def load_reranker():
    return CrossEncoder(CFG.RERANK_MODEL, device=DEVICE)


@st.cache_resource(show_spinner=False)
def load_sentence_model():
    return SentenceTransformer(CFG.EMBEDDING_MODEL, device=DEVICE)


# ==========================================================================================
# KNOWLEDGE BASE
# ==========================================================================================

def build_knowledge_base(sidebar_log, emb):

    if can_fast_load():
        sidebar_log.info("⚡ Fast loading from saved index...")
        t0     = time.time()
        chunks = pickle.load(open(CFG.CHUNKS_PKL, "rb"))
        bm25   = pickle.load(open(CFG.BM25_PKL,   "rb"))
        db     = FAISS.load_local(
            CFG.FAISS_PATH, emb,
            allow_dangerous_deserialization=True,
        )
        elapsed = round(time.time() - t0, 1)
        sidebar_log.success(f"✅ {len(chunks):,} chunks loaded in {elapsed}s")
        return db, chunks, bm25

    sidebar_log.warning("🔄 Building knowledge base — please wait...")
    t0   = time.time()
    docs = load_all_documents(sidebar_log)

    if not docs:
        st.error("❌ No documents found. Check your data/ folder.")
        st.stop()

    sidebar_log.write(f"✂️ Chunking {len(docs):,} documents...")
    chunks = create_chunks(docs)
    sidebar_log.write(f"📦 {len(chunks):,} chunks created")
    sidebar_log.write("🧠 Building FAISS + BM25 indices...")

    for f in ["index.faiss", "index.pkl"]:
        fp = Path(CFG.FAISS_PATH) / f
        if fp.exists():
            fp.unlink()

    db   = FAISS.from_documents(chunks, emb)
    db.save_local(CFG.FAISS_PATH)

    bm25   = BM25Retriever.from_documents(chunks)
    bm25.k = CFG.RETRIEVAL_K_COMPARE

    pickle.dump(chunks, open(CFG.CHUNKS_PKL, "wb"))
    pickle.dump(bm25,   open(CFG.BM25_PKL,   "wb"))
    save_data_hash()

    elapsed = round(time.time() - t0, 1)
    sidebar_log.success(f"✅ Built {len(chunks):,} chunks in {elapsed}s")
    sidebar_log.info("⚡ Next run will load in ~5 seconds")
    return db, chunks, bm25


# ==========================================================================================
# QUERY CLASSIFICATION  — v9: 8 classes
# ==========================================================================================

_FEE_RE         = re.compile(r"\bfee|tuition|cost|how much|payment\b", re.I)
_ELIGIBILITY_RE = re.compile(r"\beligib|qualify|cutoff|criteria|requirement|admission\b", re.I)
_DATE_RE        = re.compile(r"\bdate|deadline|when|schedule|calendar|exam date\b", re.I)
_FACT_RE        = re.compile(r"\bwho is|what is|where is|located|principal|founded|established\b", re.I)
_FOLLOWUP_RE    = re.compile(r"^(what about|how about|and|also|tell me more|explain|elaborate|go on|continue)\b", re.I)
_COMPARISON_RE  = re.compile(
    r"\b(compare|versus|vs\.?|difference between|better|worse|which is|pros and cons|"
    r"advantages|disadvantages|contrast|similarities)\b",
    re.I,
)

_STATS_RE = re.compile(
    r"\b("
    r"how many students?|total students?|student strength|student count|student enrolment|"
    r"placement stat|placement record|placed students?|salary package|average package|"
    r"pass percentage|pass rate|result percentage|"
    r"faculty count|number of (faculty|staff|teachers?)|"
    r"college ranking|nirf rank|number of departments?|programmes? offered|"
    r"total intake|sanctioned strength|hostel capacity"
    r")\b",
    re.I,
)


def classify_query(query: str, has_history: bool) -> str:
    """
    Classify query into one of 8 categories.
    Order matters — more specific classes checked first.
    """
    q = query.strip()

    if has_history and (_FOLLOWUP_RE.search(q) or len(q.split()) <= 5):
        return "followup"

    if _COMPARISON_RE.search(q):
        return "comparison"

    if _STATS_RE.search(q):
        return "statistics"

    if _FEE_RE.search(q):
        return "fee"

    if _ELIGIBILITY_RE.search(q):
        return "eligibility"

    if _DATE_RE.search(q):
        return "date"

    if _FACT_RE.search(q):
        return "fact"

    return "general"


def get_retrieval_k(q_class: str) -> int:
    """Dynamically scale retrieval K based on query complexity."""
    mapping = {
        "fact"       : CFG.RETRIEVAL_K_FACT,
        "fee"        : CFG.RETRIEVAL_K_FACT,
        "followup"   : CFG.RETRIEVAL_K_DEFAULT,
        "general"    : CFG.RETRIEVAL_K_DEFAULT,
        "statistics" : CFG.RETRIEVAL_K_DEEP,
        "eligibility": CFG.RETRIEVAL_K_DEEP,
        "date"       : CFG.RETRIEVAL_K_DEEP,
        "comparison" : CFG.RETRIEVAL_K_COMPARE,
    }
    return mapping.get(q_class, CFG.RETRIEVAL_K_DEFAULT)


# ==========================================================================================
# HISTORY-AWARE QUERY REWRITING  — v9: all ambiguous queries rewritten
# ==========================================================================================

def rewrite_query(query: str, history: list[dict], llm, q_class: str) -> str:
    """
    Rewrite the query into a clear, standalone, searchable form.
    For follow-up queries: resolve pronouns / references using history.
    For all other classes: expand abbreviations and add college context.
    """
    # ── Follow-up: resolve pronouns and elliptical references ─────────
    if q_class == "followup" and history:
        recent       = history[-2:]
        history_text = "".join(
            f"Student: {h['q']}\nAssistant: {h['a'][:200]}...\n\n"
            for h in recent
        )
        prompt = (
            "You are a query rewriter for a college information chatbot.\n"
            "Rewrite the follow-up question as a complete, standalone question.\n\n"
            "Rules:\n"
            "- Output ONLY the rewritten question. Nothing else.\n"
            "- Keep it under 25 words.\n"
            "- If already standalone, return unchanged.\n\n"
            f"Conversation history:\n{history_text}"
            f"Follow-up question: {query}\n\n"
            "Standalone question:"
        )
        try:
            result = ""
            for chunk in llm.stream(prompt):
                result += chunk.content
            rewritten = result.strip().strip('"').strip("'")
            if 5 < len(rewritten) < 200:
                return rewritten
        except Exception:
            pass
        return query

    # ── For short / ambiguous non-follow-up queries: expand silently ──
    if len(query.split()) <= 3:
        return f"Loyola College {query} information"

    return query


# ==========================================================================================
# RRF HYBRID RETRIEVER
# ==========================================================================================

class HybridRRFRetriever:
    """FAISS (dense) + BM25 (sparse) fused with Reciprocal Rank Fusion."""

    def __init__(self, db, bm25: BM25Retriever):
        self._db   = db
        self._bm25 = bm25

    def _dense_retrieve(self, query: str, k: int) -> list[Document]:
        retriever = self._db.as_retriever(
            search_type   = "mmr",
            search_kwargs = {"k": k, "fetch_k": k * 2},
        )
        return retriever.invoke(query)

    def _sparse_retrieve(self, query: str, k: int) -> list[Document]:
        self._bm25.k = k
        return self._bm25.invoke(query)

    def invoke(
        self,
        query      : str,
        k          : int       = CFG.RETRIEVAL_K_DEFAULT,
        type_filter: list[str] = None,
    ) -> list[Document]:
        dense_docs  = self._dense_retrieve(query, k)
        sparse_docs = self._sparse_retrieve(query, k)

        doc_map    : dict[str, Document] = {}
        rrf_scores : dict[str, float]   = {}

        def _key(d: Document) -> str:
            return d.page_content[:200]

        for rank, doc in enumerate(dense_docs):
            key = _key(doc)
            doc_map[key]    = doc
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (CFG.RRF_K + rank + 1)

        for rank, doc in enumerate(sparse_docs):
            key = _key(doc)
            doc_map[key]    = doc
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (CFG.RRF_K + rank + 1)

        sorted_keys = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
        results     = [doc_map[kk] for kk in sorted_keys]

        if type_filter:
            results = [
                d for d in results
                if d.metadata.get("type", "").lower() in [t.lower() for t in type_filter]
            ]

        return results[:k]


# ==========================================================================================
# SOURCE PRIORITY BOOST
# ==========================================================================================

_PRIORITY_PATTERNS = [
    re.compile(r'placement|annual.?report|brochure|prospectus', re.I),
    re.compile(r'naac|nirf|iqac|accreditat|handbook|admission.?guide', re.I),
    re.compile(r'statistical|census|highlights|achievements|ranking', re.I),
]


def _source_priority_score(doc: Document) -> float:
    combined = (
        doc.metadata.get("source", "") + " " +
        doc.metadata.get("title",  "")
    ).lower()
    for p in _PRIORITY_PATTERNS:
        if p.search(combined):
            return 0.30
    if doc.metadata.get("type") == "pdf":
        return 0.15
    return 0.0


# ==========================================================================================
# RERANKER  — v9: larger candidate pool + adaptive threshold
# ==========================================================================================

def _adaptive_rerank_threshold(q_class: str, base: float = CFG.RERANK_THRESHOLD) -> float:
    """
    Lower the threshold for complex queries so more potentially-relevant
    chunks survive; raise it for simple facts to keep context tight.
    """
    adjustments = {
        "fact"       :  0.05,   # tighter — only high-confidence chunks
        "fee"        :  0.05,
        "comparison" : -0.05,   # looser — need breadth
        "statistics" : -0.05,
        "eligibility": -0.03,
    }
    return base + adjustments.get(q_class, 0.0)


def rerank(
    query  : str,
    docs   : list[Document],
    model,
    q_class: str  = "general",
    debug  : bool = False,
) -> list[tuple[float, Document]]:
    """
    Score = CrossEncoder(query, chunk) + source_priority_boost.
    Returns (boosted_score, Document) sorted descending, capped at RERANK_K.
    """
    if not docs:
        return []

    # Larger candidate pool in v9
    candidates = docs[:CFG.RERANK_CANDIDATE_POOL]
    pairs      = [[query, d.page_content] for d in candidates]
    raw_scores = model.predict(pairs, batch_size=16)

    boosted = [
        float(s) + _source_priority_score(d)
        for s, d in zip(raw_scores, candidates)
    ]

    ranked    = sorted(zip(boosted, candidates), key=lambda x: x[0], reverse=True)
    threshold = _adaptive_rerank_threshold(q_class)
    filtered  = [(s, d) for s, d in ranked if s > threshold]

    if not filtered:
        filtered = ranked[:2]   # always return at least 2

    if debug:
        print(f"\n[RERANK DEBUG] class={q_class} threshold={threshold:.3f} Top-10:")
        for i, (s, d) in enumerate(ranked[:10]):
            src     = d.metadata.get("source", "?")[:70]
            snippet = d.page_content[:100].strip().replace("\n", " ")
            print(f"  [{i+1}] score={s:.4f}  src={src}")
            print(f"       {snippet!r}")

    return filtered[:CFG.RERANK_K]


# ==========================================================================================
# SEMANTIC CACHE  — v9: embedding storage, no recompute, LRU eviction
# ==========================================================================================

def semantic_cache_lookup(
    query    : str,
    cache    : "collections.OrderedDict",
    sent_model,
    threshold: float = CFG.SEMANTIC_CACHE_THRESHOLD,
) -> Optional[dict]:
    """
    v9 improvements:
    - Each cache entry stores its own embedding under key '__emb__'.
    - We batch-compare query against stored embeddings (no re-encoding cached queries).
    - Uses efficient dot-product cosine (vectors pre-normalised).
    """
    if not cache:
        return None
    try:
        q_emb = sent_model.encode(query, normalize_embeddings=True)

        # Collect keys and their pre-stored embeddings
        keys_with_emb = [
            (k, v["__emb__"])
            for k, v in cache.items()
            if isinstance(v, dict) and "__emb__" in v
        ]

        if not keys_with_emb:
            # Fallback for old-format entries (no __emb__): re-encode and upgrade
            all_keys  = list(cache.keys())
            key_embs  = sent_model.encode(all_keys, normalize_embeddings=True, batch_size=64)
            sims      = key_embs @ q_emb
            best_idx  = int(np.argmax(sims))
            if sims[best_idx] >= threshold:
                hit = cache[all_keys[best_idx]]
                return hit if isinstance(hit, dict) else None
            return None

        cached_keys = [k for k, _ in keys_with_emb]
        emb_matrix  = np.array([e for _, e in keys_with_emb])
        sims        = emb_matrix @ q_emb
        best_idx    = int(np.argmax(sims))

        if sims[best_idx] >= threshold:
            hit = cache[cached_keys[best_idx]]
            cache.move_to_end(cached_keys[best_idx])   # LRU touch
            # Return a copy without the internal embedding key
            return {k: v for k, v in hit.items() if k != "__emb__"}

    except Exception:
        pass
    return None


def semantic_cache_store(
    cache    : "collections.OrderedDict",
    key      : str,
    result   : dict,
    sent_model,
) -> None:
    """Store result with its embedding for O(1) similarity lookups later."""
    try:
        emb            = sent_model.encode(key, normalize_embeddings=True)
        entry          = dict(result)
        entry["__emb__"] = emb
        _cache_set(cache, key, entry)
    except Exception:
        _cache_set(cache, key, result)


# ==========================================================================================
# HALLUCINATION GROUNDING  — v9: compare against ALL chunks, not first 512 chars
# ==========================================================================================

def compute_grounding_score(
    answer     : str,
    scored_docs: list[tuple[float, Document]],
    sent_model,
) -> float:
    """
    v9: Encode the answer and ALL retrieved chunk texts.
    Return the MAXIMUM cosine similarity across chunks (most-grounded chunk wins).
    This is more reliable than comparing against a truncated context string.
    """
    if not scored_docs:
        return 1.0
    try:
        chunk_texts = [d.page_content[:512] for _, d in scored_docs]
        all_texts   = [answer[:512]] + chunk_texts
        # Batch encode in one call — no repeated computations
        embs = sent_model.encode(all_texts, normalize_embeddings=True, batch_size=32)
        answer_emb  = embs[0]
        chunk_embs  = embs[1:]
        sims        = chunk_embs @ answer_emb
        return float(np.max(sims))
    except Exception:
        return 1.0


def confidence_label(score: float) -> tuple[str, str]:
    if score >= CFG.CONFIDENCE_HIGH:
        return "High", "🟢"
    if score >= CFG.CONFIDENCE_MED:
        return "Medium", "🟡"
    return "Low", "🔴"


# ==========================================================================================
# FOLLOW-UP SUGGESTIONS  — v9: strictly grounded, no hallucinations
# ==========================================================================================

def generate_followups(
    query   : str,
    answer  : str,
    llm,
    context : str = "",
) -> list[str]:
    """
    Generate 3 follow-up questions grounded in the retrieved context.
    Returns an empty list when the answer is the fallback phrase or on any error.
    """
    if "could not find this information" in answer.lower():
        return []

    # Use a richer context excerpt in v9 (500 chars instead of 300)
    context_snippet = context[:500].strip() if context else ""
    grounding_note  = (
        f"\nOnly suggest questions directly answerable from this context excerpt:\n"
        f'"""{context_snippet}"""'
        if context_snippet else ""
    )

    prompt = (
        "You are a helpful assistant for Loyola College, Chennai.\n"
        "Suggest exactly 3 short follow-up questions a student might ask, "
        "based ONLY on the answer and context below.\n"
        "Each question MUST be answerable from the same topic/document as the answer.\n"
        "Do NOT suggest questions about unrelated topics or invent new facts."
        f"{grounding_note}\n\n"
        f"Question: {query}\n"
        f"Answer: {answer[:400]}\n\n"
        "Output ONLY a JSON array of 3 question strings. No explanation. No markdown.\n"
        'Example: ["Question one?", "Question two?", "Question three?"]\n\n'
        "JSON array:"
    )

    try:
        result = ""
        for chunk in llm.stream(prompt):
            result += chunk.content
        match = re.search(r"\[.*?\]", result, re.DOTALL)
        if match:
            suggestions = json.loads(match.group())
            if isinstance(suggestions, list):
                cleaned = [s.strip() for s in suggestions if isinstance(s, str) and len(s) > 5]
                return cleaned[:3]
    except Exception:
        pass
    return []


# ==========================================================================================
# CONTEXT BUILDER  — v9: dedup, diversity filter, relevance-only
# ==========================================================================================

def _chunk_fingerprint(text: str) -> str:
    """Short fingerprint for near-duplicate detection."""
    words = re.sub(r"\s+", " ", text.lower()).split()
    return " ".join(words[:30])


# Regex to detect stat-category labels so we can deduplicate conflicting numbers
_STAT_CATEGORY_RE = re.compile(
    r"\b(total students?|student (strength|count|enrolment)|"
    r"total faculty|faculty (count|strength)|teaching staff|"
    r"non.?teaching staff|total staff|hostel capacity|intake|"
    r"pass (percentage|rate)|placement (percentage|rate)|placed students?)\b",
    re.I,
)


def _stat_category(text: str) -> Optional[str]:
    """Return a normalised stat-category label if found, else None."""
    m = _STAT_CATEGORY_RE.search(text)
    if m:
        return re.sub(r"\s+", "_", m.group().lower().strip())
    return None


def build_context(
    scored_docs: list[tuple[float, Document]],
) -> tuple[str, list[str], list[dict]]:
    """
    v10 improvements over v9:
    - Numeric contradiction filter: for each stat category (e.g. "total students"),
      only the highest-scoring chunk is kept; lower-scored contradicting chunks are
      dropped to prevent the LLM from listing conflicting figures.

    v9 improvements carried forward:
    - Fingerprint-based near-duplicate detection.
    - Diversity cap (max 2 chunks per source).

    Returns:
        context_str   — text block for the LLM prompt
        sources       — list[str] of raw source paths/URLs
        rich_sources  — list[dict] with full attribution data for UI
    """
    parts, sources, rich_sources = [], [], []
    tokens        = 0
    seen_fps      = set()
    source_counts : dict[str, int] = {}
    seen_stat_cats: set[str]       = set()   # BUG FIX 3: one chunk per stat category
    MAX_PER_SOURCE = 2

    for score, d in scored_docs:
        # Near-duplicate check
        fp = _chunk_fingerprint(d.page_content)
        if fp in seen_fps:
            continue
        seen_fps.add(fp)

        # BUG FIX 3: numeric contradiction filter
        # If this chunk discusses a stat category already covered by a
        # higher-scored chunk, skip it to avoid contradictory numbers.
        cat = _stat_category(d.page_content)
        if cat:
            if cat in seen_stat_cats:
                continue        # already have a better chunk for this category
            seen_stat_cats.add(cat)

        # Diversity cap per source
        src_key = d.metadata.get("source", "")
        if source_counts.get(src_key, 0) >= MAX_PER_SOURCE:
            continue
        source_counts[src_key] = source_counts.get(src_key, 0) + 1

        txt = d.page_content[:2500]
        t   = count_tokens(txt)
        if tokens + t > CFG.MAX_CONTEXT_TOKENS:
            break

        source  = d.metadata.get("source", "")
        title   = d.metadata.get("title", "")
        page    = d.metadata.get("page", "")
        dtype   = d.metadata.get("type", "")
        section = d.metadata.get("section", "")

        header = f"[SOURCE: {title or source}"
        if page and str(page) != "N/A":
            header += f" | Page {page}"
        header += "]"

        parts.append(f"{header}\n{txt}")
        tokens += t

        if source and source not in sources:
            sources.append(source)
            rich_sources.append({
                "source"       : source,
                "title"        : title or source,
                "page"         : str(page) if page and str(page) != "N/A" else "",
                "type"         : dtype,
                "section"      : section,
                "rerank_score" : round(float(score), 3),
            })

    return "\n\n---\n\n".join(parts), sources, rich_sources


# ==========================================================================================
# LLM
# ==========================================================================================

def get_llm(api_key: str):
    return ChatGroq(
        groq_api_key = api_key,
        model_name   = CFG.LLM_MODEL,
        temperature  = 0.1,
        streaming    = True,
    )


# ==========================================================================================
# PROMPT  — v9: stronger hallucination prevention
# ==========================================================================================

def build_prompt(context: str, question: str, history: list[dict], q_class: str) -> str:
    history_text = ""
    for h in history[-CFG.HISTORY_TURNS:]:
        history_text += f"Student: {h['q']}\nAssistant: {h['a']}\n\n"

    class_hints = {
        "statistics" : (
            "Look for college-wide totals, not department-specific sub-numbers. "
            "If only department figures are present, state that a college-wide total was not found. "
            "Always include the year or source of the figure."
        ),
        "fee"        : "Report exact fee amounts and payment schedules only if present in context. Never estimate.",
        "eligibility": "State admission criteria, cutoff marks, and required documents exactly as written.",
        "date"       : "State exact dates and deadlines only. Do not estimate or approximate.",
        "fact"       : "Give a single direct factual answer in one or two sentences.",
        "comparison" : (
            "Use a markdown table to compare attributes across items. "
            "Only compare attributes explicitly mentioned in the context."
        ),
        "followup"   : "Continue consistently with the conversation history above.",
        "general"    : "Give a complete, accurate answer using only what is in the context.",
    }
    hint = class_hints.get(q_class, class_hints["general"])

    return f"""You are the official AI assistant for Loyola College (Autonomous), Chennai, Tamil Nadu.

ABSOLUTE RULES — obey every rule without exception:
1. Answer ONLY using facts explicitly stated in the CONTEXT block below. Nothing else.
2. Do NOT use your own training knowledge, general world knowledge, or any external information.
3. Do NOT combine figures or facts from different, unrelated sections of the context.
4. Do NOT infer, estimate, approximate, or generate any fact not written word-for-word in the context.
5. Do NOT mention URLs, email addresses, or phone numbers unless the user explicitly asked for them.
6. If the answer is not clearly present in the context, output this sentence exactly and nothing else:
   "I could not find this information in the knowledge base."
7. FORMAT RULES:
   – Use bullet points (–) for lists of 3 or more items.
   – Use a markdown table when comparing multiple attributes across multiple items.
   – Use plain sentences for a single fact or a two-item answer.
8. Maximum 6 sentences or 10 bullet points. Be direct and specific.
9. Do NOT use meta-phrases like "Based on the context", "According to the document", "However", "Additionally".
10. Do NOT repeat, rephrase, or acknowledge the question.
11. Do NOT say "I don't know" or "I'm not sure" — use the exact fallback phrase from Rule 6 instead.
12. {hint}
13. If two context chunks give different numbers for the same fact, report ONLY the figure from
    the highest-scored chunk (listed first in the context) and do NOT mention the discrepancy.

CONVERSATION HISTORY:
{history_text}
====== CONTEXT — use ONLY this ======
{context}
=====================================

Question: {question}

Answer:"""


# ==========================================================================================
# MAIN ASK FUNCTION
# ==========================================================================================

def ask(
    query      : str,
    retriever  : HybridRRFRetriever,
    reranker,
    llm_mdl,
    sent_model,
    history    : list[dict],
    cache      : "collections.OrderedDict",
    facts      : dict,
    type_filter: list[str] = None,
    debug      : bool      = False,
) -> dict:
    """
    Full RAG pipeline. Returns:
    {
        answer, sources, rich_sources,
        confidence, conf_emoji, score,
        followups, cache_hit, q_class,
        facts_hit, log_id,
    }
    """
    t_start = time.time()

    # ── Security ─────────────────────────────────────────────────────
    if is_malicious(query):
        return {
            "answer": "⚠️ I can't process that request.", "sources": [],
            "rich_sources": [], "confidence": "N/A", "conf_emoji": "🔴",
            "score": 0.0, "followups": [], "cache_hit": False,
            "q_class": "blocked", "facts_hit": False, "log_id": None,
        }

    q_class     = classify_query(query, has_history=bool(history))
    retrieval_k = get_retrieval_k(q_class)

    # ── Facts layer ───────────────────────────────────────────────────
    fact_answer = query_facts_layer(query, facts)
    if fact_answer:
        lid = log_query(query, q_class, time.time() - t_start, False, "High", True, "facts")
        return {
            "answer"      : fact_answer,
            "sources"     : ["Loyola College — Official Facts"],
            "rich_sources": [{"source": "facts", "title": "Loyola College — Official Facts",
                               "page": "", "type": "facts", "section": "facts", "rerank_score": 1.0}],
            "confidence"  : "High",
            "conf_emoji"  : "🟢",
            "score"       : 1.0,
            "followups"   : generate_followups(query, fact_answer, llm_mdl),
            "cache_hit"   : False,
            "q_class"     : q_class,
            "facts_hit"   : True,
            "log_id"      : lid,
        }

    # ── Exact cache ───────────────────────────────────────────────────
    cache_key = query.lower().strip()
    if cache_key in cache:
        r   = cache[cache_key]
        lid = log_query(query, q_class, time.time() - t_start, True, r.get("confidence", "?"), True)
        cache.move_to_end(cache_key)
        return {**{k: v for k, v in r.items() if k != "__emb__"}, "cache_hit": True, "log_id": lid}

    # ── Semantic cache ────────────────────────────────────────────────
    sem_hit = semantic_cache_lookup(query, cache, sent_model)
    if sem_hit:
        lid = log_query(query, q_class, time.time() - t_start, True, sem_hit.get("confidence", "?"), True)
        return {**sem_hit, "cache_hit": True, "log_id": lid}

    # ── Query rewriting ───────────────────────────────────────────────
    effective_query = rewrite_query(query, history, llm_mdl, q_class)

    # ── Query expansion — NEW v9 ─────────────────────────────────────
    expanded_query = expand_query(effective_query)

    # ── Multi-query RRF retrieval ─────────────────────────────────────
    # Use expanded query for primary retrieval, original for secondary
    retrieval_queries = [expanded_query, effective_query]
    all_docs : dict[str, Document] = {}
    rrf_agg  : dict[str, float]   = {}

    for q in retrieval_queries:
        batch = retriever.invoke(q, k=retrieval_k, type_filter=type_filter)
        for rank, doc in enumerate(batch):
            key = doc.page_content[:200]
            all_docs[key] = doc
            rrf_agg[key]  = rrf_agg.get(key, 0.0) + 1.0 / (CFG.RRF_K + rank + 1)

    top_keys = sorted(rrf_agg, key=lambda x: rrf_agg[x], reverse=True)
    merged   = [all_docs[k] for k in top_keys]

    # ── Rerank with adaptive threshold ────────────────────────────────
    scored_docs = rerank(effective_query, merged, reranker, q_class=q_class, debug=debug)

    # ── Build context with dedup + diversity ──────────────────────────
    context, sources, rich_sources = build_context(scored_docs)

    if not context.strip():
        answer = "I could not find this information in the knowledge base."
        lid = log_query(query, q_class, time.time() - t_start, False, "Low", False)
        return {
            "answer": answer, "sources": [], "rich_sources": [],
            "confidence": "Low", "conf_emoji": "🔴", "score": 0.0,
            "followups": [], "cache_hit": False, "q_class": q_class,
            "facts_hit": False, "log_id": lid,
        }

    # ── LLM generation ────────────────────────────────────────────────
    prompt = build_prompt(context, effective_query, history, q_class)
    answer = ""
    for chunk in llm_mdl.stream(prompt):
        answer += chunk.content
    answer = answer.strip()

    # ── Hallucination grounding — v9: all chunks, max similarity ─────
    score = compute_grounding_score(answer, scored_docs, sent_model)
    conf_lbl, conf_emoji = confidence_label(score)

    if score < CFG.GROUNDING_THRESHOLD:
        answer += (
            "\n\n⚠️ *Low confidence: this answer may not be fully supported by "
            "the available information. Please verify at www.loyolacollege.edu.*"
        )

    # ── Follow-ups — grounded in retrieved context ────────────────────
    followups = generate_followups(query, answer, llm_mdl, context=context)

    top_section = rich_sources[0]["section"] if rich_sources else ""
    lid = log_query(query, q_class, time.time() - t_start, False, conf_lbl, True, top_section)

    result = {
        "answer"      : answer,
        "sources"     : sources,
        "rich_sources": rich_sources,
        "confidence"  : conf_lbl,
        "conf_emoji"  : conf_emoji,
        "score"       : round(score, 3),
        "followups"   : followups,
        "cache_hit"   : False,
        "q_class"     : q_class,
        "facts_hit"   : False,
        "log_id"      : lid,
    }

    # Store with embedding for fast future lookups
    semantic_cache_store(cache, cache_key, result, sent_model)
    _save_cache(cache)
    return result


# ==========================================================================================
# RETRIEVAL EVALUATION FRAMEWORK
# ==========================================================================================

def _load_eval_dataset() -> list[dict]:
    p = Path(CFG.EVAL_DATASET_FILE)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def _precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    hits = sum(1 for r in retrieved[:k] if any(rel in r for rel in relevant))
    return hits / k if k > 0 else 0.0


def _recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for r in retrieved[:k] if any(rel in r for rel in relevant))
    return hits / len(relevant)


def _mrr(retrieved: list[str], relevant: list[str]) -> float:
    for i, r in enumerate(retrieved):
        if any(rel in r for rel in relevant):
            return 1.0 / (i + 1)
    return 0.0


def _dcg(retrieved: list[str], relevant: list[str], k: int) -> float:
    dcg = 0.0
    for i, r in enumerate(retrieved[:k]):
        rel = 1.0 if any(rel in r for rel in relevant) else 0.0
        dcg += rel / math.log2(i + 2)
    return dcg


def _ndcg(retrieved: list[str], relevant: list[str], k: int) -> float:
    ideal = sorted([1.0] * min(len(relevant), k) + [0.0] * max(0, k - len(relevant)), reverse=True)
    idcg  = sum(v / math.log2(i + 2) for i, v in enumerate(ideal))
    if idcg == 0:
        return 0.0
    return _dcg(retrieved, relevant, k) / idcg


def run_retrieval_evaluation(
    retriever : HybridRRFRetriever,
    k         : int = 5,
    run_label : str = "",
) -> Optional[dict]:
    dataset = _load_eval_dataset()
    if not dataset:
        return None

    precisions, recalls, mrrs, ndcgs = [], [], [], []
    per_query_results = []

    for item in dataset:
        question = item.get("question", "")
        relevant = item.get("relevant_sources", [])
        if not question or not relevant:
            continue

        # Use expanded query for evaluation too
        expanded = expand_query(question)
        docs      = retriever.invoke(expanded, k=k)
        retrieved = [d.metadata.get("source", "") for d in docs]

        p_k = _precision_at_k(retrieved, relevant, k)
        r_k = _recall_at_k(retrieved, relevant, k)
        m   = _mrr(retrieved, relevant)
        n   = _ndcg(retrieved, relevant, k)

        precisions.append(p_k)
        recalls.append(r_k)
        mrrs.append(m)
        ndcgs.append(n)
        per_query_results.append({
            "question"   : question,
            "precision_k": round(p_k, 4),
            "recall_k"   : round(r_k, 4),
            "mrr"        : round(m, 4),
            "ndcg"       : round(n, 4),
        })

    if not precisions:
        return None

    metrics = {
        "run_label"       : run_label or datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "k"               : k,
        "precision_k"     : round(float(np.mean(precisions)), 4),
        "recall_k"        : round(float(np.mean(recalls)), 4),
        "mrr"             : round(float(np.mean(mrrs)), 4),
        "ndcg"            : round(float(np.mean(ndcgs)), 4),
        "num_queries"     : len(precisions),
        # v9: summary statistics
        "precision_std"   : round(float(np.std(precisions)), 4),
        "recall_std"      : round(float(np.std(recalls)), 4),
        "mrr_std"         : round(float(np.std(mrrs)), 4),
        "ndcg_std"        : round(float(np.std(ndcgs)), 4),
        "per_query"       : per_query_results,
    }

    try:
        conn = _init_db()
        conn.execute(
            """INSERT INTO eval_results
               (ts, run_label, k, precision_k, recall_k, mrr, ndcg, num_queries)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                datetime.datetime.now().isoformat(),
                metrics["run_label"], metrics["k"],
                metrics["precision_k"], metrics["recall_k"],
                metrics["mrr"], metrics["ndcg"], metrics["num_queries"],
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass

    return metrics


def metrics_to_csv(metrics_rows: list[dict]) -> str:
    if not metrics_rows:
        return ""
    buf    = io.StringIO()
    # Exclude per_query nested key for CSV export
    clean_rows = [{k: v for k, v in row.items() if k != "per_query"} for row in metrics_rows]
    writer = csv.DictWriter(buf, fieldnames=list(clean_rows[0].keys()))
    writer.writeheader()
    writer.writerows(clean_rows)
    return buf.getvalue()


# ==========================================================================================
# DEFAULT EVAL DATASET
# ==========================================================================================

_DEFAULT_EVAL_DATASET = [
    {"question": "How many students are enrolled in Loyola College?",
     "relevant_sources": ["loyolacollege", "students", "strength", "enrolment"]},
    {"question": "What is the NAAC accreditation grade of Loyola College?",
     "relevant_sources": ["naac", "accreditation", "grade", "loyolacollege"]},
    {"question": "What UG programmes does Loyola College offer?",
     "relevant_sources": ["loyolacollege", "undergraduate", "ug", "programmes"]},
    {"question": "What is the placement percentage at Loyola College?",
     "relevant_sources": ["placement", "loyolacollege", "recruited", "campus"]},
    {"question": "What is the average salary package offered during placements?",
     "relevant_sources": ["placement", "salary", "package", "loyolacollege"]},
    {"question": "How many faculty members are in Loyola College?",
     "relevant_sources": ["faculty", "staff", "teachers", "loyolacollege"]},
    {"question": "What is the NIRF ranking of Loyola College?",
     "relevant_sources": ["nirf", "ranking", "loyolacollege"]},
    {"question": "When was Loyola College established?",
     "relevant_sources": ["established", "founded", "1925", "loyolacollege"]},
    {"question": "What are the admission requirements for BSc Computer Science?",
     "relevant_sources": ["computer science", "admission", "eligibility", "loyolacollege"]},
    {"question": "What PG programmes does Loyola College offer?",
     "relevant_sources": ["postgraduate", "pg", "msc", "mcom", "loyolacollege"]},
    {"question": "What companies recruit from Loyola College?",
     "relevant_sources": ["placement", "recruiters", "companies", "loyolacollege"]},
    {"question": "What is the fee structure for BCA?",
     "relevant_sources": ["bca", "fee", "tuition", "loyolacollege"]},
    {"question": "What research facilities are available at Loyola College?",
     "relevant_sources": ["research", "laboratory", "facilities", "loyolacollege"]},
    {"question": "What is the hostel capacity of Loyola College?",
     "relevant_sources": ["hostel", "accommodation", "loyolacollege"]},
    {"question": "What scholarships are available at Loyola College?",
     "relevant_sources": ["scholarship", "financial aid", "loyolacollege"]},
    {"question": "Who is the principal of Loyola College?",
     "relevant_sources": ["principal", "loyolacollege", "administration"]},
    {"question": "What sports facilities are available at Loyola College?",
     "relevant_sources": ["sports", "gymnasium", "playground", "loyolacollege"]},
    {"question": "What are the library facilities at Loyola College?",
     "relevant_sources": ["library", "loyolacollege", "books", "journals"]},
    {"question": "What is the pass percentage in Loyola College university exams?",
     "relevant_sources": ["pass", "result", "percentage", "university", "loyolacollege"]},
    {"question": "What MBA specialisations are offered at Loyola College?",
     "relevant_sources": ["mba", "specialisation", "management", "loyolacollege"]},
    {"question": "Does Loyola College offer PhD programmes?",
     "relevant_sources": ["phd", "doctorate", "research", "loyolacollege"]},
    {"question": "What cultural events are held at Loyola College?",
     "relevant_sources": ["cultural", "fest", "events", "loyolacollege"]},
    {"question": "What is the intake capacity of BSc Physics?",
     "relevant_sources": ["physics", "intake", "seats", "loyolacollege"]},
    {"question": "What affiliation does Loyola College have?",
     "relevant_sources": ["university of madras", "autonomous", "affiliation", "loyolacollege"]},
    {"question": "What international collaborations does Loyola College have?",
     "relevant_sources": ["international", "collaboration", "mou", "loyolacollege"]},
]


def _ensure_eval_dataset() -> None:
    """Create the default evaluation dataset if the file does not exist. Silent."""
    p = Path(CFG.EVAL_DATASET_FILE)
    if p.exists():
        return
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(_DEFAULT_EVAL_DATASET, indent=2), encoding="utf-8")
    except Exception:
        pass


# ==========================================================================================
# RICH SOURCE CARD UI
# ==========================================================================================

def render_source_cards(rich_sources: list[dict]) -> None:
    if not rich_sources:
        return
    with st.expander(f"📎 Sources ({len(rich_sources)})"):
        for rs in rich_sources:
            title  = rs.get("title", "") or rs.get("source", "Unknown")
            source = rs.get("source", "")
            page   = rs.get("page", "")
            dtype  = rs.get("type", "")
            score  = rs.get("rerank_score", None)

            col_main, col_meta = st.columns([3, 1])

            with col_main:
                if source.startswith("http"):
                    st.markdown(f"**[{title}]({source})**")
                else:
                    st.markdown(f"**{title}**")
                    st.caption(source)

            with col_meta:
                badges = []
                if dtype:
                    badges.append(f"`{dtype}`")
                if page:
                    badges.append(f"p.{page}")
                if score is not None:
                    badges.append(f"score: `{score:.3f}`")
                st.markdown("  ".join(badges))

            st.divider()


# ==========================================================================================
# ANALYTICS TAB
# ==========================================================================================

def render_analytics_tab() -> None:
    st.header("📊 Query Analytics")
    try:
        conn  = _init_db()
        total = conn.execute("SELECT COUNT(*) FROM query_log").fetchone()[0]

        if total == 0:
            st.info("No queries logged yet.")
            conn.close()
            return

        avg_lat      = conn.execute("SELECT AVG(latency_s) FROM query_log").fetchone()[0]
        cache_pct    = conn.execute("SELECT 100.0*SUM(cache_hit)/COUNT(*) FROM query_log").fetchone()[0]
        answered_pct = conn.execute("SELECT 100.0*SUM(answered)/COUNT(*) FROM query_log").fetchone()[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total queries",  total)
        c2.metric("Avg latency",    f"{avg_lat:.2f}s" if avg_lat else "–")
        c3.metric("Cache hit rate", f"{cache_pct:.1f}%" if cache_pct else "0%")
        c4.metric("Answer rate",    f"{answered_pct:.1f}%" if answered_pct else "0%")

        st.divider()
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("📂 Query types")
            class_rows = conn.execute(
                "SELECT q_class, COUNT(*) as n FROM query_log GROUP BY q_class ORDER BY n DESC"
            ).fetchall()
            if class_rows:
                df_class = pd.DataFrame(class_rows, columns=["Query type", "Count"])
                st.bar_chart(df_class.set_index("Query type"))

        with col_right:
            st.subheader("🌐 Top website sections")
            sec_rows = conn.execute(
                """SELECT section, COUNT(*) as n FROM query_log
                   WHERE section IS NOT NULL AND section != ''
                   GROUP BY section ORDER BY n DESC LIMIT 10"""
            ).fetchall()
            if sec_rows:
                df_sec = pd.DataFrame(sec_rows, columns=["Section", "Count"])
                st.bar_chart(df_sec.set_index("Section"))
            else:
                st.caption("No section data yet.")

        st.divider()
        st.subheader("🔝 Top queries")
        rows = conn.execute(
            "SELECT query, COUNT(*) as n FROM query_log GROUP BY LOWER(query) ORDER BY n DESC LIMIT 10"
        ).fetchall()
        for q, n in rows:
            st.write(f"- ({n}×) {q}")

        st.divider()
        st.subheader("🕐 Recent queries")
        recent = conn.execute(
            """SELECT ts, query, q_class, section, latency_s, cache_hit, confidence
               FROM query_log ORDER BY id DESC LIMIT 25"""
        ).fetchall()
        df_recent = pd.DataFrame(
            recent,
            columns=["Timestamp", "Query", "Class", "Section", "Latency (s)", "Cache hit", "Confidence"],
        )
        st.dataframe(df_recent, use_container_width=True)
        conn.close()

    except Exception as e:
        st.warning(f"Analytics unavailable: {e}")


# ==========================================================================================
# FAILURE ANALYTICS TAB
# ==========================================================================================

def render_failures_tab() -> None:
    st.header("⚠️ Failure Analytics")
    try:
        conn       = _init_db()
        total      = conn.execute("SELECT COUNT(*) FROM query_log").fetchone()[0]
        unanswered = conn.execute("SELECT COUNT(*) FROM query_log WHERE answered=0").fetchone()[0]
        low_conf   = conn.execute(
            "SELECT COUNT(*) FROM query_log WHERE confidence='Low' AND answered=1"
        ).fetchone()[0]

        c1, c2, c3 = st.columns(3)
        c1.metric("Total queries", total)
        c2.metric("Unanswered",    unanswered,
                  delta=f"{100*unanswered/max(total,1):.1f}% failure rate", delta_color="inverse")
        c3.metric("Low-confidence", low_conf,
                  delta=f"{100*low_conf/max(total,1):.1f}% of queries", delta_color="inverse")

        st.divider()
        col_l, col_r = st.columns(2)

        with col_l:
            st.subheader("❌ Unanswered queries")
            rows = conn.execute(
                "SELECT query, ts FROM query_log WHERE answered=0 ORDER BY id DESC LIMIT 20"
            ).fetchall()
            if rows:
                st.dataframe(pd.DataFrame(rows, columns=["Query", "Timestamp"]), use_container_width=True)
            else:
                st.success("No unanswered queries yet!")

        with col_r:
            st.subheader("🟡 Low-confidence answers")
            rows = conn.execute(
                """SELECT query, section, ts FROM query_log
                   WHERE confidence='Low' AND answered=1 ORDER BY id DESC LIMIT 20"""
            ).fetchall()
            if rows:
                st.dataframe(pd.DataFrame(rows, columns=["Query", "Section", "Timestamp"]), use_container_width=True)
            else:
                st.success("No low-confidence answers logged yet.")

        st.divider()
        st.subheader("📉 Sections with low answer rate")
        sec_fail = conn.execute(
            """SELECT section,
                      COUNT(*) as total,
                      SUM(CASE WHEN answered=0 THEN 1 ELSE 0 END) as failed,
                      ROUND(100.0*SUM(CASE WHEN answered=0 THEN 1 ELSE 0 END)/COUNT(*),1) as pct
               FROM query_log
               WHERE section IS NOT NULL AND section != ''
               GROUP BY section HAVING total >= 2
               ORDER BY pct DESC LIMIT 10"""
        ).fetchall()
        if sec_fail:
            df_sf = pd.DataFrame(sec_fail, columns=["Section", "Total", "Failed", "Fail %"])
            st.dataframe(df_sf, use_container_width=True)
            st.bar_chart(df_sf.set_index("Section")["Fail %"])
        else:
            st.caption("Not enough section data yet.")

        conn.close()
    except Exception as e:
        st.warning(f"Failure analytics unavailable: {e}")


# ==========================================================================================
# FEEDBACK TAB
# ==========================================================================================

def render_feedback_tab() -> None:
    st.header("👍 User Feedback")
    try:
        conn     = _init_db()
        total_fb = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]

        if total_fb == 0:
            st.info("No feedback recorded yet.")
            conn.close()
            return

        helpful = conn.execute("SELECT COUNT(*) FROM feedback WHERE helpful=1").fetchone()[0]
        pct     = 100 * helpful / total_fb if total_fb else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Total feedback",    total_fb)
        c2.metric("👍 Helpful",        helpful)
        c3.metric("Satisfaction rate", f"{pct:.1f}%")

        st.divider()
        col_l, col_r = st.columns(2)

        with col_l:
            st.subheader("📅 Feedback over time")
            daily = conn.execute(
                """SELECT substr(ts,1,10) as day,
                          SUM(helpful) as helpful, SUM(1-helpful) as not_helpful
                   FROM feedback GROUP BY day ORDER BY day"""
            ).fetchall()
            if daily:
                st.bar_chart(pd.DataFrame(daily, columns=["Date", "Helpful", "Not helpful"]).set_index("Date"))

        with col_r:
            st.subheader("👎 Most disliked queries")
            disliked = conn.execute(
                """SELECT query, COUNT(*) as thumbs_down FROM feedback WHERE helpful=0
                   GROUP BY LOWER(query) ORDER BY thumbs_down DESC LIMIT 10"""
            ).fetchall()
            if disliked:
                st.dataframe(pd.DataFrame(disliked, columns=["Query", "Thumbs down"]), use_container_width=True)

        st.divider()
        st.subheader("🕐 Recent feedback")
        recent = conn.execute(
            """SELECT ts, query,
                      CASE helpful WHEN 1 THEN '👍' ELSE '👎' END as vote
               FROM feedback ORDER BY id DESC LIMIT 30"""
        ).fetchall()
        st.dataframe(pd.DataFrame(recent, columns=["Timestamp", "Query", "Vote"]), use_container_width=True)
        conn.close()

    except Exception as e:
        st.warning(f"Feedback analytics unavailable: {e}")


# ==========================================================================================
# EVALUATION TAB  — v9: per-query breakdown, summary stats, multi-run comparison
# ==========================================================================================

def render_evaluation_tab(retriever: HybridRRFRetriever) -> None:
    st.header("📐 Retrieval Evaluation")

    _ensure_eval_dataset()
    dataset = _load_eval_dataset()

    if not dataset:
        st.info("Evaluation dataset unavailable. Add `data/eval_dataset.json` to enable this tab.")
        return

    st.success(f"✅ Evaluation dataset — {len(dataset)} questions")

    col_k, col_label, col_run = st.columns([1, 2, 1])
    k         = col_k.selectbox("Top-K to evaluate", [3, 5, 10], index=1)
    run_label = col_label.text_input("Run label (optional)", placeholder="e.g. v9-baseline")
    run_now   = col_run.button("▶ Run evaluation", use_container_width=True)

    if run_now:
        with st.spinner(f"Evaluating {len(dataset)} queries at K={k}..."):
            metrics = run_retrieval_evaluation(retriever, k=k, run_label=run_label)

        if metrics:
            st.success("Evaluation complete!")

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric(f"Precision@{k}", f"{metrics['precision_k']:.4f}",
                       delta=f"±{metrics['precision_std']:.4f}")
            mc2.metric(f"Recall@{k}",    f"{metrics['recall_k']:.4f}",
                       delta=f"±{metrics['recall_std']:.4f}")
            mc3.metric("MRR",            f"{metrics['mrr']:.4f}",
                       delta=f"±{metrics['mrr_std']:.4f}")
            mc4.metric(f"NDCG@{k}",      f"{metrics['ndcg']:.4f}",
                       delta=f"±{metrics['ndcg_std']:.4f}")

            # Per-query breakdown
            if metrics.get("per_query"):
                with st.expander("📋 Per-query breakdown"):
                    df_pq = pd.DataFrame(metrics["per_query"])
                    st.dataframe(df_pq, use_container_width=True)

                    # Export per-query results
                    pq_csv = df_pq.to_csv(index=False)
                    st.download_button(
                        label     = "⬇️ Export per-query results as CSV",
                        data      = pq_csv,
                        file_name = f"loyola_eval_per_query_{run_label or 'run'}.csv",
                        mime      = "text/csv",
                    )
        else:
            st.error("Evaluation failed — check dataset format.")

    st.divider()
    st.subheader("📜 Evaluation history")

    try:
        conn = _init_db()
        rows = conn.execute(
            """SELECT ts, run_label, k, precision_k, recall_k, mrr, ndcg, num_queries
               FROM eval_results ORDER BY id DESC LIMIT 20"""
        ).fetchall()
        conn.close()

        if rows:
            cols    = ["Timestamp", "Run label", "K", "Precision@K", "Recall@K", "MRR", "NDCG", "Queries"]
            df_eval = pd.DataFrame(rows, columns=cols)
            st.dataframe(df_eval, use_container_width=True)

            csv_str = metrics_to_csv([dict(zip(cols, r)) for r in rows])
            st.download_button(
                label     = "⬇️ Export evaluation history as CSV",
                data      = csv_str,
                file_name = "loyola_rag_eval_results.csv",
                mime      = "text/csv",
            )

            # Multi-run comparison chart
            if len(df_eval) > 1:
                st.subheader("📈 Metric trends across runs")
                chart_df = df_eval.set_index("Run label")[["Precision@K", "Recall@K", "MRR", "NDCG"]]
                st.line_chart(chart_df)
        else:
            st.info("No evaluation runs yet. Click 'Run evaluation' above.")

    except Exception as e:
        st.warning(f"Could not load evaluation history: {e}")


# ==========================================================================================
# STREAMLIT UI
# ==========================================================================================

def main():

    st.markdown("""
        <div style='text-align:center; padding: 1rem 0'>
            <h1>🎓 Loyola College Assistant</h1>
            <p style='color: gray'>Ask anything about Loyola College, Chennai — v9.0</p>
        </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────
    st.sidebar.image(
        "https://www.loyolacollege.edu/wp-content/uploads/2021/09/loyola-logo.png",
        width=120,
    )
    st.sidebar.title("⚙️ Settings")

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        st.error("❌ GROQ_API_KEY not found in .env file.")
        st.stop()

    st.sidebar.divider()

    if st.sidebar.button("🔄 Rebuild Knowledge Base", use_container_width=True):
        for f in [CFG.CHUNKS_PKL, CFG.BM25_PKL, CFG.HASH_FILE, CFG.QUERY_CACHE_FILE]:
            if os.path.exists(f):
                os.remove(f)
        for f in ["index.faiss", "index.pkl"]:
            fp = Path(CFG.FAISS_PATH) / f
            if fp.exists():
                fp.unlink()
        st.cache_resource.clear()
        st.sidebar.success("Cleared — rebuilding...")
        st.rerun()

    # Source type filter
    st.sidebar.divider()
    st.sidebar.caption("🔍 Source filter (optional)")
    selected_types = st.sidebar.multiselect(
        "Restrict retrieval to",
        options = ["website", "pdf", "ocr", "cleaned_text"],
        default = [],
        help    = "Leave blank to search all sources.",
    )
    active_filter = selected_types if selected_types else None

    # Debug mode
    st.sidebar.divider()
    debug_mode = st.sidebar.checkbox(
        "🐛 Debug mode",
        value = False,
        help  = "Show retrieved chunks, rerank scores, and expanded queries.",
    )

    st.sidebar.divider()
    st.sidebar.caption("📁 Knowledge Base Status")
    sidebar_log = st.sidebar.container()

    if can_fast_load():
        st.sidebar.success("⚡ Index ready — fast load")
    else:
        st.sidebar.warning("🔄 First run — full build required")

    with st.sidebar.expander("ℹ️ System info"):
        st.caption(f"Device: `{DEVICE}`")
        st.caption(f"Embedding: `{CFG.EMBEDDING_MODEL}`")
        st.caption(f"Reranker: `{CFG.RERANK_MODEL}`")
        st.caption(f"LLM: `{CFG.LLM_MODEL}`")
        st.caption(f"Chunk: {CFG.CHUNK_SIZE} / overlap {CFG.CHUNK_OVERLAP}")
        st.caption(f"Rerank K={CFG.RERANK_K} | pool={CFG.RERANK_CANDIDATE_POOL}")
        st.caption(f"Retrieval K — fact:{CFG.RETRIEVAL_K_FACT} default:{CFG.RETRIEVAL_K_DEFAULT} "
                   f"deep:{CFG.RETRIEVAL_K_DEEP} compare:{CFG.RETRIEVAL_K_COMPARE}")
        st.caption(f"Cache max size: {CFG.SEMANTIC_CACHE_MAX_SIZE} entries (LRU)")

    # ── Load models ───────────────────────────────────────────────────
    with st.spinner("Loading AI models..."):
        emb        = load_embedding_model()
        reranker   = load_reranker()
        sent_model = load_sentence_model()
        llm_mdl    = get_llm(api_key)
        facts      = load_facts()

    # ── Load knowledge base ───────────────────────────────────────────
    with st.spinner(
        "Loading knowledge base..." if can_fast_load()
        else "Building knowledge base (first time only)..."
    ):
        db, chunks, bm25 = build_knowledge_base(sidebar_log, emb)
        retriever = HybridRRFRetriever(db, bm25)

    st.success(f"✅ Ready — {len(chunks):,} chunks loaded")

    # ── Session state ─────────────────────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "query_cache" not in st.session_state:
        st.session_state.query_cache = _load_cache()

    # ── Tabs ──────────────────────────────────────────────────────────
    tab_chat, tab_analytics, tab_failures, tab_feedback, tab_eval = st.tabs([
        "💬 Chat", "📊 Analytics", "⚠️ Failures", "👍 Feedback", "📐 Evaluation",
    ])

    # ================================================================
    # CHAT TAB
    # ================================================================
    with tab_chat:

        # Render conversation history
        for idx, chat in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.markdown(chat["q"])
            with st.chat_message("assistant"):
                st.markdown(chat["a"])

                conf      = chat.get("confidence", "?")
                emoji     = chat.get("conf_emoji", "")
                score     = chat.get("score", 0.0)
                facts_hit = chat.get("facts_hit", False)

                if facts_hit:
                    st.caption("⚡ Answered from structured facts layer")
                elif conf != "?":
                    st.caption(f"{emoji} Confidence: **{conf}** (grounding score: {score:.2f})")

                rich_sources = chat.get("rich_sources", [])
                if rich_sources:
                    render_source_cards(rich_sources)
                elif chat.get("sources"):
                    with st.expander("📎 Sources"):
                        for src in chat["sources"]:
                            if src.startswith("http"):
                                st.markdown(f"- [{src}]({src})")
                            else:
                                st.caption(f"- {src}")

                if chat.get("followups"):
                    st.markdown("**💡 You might also ask:**")
                    cols = st.columns(len(chat["followups"]))
                    for i, fq in enumerate(chat["followups"]):
                        if cols[i].button(fq, key=f"fq_hist_{idx}_{i}"):
                            st.session_state["prefill_query"] = fq
                            st.rerun()

                log_id    = chat.get("log_id")
                voted_key = f"voted_{idx}"
                if not st.session_state.get(voted_key, False):
                    fb_cols = st.columns([1, 1, 8])
                    if fb_cols[0].button("👍", key=f"up_{idx}"):
                        log_feedback(log_id, chat["q"], helpful=True)
                        st.session_state[voted_key] = True
                        st.rerun()
                    if fb_cols[1].button("👎", key=f"dn_{idx}"):
                        log_feedback(log_id, chat["q"], helpful=False)
                        st.session_state[voted_key] = True
                        st.rerun()
                else:
                    st.caption("✅ Feedback recorded — thank you!")

        # ── Chat input ────────────────────────────────────────────────
        prefill    = st.session_state.pop("prefill_query", None)
        user_query = st.chat_input(
            "Ask about admissions, programmes, placements, fees, faculty…"
        ) or prefill

        if user_query:

            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                placeholder   = st.empty()
                conf_slot     = st.empty()
                debug_slot    = st.empty()
                sources_slot  = st.empty()
                followup_slot = st.empty()
                feedback_slot = st.empty()

                with st.spinner("Thinking..."):
                    result = ask(
                        query       = user_query,
                        retriever   = retriever,
                        reranker    = reranker,
                        llm_mdl     = llm_mdl,
                        sent_model  = sent_model,
                        history     = st.session_state.chat_history,
                        cache       = st.session_state.query_cache,
                        facts       = facts,
                        type_filter = active_filter,
                        debug       = debug_mode,
                    )

                answer       = result["answer"]
                sources      = result["sources"]
                rich_sources = result.get("rich_sources", [])
                conf         = result["confidence"]
                emoji        = result["conf_emoji"]
                score        = result["score"]
                followups    = result["followups"]
                q_class      = result["q_class"]
                cache_hit    = result["cache_hit"]
                facts_hit    = result["facts_hit"]
                log_id       = result.get("log_id")

                placeholder.markdown(answer)

                badges = []
                if facts_hit:
                    badges.append("⚡ Structured facts")
                elif conf != "N/A":
                    badges.append(f"{emoji} Confidence: **{conf}** (score: {score:.2f})")
                if cache_hit:
                    badges.append("🗂 Cached")
                if active_filter:
                    badges.append(f"🔍 Filtered: {', '.join(active_filter)}")
                badges.append(f"🏷 `{q_class}`")
                conf_slot.caption("  |  ".join(badges))

                # Debug expander — v9 also shows expanded query
                if debug_mode:
                    with debug_slot.expander("🐛 Debug: retrieval details", expanded=False):
                        st.markdown(f"**Effective query:** `{user_query}`")
                        st.markdown(f"**Expanded query:** `{expand_query(user_query)}`")
                        st.markdown(f"**Class:** `{q_class}` | **K:** `{get_retrieval_k(q_class)}`")
                        if rich_sources:
                            for i, rs in enumerate(rich_sources):
                                st.markdown(
                                    f"**[{i+1}]** score=`{rs['rerank_score']}` | "
                                    f"type=`{rs['type']}` | "
                                    f"`{rs['source'][:80]}`"
                                )

                if rich_sources:
                    with sources_slot.container():
                        render_source_cards(rich_sources)
                elif sources:
                    with sources_slot.expander("📎 Sources"):
                        for src in sources:
                            if src.startswith("http"):
                                st.markdown(f"- [{src}]({src})")
                            else:
                                st.caption(f"- {src}")

                if followups:
                    followup_slot.markdown("**💡 You might also ask:**")
                    cols = st.columns(len(followups))
                    for i, fq in enumerate(followups):
                        if cols[i].button(fq, key=f"fq_new_{i}"):
                            st.session_state["prefill_query"] = fq
                            st.rerun()

                with feedback_slot.container():
                    fb_cols = st.columns([1, 1, 8])
                    if fb_cols[0].button("👍", key="fb_up_new"):
                        log_feedback(log_id, user_query, helpful=True)
                        st.toast("Thanks for the feedback! 👍")
                    if fb_cols[1].button("👎", key="fb_dn_new"):
                        log_feedback(log_id, user_query, helpful=False)
                        st.toast("Thanks for the feedback! 👎")

            chat_entry = {
                "q"           : user_query,
                "a"           : answer,
                "sources"     : sources,
                "rich_sources": rich_sources,
                "confidence"  : conf,
                "conf_emoji"  : emoji,
                "score"       : score,
                "followups"   : followups,
                "facts_hit"   : facts_hit,
                "log_id"      : log_id,
            }
            st.session_state.chat_history.append(chat_entry)
            st.session_state.chat_history = \
                st.session_state.chat_history[-CFG.HISTORY_TURNS * 2:]

    # ================================================================
    # OTHER TABS
    # ================================================================
    with tab_analytics:
        render_analytics_tab()

    with tab_failures:
        render_failures_tab()

    with tab_feedback:
        render_feedback_tab()

    with tab_eval:
        render_evaluation_tab(retriever)


# ==========================================================================================
# ENTRY POINT
# ==========================================================================================

if __name__ == "__main__":
    main()