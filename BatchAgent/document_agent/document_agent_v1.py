import os
import json
import re
import time
import argparse
import requests
import numpy as np
from pathlib import Path
from typing import Any, List, Dict, Optional
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
# Import the local package modules (make sure ia_phase1 is in your PYTHONPATH)
from BatchAgent.document_agent import parser, figures, chunking
from BatchAgent.document_agent import tables, equations


# ==========================================
# 0. SHARED EMBEDDING HELPERS
# ==========================================
def get_embeddings(
    texts: List[str],
    embedding_type: str = "local",
    local_model: Any = None,
    base_url: str = "",
    api_key: str = "",
    api_model: str = "",
    batch_size: int = 32,
    timeout: float = 60.0,
    max_retries: int = 3,
) -> np.ndarray:
    """
    Shared embedding helper with batching and retry support.

    For API mode, texts are split into batches to avoid timeouts on large inputs.
    Each batch is retried up to max_retries times with exponential backoff.
    Raises RuntimeError if the API is unreachable (no mock fallback).
    """
    if not texts:
        return np.array([])

    if embedding_type == "api":
        endpoint = base_url.strip().rstrip("/")
        if not endpoint.endswith("/embeddings"):
            endpoint += "/embeddings" if endpoint.endswith("/v1") else "/v1/embeddings"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        all_embeddings: List[List[float]] = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch_texts = texts[batch_start : batch_start + batch_size]
            payload = {"input": batch_texts, "model": api_model}

            last_error: Optional[str] = None
            for attempt in range(1, max_retries + 1):
                try:
                    response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
                    if response.status_code == 200:
                        data = sorted(response.json()["data"], key=lambda x: x.get("index", 0))
                        all_embeddings.extend([item["embedding"] for item in data])
                        print(f"[Embeddings] Batch {batch_idx + 1}/{total_batches} OK ({len(batch_texts)} texts)")
                        last_error = None
                        break
                    else:
                        last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                        print(f"[Embeddings] Batch {batch_idx + 1} attempt {attempt}/{max_retries} failed: {last_error}")
                except requests.exceptions.Timeout:
                    last_error = f"Timeout after {timeout}s"
                    print(f"[Embeddings] Batch {batch_idx + 1} attempt {attempt}/{max_retries} timed out")
                except Exception as e:
                    last_error = str(e)
                    print(f"[Embeddings] Batch {batch_idx + 1} attempt {attempt}/{max_retries} error: {last_error}")

                if attempt < max_retries:
                    wait = 2 ** attempt
                    print(f"[Embeddings] Retrying in {wait}s...")
                    time.sleep(wait)

            if last_error:
                raise RuntimeError(
                    f"Embedding API failed after {max_retries} retries for batch {batch_idx + 1} "
                    f"(texts {batch_start}-{batch_start + len(batch_texts) - 1}): {last_error}"
                )

        return np.array(all_embeddings)
    else:
        if local_model is None:
            raise ValueError("local_model must be provided for embedding_type='local'")
        return local_model.encode(texts)


# ==========================================
# 0b. ACTIVE PIPELINE REGISTRY
# ==========================================
_active_pipelines: Dict[str, Any] = {}


def register_pipeline(name: str, pipeline: Dict[str, Any]):
    """Register a loaded document pipeline for use by tool handlers."""
    _active_pipelines[name] = pipeline
    print(f"[DocumentAgent] Registered pipeline '{name}' ({len(pipeline.get('chunks', []))} chunks)")


def unregister_pipeline(name: str):
    """Remove a pipeline from the registry."""
    _active_pipelines.pop(name, None)


def get_active_chunk_index() -> Optional['ChunkRAGIndex']:
    """Get the first available ChunkRAGIndex from registered pipelines."""
    for pipeline in _active_pipelines.values():
        idx = pipeline.get("chunk_index")
        if idx is not None:
            return idx
    return None


def get_active_section_agent() -> Optional['StructuredDocumentAgent']:
    """Get the first available StructuredDocumentAgent from registered pipelines."""
    for pipeline in _active_pipelines.values():
        agent = pipeline.get("section_agent")
        if agent is not None:
            return agent
    return None


# ==========================================
# 1. THE INSTRUCTOR-ASSISTANT EXTRACTOR
# ==========================================
class InstructorAssistantExtractor:
    """
    Wraps the ia_phase1 modules. Runs their multi-modal extraction pipeline,
    reads the generated JSON manifests, and stitches them into LLM-ready Markdown.
    """
    def __init__(self, pdf_path: str, paper_id: int = 1, data_dir: str = None):
        self.pdf_path = Path(pdf_path).resolve()
        self.paper_id = paper_id
        self._raw_blocks: List[Dict[str, Any]] = []

        # Derive data_dir from pdf_path unless explicitly provided
        if data_dir:
            self.data_dir = Path(data_dir).resolve()
        else:
            self.data_dir = self.pdf_path.parent / ".document_agent_data" / str(self.paper_id)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _get_heading_prefix(self, block: Dict[str, Any]) -> str:
        """Uses ia_phase1's font metadata to determine if a block is a header."""
        meta = block.get("metadata", {})
        avg_font = meta.get("avg_font_size", 0)
        max_font = meta.get("max_font_size", 0)
        bold_ratio = meta.get("bold_ratio", 0)
        
        # Simple heuristic based on phase1 metadata
        if max_font > 14 or bold_ratio > 0.8:
            return "## "
        elif max_font > 12:
            return "### "
        return ""

    def extract_to_markdown(self) -> str:
        print(f"📄 Running ia_phase1 Parser on: {self.pdf_path}...")
        
        # 1. Base Text Extraction (Provides layout-aware blocks with font metadata)
        blocks = parser.extract_text_blocks(self.pdf_path)
        self._raw_blocks = blocks  # preserve for chunk-level RAG
        
        # 2. Multi-modal Extraction Pipeline
        print("🖼️ Extracting Figures via ia_phase1...")
        figures.extract_and_store_paper_figures(self.pdf_path, self.paper_id, blocks)
        
        # print("📊 Extracting Tables via ia_phase1...")
        tables.extract_and_store_paper_tables(self.pdf_path, self.paper_id, blocks)
        
        # print("🧮 Extracting Equations via ia_phase1...")
        equations.extract_and_store_paper_equations(self.pdf_path, self.paper_id, blocks)

        # 3. Load the Manifests
        fig_manifest = figures.load_paper_figure_manifest(self.paper_id)
        extracted_figures = fig_manifest.get("images", [])
        
        # Group figures by page for easy injection
        figures_by_page = {}
        for fig in extracted_figures:
            page = fig.get("page_no")
            figures_by_page.setdefault(page, []).append(fig)

        # 4. Stitch Markdown Together
        print("🧵 Stitching multi-modal Markdown...")
        full_markdown = []
        current_page = 0
        
        for block in blocks:
            page_no = block.get("page_no")
            
            # Inject Page Markers and Modalities when we cross to a new page
            if page_no != current_page:
                full_markdown.append(f"\n\n[PAGE: {page_no}]\n")
                
                # Inject any figures found on this page
                if page_no in figures_by_page:
                    for fig in figures_by_page[page_no]:
                        img_path = Path(fig["image_path"]).resolve()
                        caption = fig.get("figure_caption") or "Extracted Figure"
                        full_markdown.append(f"\n![{caption}]({img_path})\n")
                
                current_page = page_no

            # Format the text block using ia_phase1's rich metadata
            text = block.get("text", "").strip()
            if not text: continue
                
            prefix = self._get_heading_prefix(block)
            if prefix:
                full_markdown.append(f"\n\n{prefix}{text}\n\n")
            else:
                full_markdown.append(f"{text} ")

        return "\n".join(full_markdown)

# ==========================================
# 2. THE LLM DOCUMENT AGENT
# ==========================================
class StructuredDocumentAgent:
    """Takes the stitched Markdown, builds a searchable tree, and exposes tools."""
    
    def __init__(self, markdown_text: str, embedding_type: str = "local", local_model_name: str = "all-MiniLM-L6-v2"):
        self.raw_text = markdown_text
        self.sections: Dict[str, Any] = {}
        self.tree: List[Dict[str, Any]] = []
        self.embedding_type = embedding_type
        
        # Configure Embeddings Setup
        if self.embedding_type == "api":
            self.base_url = os.environ.get("EMBEDDING_BASE_URL", "http://100.83.246.7:8003/v1")
            self.api_key = os.environ.get("EMBEDDING_API_KEY", "EMPTY")
            self.api_model = os.environ.get("EMBEDDING_MODEL", "custom_bge_gpu1")
            print(f"🔗 Configured to use External Embedding API ({self.api_model} at {self.base_url})")
        else:
            print(f"⚙️ Configured to use Local SentenceTransformer ({local_model_name})")
            self.local_model = SentenceTransformer(local_model_name)

        print("🌳 Building document structure tree...")
        self._build_structure_tree()
        
        print("🔍 Building hybrid search indices (this takes a moment)...")
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self._build_search_indices()
        print("✅ Document Agent is ready for queries!")

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        return get_embeddings(
            texts,
            embedding_type=self.embedding_type,
            local_model=getattr(self, "local_model", None),
            base_url=getattr(self, "base_url", ""),
            api_key=getattr(self, "api_key", ""),
            api_model=getattr(self, "api_model", ""),
        )

    def _build_structure_tree(self):
        lines = self.raw_text.split('\n')
        current_section = {"id": "root", "title": "Document Start", "level": 0, "content": "", "children": [], "pages": set()}
        self.sections["root"] = current_section
        self.tree.append(current_section)
        
        stack = [current_section]
        section_counter = 1
        current_page = 1

        for line in lines:
            page_match = re.search(r'\[PAGE:\s*(\d+)\]', line)
            if page_match:
                current_page = int(page_match.group(1))
                continue

            header_match = re.match(r'^(#{1,6})\s+(.*)', line)
            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                sec_id = f"sec_{section_counter}"
                section_counter += 1
                
                new_section = {"id": sec_id, "title": title, "level": level, "content": "", "children": [], "pages": set([current_page])}
                self.sections[sec_id] = new_section
                
                while stack and stack[-1]["level"] >= level: stack.pop()
                if stack: stack[-1]["children"].append(new_section)
                else: self.tree.append(new_section)
                    
                stack.append(new_section)
                current_section = new_section
            else:
                if line.strip() and not line.startswith("---"):
                    current_section["content"] += line + "\n"
                    current_section["pages"].add(current_page)

    def _build_search_indices(self):
        self.section_ids_ordered = [sec_id for sec_id, data in self.sections.items() if data["content"].strip()]
        texts = [self.sections[sec_id]["content"] for sec_id in self.section_ids_ordered]
        if not texts: return
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        self.section_embeddings = self._get_embeddings(texts)

    def get_overview(self) -> str:
        def build_toc(nodes: List[Dict], indent=""):
            toc = []
            for node in nodes:
                if node["id"] != "root" or node["title"] != "Document Start":
                    toc.append(f"{indent}- [{node['id']}] {node['title']}")
                toc.extend(build_toc(node["children"], indent + "  "))
            return toc
        return "Document Overview:\n" + "\n".join(build_toc(self.tree))

    def read_details(self, section_id: str) -> str:
        if section_id not in self.sections: return "Error: Section ID not found."
        section = self.sections[section_id]
        
        pages = sorted(list(section["pages"]))
        page_str = f"Page {pages[0]}" if len(pages) == 1 else f"Pages {pages[0]}-{pages[-1]}"
        
        result = f"--- SECTION: {section['title']} ({section_id}) | Found on {page_str} ---\n\n"
        result += section['content'].strip()
        
        if section["children"]:
            result += "\n\nSub-sections available:\n" + "\n".join([f"- {c['title']} ({c['id']})" for c in section["children"]])
        return result

    def search(self, query: str, top_k: int = 3) -> str:
        if not self.section_ids_ordered: return "Error: Document is empty."
        
        query_tfidf = self.tfidf_vectorizer.transform([query])
        keyword_scores = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]
        query_embedding = self._get_embeddings([query])
        semantic_scores = cosine_similarity(query_embedding, self.section_embeddings)[0]
        
        hybrid_scores = (keyword_scores * 0.4) + (semantic_scores * 0.6)
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        
        results = f"Search Results for: '{query}'\n\n"
        for rank, idx in enumerate(top_indices):
            sec_id = self.section_ids_ordered[idx]
            section = self.sections[sec_id]
            
            pages = sorted(list(section["pages"]))
            page_str = f"Page {pages[0]}" if len(pages) == 1 else f"Pages {pages[0]}-{pages[-1]}"
            
            snippet = section["content"].strip()[:250] + "..."
            results += f"{rank+1}. [{sec_id}] {section['title']} (Score: {hybrid_scores[idx]:.2f} | Location: {page_str})\n   Snippet: {snippet}\n\n"
        return results

# ==========================================
# 3. PIPELINE WRAPPER
# ==========================================
def create_agent_from_pdf(pdf_filepath: str, embedding_type: str = "local", local_model_name: str = "all-MiniLM-L6-v2") -> StructuredDocumentAgent:
    """Wrapper that extracts PDF to MD using ia_phase1 and initializes the Agent."""
    if not os.path.exists(pdf_filepath):
        raise FileNotFoundError(f"Could not find PDF at: {pdf_filepath}")
        
    # We use a hash of the filename as a pseudo paper_id to keep outputs separated
    import hashlib
    paper_id = int(hashlib.sha256(pdf_filepath.encode('utf-8')).hexdigest()[:8], 16)
    
    extractor = InstructorAssistantExtractor(pdf_filepath, paper_id=paper_id)
    markdown_text = extractor.extract_to_markdown()
    
    md_path = extractor.data_dir / (Path(pdf_filepath).stem + '_extracted.md')
    with open(str(md_path), 'w', encoding='utf-8') as f:
        f.write(markdown_text)
    print(f"Saved intermediate markdown to {md_path}")
    
    return StructuredDocumentAgent(markdown_text, embedding_type=embedding_type, local_model_name=local_model_name)


# ==========================================
# 4. CHUNK-LEVEL RAG INDEX
# ==========================================
class ChunkRAGIndex:
    """
    Chunk-level RAG index. Takes raw text blocks from parser.extract_text_blocks(),
    runs them through chunking.chunk_text_blocks(), embeds each chunk, and provides
    hybrid search at the chunk granularity.
    """

    def __init__(
        self,
        blocks: List[Dict[str, Any]],
        embedding_type: str = "local",
        local_model_name: str = "all-MiniLM-L6-v2",
        target_chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.embedding_type = embedding_type

        # Configure embeddings
        if self.embedding_type == "api":
            self.base_url = os.environ.get("EMBEDDING_BASE_URL", "http://100.83.246.7:8003/v1")
            self.api_key = os.environ.get("EMBEDDING_API_KEY", "EMPTY")
            self.api_model = os.environ.get("EMBEDDING_MODEL", "custom_bge_gpu1")
        else:
            self.local_model = SentenceTransformer(local_model_name)

        # Chunk the blocks
        self.chunks = chunking.chunk_text_blocks(
            blocks,
            target_size=target_chunk_size,
            overlap=chunk_overlap,
        )
        print(f"  Chunked {len(blocks)} blocks into {len(self.chunks)} chunks")

        # Build search indices over chunks
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self._build_chunk_indices()

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        return get_embeddings(
            texts,
            embedding_type=self.embedding_type,
            local_model=getattr(self, "local_model", None),
            base_url=getattr(self, "base_url", ""),
            api_key=getattr(self, "api_key", ""),
            api_model=getattr(self, "api_model", ""),
        )

    def _build_chunk_indices(self):
        self._valid_indices = [i for i, c in enumerate(self.chunks) if c.get("text", "").strip()]
        texts = [self.chunks[i]["text"] for i in self._valid_indices]
        if not texts:
            self.tfidf_matrix = None
            self.chunk_embeddings = None
            return
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        self.chunk_embeddings = self._get_embeddings(texts)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search chunks with hybrid TF-IDF + semantic scoring."""
        if self.tfidf_matrix is None:
            return []

        query_tfidf = self.tfidf_vectorizer.transform([query])
        keyword_scores = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]
        query_embedding = self._get_embeddings([query])
        semantic_scores = cosine_similarity(query_embedding, self.chunk_embeddings)[0]

        hybrid_scores = (keyword_scores * 0.4) + (semantic_scores * 0.6)
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk_idx = self._valid_indices[idx]
            chunk = self.chunks[chunk_idx]
            results.append({
                "text": chunk["text"],
                "page_no": chunk.get("page_no"),
                "score": float(hybrid_scores[idx]),
                "metadata": chunk.get("metadata", {}),
            })
        return results

    def search_formatted(self, query: str, top_k: int = 5) -> str:
        """Returns a formatted string of search results for LLM consumption."""
        results = self.search(query, top_k)
        if not results:
            return f"No chunk results for: '{query}'"

        output = f"Chunk Search Results for: '{query}'\n\n"
        for rank, r in enumerate(results, 1):
            section = r["metadata"].get("section_primary", "unknown")
            snippet = r["text"][:300] + ("..." if len(r["text"]) > 300 else "")
            output += (
                f"{rank}. [Page {r['page_no']}] Section: {section} "
                f"(Score: {r['score']:.2f})\n   {snippet}\n\n"
            )
        return output


def create_full_pipeline_from_pdf(
    pdf_filepath: str,
    embedding_type: str = "local",
    local_model_name: str = "all-MiniLM-L6-v2",
    target_chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> Dict[str, Any]:
    """
    Full pipeline: extract PDF -> section agent + chunk RAG index.

    Returns dict with keys:
        section_agent, chunk_index, markdown, blocks, chunks
    """
    if not os.path.exists(pdf_filepath):
        raise FileNotFoundError(f"Could not find PDF at: {pdf_filepath}")

    import hashlib
    paper_id = int(hashlib.sha256(pdf_filepath.encode('utf-8')).hexdigest()[:8], 16)

    extractor = InstructorAssistantExtractor(pdf_filepath, paper_id=paper_id)
    markdown_text = extractor.extract_to_markdown()
    raw_blocks = extractor._raw_blocks

    # Save intermediate markdown
    md_path = extractor.data_dir / (Path(pdf_filepath).stem + '_extracted.md')
    with open(str(md_path), 'w', encoding='utf-8') as f:
        f.write(markdown_text)
    print(f"Saved intermediate markdown to {md_path}")

    # Build section-level agent (backward compat)
    print("Building section-level agent...")
    section_agent = StructuredDocumentAgent(
        markdown_text,
        embedding_type=embedding_type,
        local_model_name=local_model_name,
    )

    # Build chunk-level RAG index
    print("Building chunk-level RAG index...")
    chunk_index = ChunkRAGIndex(
        raw_blocks,
        embedding_type=embedding_type,
        local_model_name=local_model_name,
        target_chunk_size=target_chunk_size,
        chunk_overlap=chunk_overlap,
    )

    result = {
        "section_agent": section_agent,
        "chunk_index": chunk_index,
        "markdown": markdown_text,
        "blocks": raw_blocks,
        "chunks": chunk_index.chunks,
    }

    # Auto-register so tool_handler can access it
    pipeline_name = Path(pdf_filepath).stem
    register_pipeline(pipeline_name, result)

    return result


# ==========================================
# 5. TEST CASES
# ==========================================
def test_pdf(args):
    print("\n" + "="*50)
    print(f"RUNNING PDF TEST on: {args.pdf_path}")
    print("="*50)

    if not os.path.exists(args.pdf_path):
        print(f"PDF not found: '{args.pdf_path}'. Please check the path.")
        return

    result = create_full_pipeline_from_pdf(
        pdf_filepath=args.pdf_path,
        embedding_type=args.embedding_type,
        local_model_name=args.local_model,
    )
    agent = result["section_agent"]
    chunk_index = result["chunk_index"]

    print("\n--- TEST: GET OVERVIEW ---")
    print(agent.get_overview())

    print("\n--- TEST: SECTION SEARCH ---")
    print(agent.search("methodology or implementation details"))

    print("\n--- TEST: CHUNK SEARCH ---")
    print(chunk_index.search_formatted("methodology or implementation details"))

    print(f"\nPipeline stats: {len(result['blocks'])} blocks, {len(result['chunks'])} chunks")

def test_agent(args):
    print("\n" + "="*50)
    print("RUNNING AGENT MOCK TEXT TEST")
    print("="*50)

    sample_markdown = """
# Abstract
This paper introduces a novel framework for analyzing data.

[PAGE: 1]
# 1. Introduction
Data analysis is hard. We present a new method.
## 1.1 Background
Historically, people used abacuses.
## 1.2 Motivation
We need faster tools because data is growing.

[PAGE: 2]
# 2. Methodology
Our method relies on hybrid extraction.
## 2.1 The OpenDataLoader Approach
We use bounding boxes and layout analysis.
## 2.2 The PaperIndex Structure
We build trees out of documents so LLMs can read them.

# 3. Results
It works very well.
    """
    # Section-level agent test
    agent_kb = StructuredDocumentAgent(
        sample_markdown,
        embedding_type=args.embedding_type,
        local_model_name=args.local_model
    )

    print("\nTOOL CALL: get_overview()")
    print(agent_kb.get_overview())

    print("\nTOOL CALL: read_details('sec_4')")
    print(agent_kb.read_details('sec_4'))

    print("\nTOOL CALL: search('how do you extract layout and bounding boxes?')")
    print(agent_kb.search('how do you extract layout and bounding boxes?'))

    # Chunk-level RAG test with synthetic blocks
    print("\n" + "-"*50)
    print("CHUNK RAG INDEX TEST (synthetic blocks)")
    print("-"*50)
    sample_blocks = [
        {"page_no": 1, "block_index": 0, "text": "Abstract This paper introduces a novel framework for analyzing data.",
         "bbox": {"x0": 72, "y0": 100, "x1": 540, "y1": 120},
         "metadata": {"max_font_size": 16.0, "avg_font_size": 12.0, "bold_ratio": 1.0}},
        {"page_no": 1, "block_index": 1, "text": "Data analysis is a critical challenge in modern computing. We present a method that leverages layout-aware extraction.",
         "bbox": {"x0": 72, "y0": 130, "x1": 540, "y1": 200},
         "metadata": {"max_font_size": 11.0, "avg_font_size": 11.0, "bold_ratio": 0.0}},
        {"page_no": 2, "block_index": 0, "text": "Our methodology relies on hybrid extraction using bounding boxes and layout analysis to build hierarchical document structures.",
         "bbox": {"x0": 72, "y0": 100, "x1": 540, "y1": 180},
         "metadata": {"max_font_size": 11.0, "avg_font_size": 11.0, "bold_ratio": 0.0}},
        {"page_no": 2, "block_index": 1, "text": "Results show significant improvement in retrieval accuracy compared to naive full-text search approaches.",
         "bbox": {"x0": 72, "y0": 190, "x1": 540, "y1": 240},
         "metadata": {"max_font_size": 11.0, "avg_font_size": 11.0, "bold_ratio": 0.0}},
    ]
    chunk_index = ChunkRAGIndex(
        sample_blocks,
        embedding_type=args.embedding_type,
        local_model_name=args.local_model,
        target_chunk_size=200,
    )
    print(f"\nChunks created: {len(chunk_index.chunks)}")
    print("\nCHUNK SEARCH: 'bounding boxes and layout'")
    print(chunk_index.search_formatted("bounding boxes and layout"))


# ==========================================
# MAIN EXECUTION & ARGPARSE
# ==========================================
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Structured Document Agent Pipeline")

    # Test Execution Options
    arg_parser.add_argument("--test", type=str, choices=["pdf", "agent", "both"], default="both",
                        help="Which test to run: 'pdf', 'agent', or 'both'")
    arg_parser.add_argument("--pdf_path", type=str, default="data/NVIDIA-Nemotron-3-Super-Technical-Report.pdf",
                        help="Path to the PDF file to process")

    # Embedding Configuration
    arg_parser.add_argument("--embedding_type", type=str, choices=["local", "api"], default="local",
                        help="Use 'local' SentenceTransformers or external 'api'")
    arg_parser.add_argument("--local_model", type=str, default="all-MiniLM-L6-v2",
                        help="HuggingFace model name for local embeddings")
    arg_parser.add_argument("--api_url", type=str, default="http://100.83.246.7:8003/v1",
                        help="Base URL for the embedding API")
    arg_parser.add_argument("--api_key", type=str, default="EMPTY",
                        help="API Key for the embedding API")
    arg_parser.add_argument("--api_model", type=str, default="custom_bge_gpu1",
                        help="Model name to pass to the API")

    args = arg_parser.parse_args()

    # Set environment variables for the API if requested
    if args.embedding_type == "api":
        os.environ["EMBEDDING_BASE_URL"] = args.api_url
        os.environ["EMBEDDING_API_KEY"] = args.api_key
        os.environ["EMBEDDING_MODEL"] = args.api_model

    # Execute tests
    if args.test in ["agent", "both"]:
        test_agent(args)
        
    if args.test in ["pdf", "both"]:
        test_pdf(args)

"""
python document_agent_v1.py --test pdf --pdf_path "../../data/pdftest/NVIDIA-Nemotron-3-Super-Technical-Report.pdf" --embedding_type api --api_url "http://100.81.148.35:8003/v1" --api_model "BAAI/bge-large-zh-v1.5"

"""
