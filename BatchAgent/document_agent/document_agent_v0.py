"""
To build a robust PDF-to-Markdown extractor from scratch that feeds perfectly into your StructuredDocumentAgent, we need to emulate the core philosophies of both OpenDataLoader and PaperIndex:

Layout Awareness: We can't just scrape text left-to-right; we need to respect spatial boundaries (bounding boxes) to reconstruct columns and paragraphs.

Structural Hierarchies: We need to dynamically detect headers (Title, H1, H2) so PaperIndex can build its navigation tree.

Table Extraction: Standard text extractors mangle tables. We need a dedicated table parser.

To achieve this without heavy AI models, we will use a "hybrid" approach utilizing two of the best standard Python libraries: PyMuPDF (for high-speed, font-aware text extraction and layout) and pdfplumber (for precise, line-based table extraction).

pip install pymupdf pdfplumber


for the hybrid search (keyword + vector) to function:
pip install sentence-transformers scikit-learn numpy
pip install pymupdf pdfplumber sentence-transformers scikit-learn numpy
pip install requests

Create a Python class that acts as a "Document Brain" for the LLM. It will parse the extracted Markdown into a hierarchical tree, index the sections for search, and expose the three specific tools you requested.

Context Window Management: Instead of shoving a 50-page PDF into the LLM's prompt, the LLM calls get_overview() to see the skeleton of the document (consuming only ~500 tokens).

Targeted Reading: The LLM can logically decide, "I need to look at the Methodology, let me call read_details('sec_5')".

Fallback Discovery: If the table of contents is too vague, the search() tool uses a hybrid approach. It catches exact terminology (TF-IDF keyword matching) and conceptual questions (Sentence Transformers), preventing the LLM from hallucinating answers when it can't find a section.
"""

import os
import re
import argparse
import requests
import fitz  # PyMuPDF
import pdfplumber
import statistics
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. THE PDF TO MARKDOWN EXTRACTOR
# ==========================================

class HybridPDFExtractor:
    """
    A standalone PDF-to-Markdown extractor.
    Features 'Text-Snapping' for figures to ensure axes and legends aren't cut off,
    and isolates tables before processing charts to prevent over-merging.
    """
    def __init__(self, pdf_path: str):
        self.pdf_path = os.path.abspath(pdf_path)
        self.output_dir = os.path.dirname(self.pdf_path)
        self.pdf_name = os.path.splitext(os.path.basename(self.pdf_path))[0]

    def _get_base_font_size(self, doc: fitz.Document) -> float:
        sizes = []
        for page in doc:
            for b in page.get_text("dict")["blocks"]:
                if "lines" in b:
                    for l in b["lines"]:
                        for s in l["spans"]:
                            if s["text"].strip():
                                sizes.append(s["size"])
        return statistics.mode([round(s) for s in sizes]) if sizes else 11.0

    def _get_header_prefix(self, span_size: float, base_size: float) -> str:
        ratio = span_size / base_size
        if ratio >= 1.7: return "# "
        if ratio >= 1.3: return "## "
        if ratio >= 1.1: return "### "
        return ""

    def _get_figure_zones(self, page_fitz: fitz.Page, table_bboxes: List[fitz.Rect]) -> List[fitz.Rect]:
        """
        Calculates figure bounding boxes by isolating drawings from tables,
        grouping them, and then 'snapping' nearby text labels into the crop area.
        """
        zones = [fitz.Rect(img["bbox"]) for img in page_fitz.get_image_info()]
            
        # 1. Get Drawings, but IGNORE any drawings that fall inside a Table
        for d in page_fitz.get_drawings():
            rect = d["rect"]
            if 10 < rect.width < page_fitz.rect.width * 0.95 and 10 < rect.height < page_fitz.rect.height * 0.95:
                # Manually calculate center to avoid PyMuPDF version conflicts
                cx = (rect.x0 + rect.x1) / 2
                cy = (rect.y0 + rect.y1) / 2
                
                # Is this drawing inside a table?
                if not any(tb.contains(fitz.Point(cx, cy)) for tb in table_bboxes):
                    zones.append(rect)
                
        # 2. Merge nearby zones (36pts / 0.5 inches margin)
        margin = 36  
        changed = True
        while changed:
            changed = False
            merged_zones = []
            while zones:
                current = zones.pop(0)
                expanded_current = current + (-margin, -margin, margin, margin)
                
                merged_this_round = False
                for i, mz in enumerate(merged_zones):
                    if (mz + (-margin, -margin, margin, margin)).intersects(expanded_current):
                        merged_zones[i] = mz | current 
                        merged_this_round = True
                        changed = True
                        break
                if not merged_this_round:
                    merged_zones.append(current)
            zones = merged_zones

        # 3. TEXT SNAPPING: Expand the zones to include touching/overlapping text (Axes, Legends)
        text_blocks = [b for b in page_fitz.get_text("dict")["blocks"] if b["type"] == 0]
        for i, z in enumerate(zones):
            expanded_z = z
            text_added = True
            # Keep expanding until no new text blocks are touched
            while text_added:
                text_added = False
                for b in text_blocks:
                    b_rect = fitz.Rect(b["bbox"])
                    # If the text block touches the current figure zone (with a 5pt grace area)
                    if (expanded_z + (-5, -5, 5, 5)).intersects(b_rect):
                        # Only snap small text blocks (ignore whole paragraphs)
                        if b_rect.height < 100 and b_rect.width < 300:
                            if not expanded_z.contains(b_rect):
                                expanded_z = expanded_z | b_rect
                                text_added = True
            zones[i] = expanded_z
                
        return zones

    def extract_to_markdown(self) -> str:
        print(f"📄 Opening {self.pdf_path}...")
        doc_fitz = fitz.open(self.pdf_path)
        doc_plumber = pdfplumber.open(self.pdf_path)
        base_size = self._get_base_font_size(doc_fitz)
        full_markdown = []

        for page_num in range(len(doc_fitz)):
            page_fitz = doc_fitz[page_num]
            page_plumber = doc_plumber.pages[page_num]
            layout_items = []

            # ==========================================
            # 1. EXTRACT TABLES (With Smart Post-Processing)
            # ==========================================
            table_settings = {
                "vertical_strategy": "text", 
                "horizontal_strategy": "text",
                "intersection_y_tolerance": 15,
            }
            plumber_tables = page_plumber.find_tables(table_settings)
            table_bboxes = []
            
            for table in plumber_tables:
                tb_rect = fitz.Rect(table.bbox)
                extracted_data = table.extract()
                
                if not extracted_data or not extracted_data[0]: 
                    continue

                cleaned_table = []
                hit_paragraph = False
                
                for row in extracted_data:
                    # 1. Skip completely blank rows
                    if not any(cell and str(cell).strip() for cell in row):
                        continue
                        
                    clean_row = [re.sub(r'\s+', ' ', str(cell).strip()) if cell else "" for cell in row]
                    joined_text = " ".join(clean_row).strip()
                    
                    # --- BLEED DETECTOR (Circuit Breaker) ---
                    # 2a. Break instantly if we hit a Table/Figure caption
                    if re.search(r'^(Table|Figure)\s+\d+', joined_text, re.IGNORECASE):
                        hit_paragraph = True
                        break
                        
                    # 2b. The "Chopped Paragraph" Heuristic
                    # If 3 or more cells contain long text (>12 chars) and the row is wide,
                    # it is almost certainly a paragraph that got sliced by imaginary columns.
                    long_cells = sum(1 for c in clean_row if len(c) > 12)
                    if long_cells >= 3 and len(joined_text) > 80:
                        hit_paragraph = True
                        break
                        
                    # --- HEADER REPAIR ---
                    # 3. Fix severed Category Headers (e.g., "General Knowled" | "ge" | "" | "")
                    non_empty = [c for c in clean_row if c]
                    # If it's a sparse row lacking numbers, seamlessly merge it
                    if len(non_empty) <= 2 and not any(char.isdigit() for char in joined_text):
                        # "".join() elegantly reconnects "Knowled" and "ge"
                        merged_header = "".join(non_empty) 
                        clean_row = [merged_header] + [""] * (len(clean_row) - 1)
                        
                    cleaned_table.append(clean_row)

                # Stop building and break loop if we bled into body text
                if hit_paragraph and len(cleaned_table) < 2:
                    continue # Skip saving this bbox so the text extractor can pick it up
                    
                if len(cleaned_table) > 1:
                    table_bboxes.append(tb_rect) 
                    
                    table_md = "\n| " + " | ".join(cleaned_table[0]) + " |\n"
                    table_md += "|" + "|".join(["---"] * len(cleaned_table[0])) + "|\n"
                    for row in cleaned_table[1:]:
                        table_md += "| " + " | ".join(row) + " |\n"
                    table_md += "\n"
                    
                    layout_items.append({"type": "table", "bbox": table.bbox, "content": table_md})
                    
                    
            # ==========================================
            # 2. EXTRACT FIGURES (Unchanged)
            # ==========================================
            figure_zones = self._get_figure_zones(page_fitz, table_bboxes)
            for i, fz in enumerate(figure_zones):
                clip_rect = (fz + (-10, -10, 10, 10)).intersect(page_fitz.rect)
                if clip_rect.is_empty or clip_rect.get_area() < 1000: continue
                
                pix = page_fitz.get_pixmap(clip=clip_rect, dpi=150)
                img_filename = f"{self.pdf_name}_page_{page_num + 1}_fig_{i + 1}.png"
                img_filepath = os.path.join(self.output_dir, img_filename)
                pix.save(img_filepath)
                
                layout_items.append({
                    "type": "image",
                    "bbox": (fz.x0, fz.y0, fz.x1, fz.y1),
                    "content": f"\n\n![Figure on Page {page_num + 1}]({img_filepath})\n\n"
                })

            # ==========================================
            # 3. EXTRACT TEXT
            # ==========================================
            blocks = [b for b in page_fitz.get_text("dict")["blocks"] if b["type"] == 0]
            
            for b in blocks:
                b_rect = fitz.Rect(b["bbox"])
                cx = (b_rect.x0 + b_rect.x1) / 2
                cy = (b_rect.y0 + b_rect.y1) / 2
                
                # --- NEW: FOOTER / PAGE NUMBER FILTER ---
                # If the text is in the bottom 8% of the page
                if b_rect.y0 > page_fitz.rect.height * 0.92:
                    # Get the raw string to check if it's just a number
                    raw_text = "".join([s["text"] for l in b["lines"] for s in l["spans"]]).strip()
                    if re.match(r'^(page\s*)?\d+$', raw_text, re.IGNORECASE):
                        continue # Skip it, we already inject [PAGE: X] at the top
                # ----------------------------------------

                is_in_table = any(tb.contains(fitz.Point(cx, cy)) for tb in table_bboxes)
                is_in_figure = any((fz + (-10, -10, 10, 10)).contains(fitz.Point(cx, cy)) for fz in figure_zones)
                
                if is_in_table or is_in_figure:
                    continue
                
                if any(tb.contains(fitz.Point(cx, cy)) for tb in table_bboxes): continue
                if any((fz + (-10, -10, 10, 10)).contains(fitz.Point(cx, cy)) for fz in figure_zones): continue

                block_md = ""
                for line in b["lines"]:
                    line_text = ""
                    max_size = 0.0
                    for span in line["spans"]:
                        if span["text"].strip(): max_size = max(max_size, span["size"])
                        line_text += span["text"]
                        
                    line_text = line_text.strip()
                    if not line_text: continue
                        
                    prefix = self._get_header_prefix(max_size, base_size)
                    block_md += f"\n\n{prefix}{line_text}\n\n" if prefix else f"{line_text} "
                
                if block_md.strip():
                    clean_md = re.sub(r'[ \t]+', ' ', block_md.strip())
                    layout_items.append({"type": "text", "bbox": b["bbox"], "content": clean_md})

            # ==========================================
            # 4. PAGE ASSEMBLY WITH EXPLICIT PAGE MARKERS
            # ==========================================
            layout_items.sort(key=lambda item: (round(item["bbox"][0] / 200) * 200, item["bbox"][1]))
            
            # Inject an explicit Page Token that the Agent can parse!
            full_markdown.append(f"\n\n[PAGE: {page_num + 1}]\n")
            full_markdown.append("\n\n".join([item["content"] for item in layout_items]))
            full_markdown.append("\n---\n")

        doc_fitz.close()
        doc_plumber.close()
        
        return re.sub(r'\n{3,}', '\n\n', "\n".join(full_markdown))

# # ==========================================
# # EXAMPLE USAGE
# # ==========================================
# if __name__ == "__main__":
#     extractor = HybridPDFExtractor("test_paper.pdf", output_dir="extracted_paper_data")
#     markdown_output = extractor.extract_to_markdown()
    
#     with open("extracted_paper_data/document.md", "w", encoding="utf-8") as f:
#         f.write(markdown_output)
        
#     print("✅ Extraction complete! Images and Markdown are in 'extracted_paper_data/'")

# ==========================================
# 2. THE LLM DOCUMENT AGENT
# ==========================================
class StructuredDocumentAgent:
    """Parses Markdown into a searchable tree and exposes LLM tools."""
    
    def __init__(self, markdown_text: str, embedding_type: str = "local", local_model_name: str = "all-MiniLM-L6-v2"):
        self.raw_text = markdown_text
        self.sections: Dict[str, Any] = {}
        self.tree: List[Dict[str, Any]] = []
        self.embedding_type = embedding_type
        
        if self.embedding_type == "api":
            self.base_url = os.environ.get("EMBEDDING_BASE_URL", "http://100.83.246.7:8003/v1")
            self.api_key = os.environ.get("EMBEDDING_API_KEY", "EMPTY")
            self.api_model = os.environ.get("EMBEDDING_MODEL", "custom_bge_gpu1")
            print(f"🔗 Using External Embedding API: {self.api_model} at {self.base_url}")
        else:
            print(f"⚙️ Using Local SentenceTransformer: {local_model_name}")
            self.local_model = SentenceTransformer(local_model_name)

        print("🌳 Building document structure tree...")
        self._build_structure_tree()
        
        print("🔍 Building hybrid search indices (this takes a moment)...")
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self._build_search_indices()
        print("✅ Document Agent is ready for queries!")

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Helper method to route embedding generation to API or local model."""
        if not texts: 
            return np.array([])
        
        if self.embedding_type == "api":
            # 1. URL Sanitization (based on your snippet)
            endpoint = self.base_url.strip().rstrip("/")
            if not endpoint.endswith("/embeddings"):
                if endpoint.endswith("/v1"):
                    endpoint += "/embeddings"
                else:
                    endpoint += "/v1/embeddings"

            headers = {
                "Authorization": f"Bearer {self.api_key}", 
                "Content-Type": "application/json"
            }
            payload = {
                "input": texts, 
                "model": self.api_model
            }
            
            # 2. Resilient API Call with Timeout and Mock Fallback
            try:
                response = requests.post(endpoint, headers=headers, json=payload, timeout=5.0)
                if response.status_code == 200:
                    # Successfully got embeddings, sort them to maintain correct section order
                    data = sorted(response.json()["data"], key=lambda x: x.get("index", 0))
                    return np.array([item["embedding"] for item in data])
                else:
                    print(f"⚠️ [Embeddings] HTTP {response.status_code}. Falling back to mock embeddings.")
                    return self._generate_mock_embeddings(texts)
                    
            except Exception as e:
                print(f"⚠️ [Embeddings] Connection failed ({e}). Falling back to mock embeddings.")
                return self._generate_mock_embeddings(texts)
        else:
            return self.local_model.encode(texts)

    def _generate_mock_embeddings(self, texts: List[str], dim: int = 768) -> np.ndarray:
        """
        Generates random vector arrays as a fallback for local testing.
        Using dim=768 as a standard stand-in for BGE/OpenAI models.
        """
        # Create a matrix of random floats between 0 and 1, then normalize it 
        # so the cosine similarity math downstream doesn't break.
        mock_matrix = np.random.rand(len(texts), dim)
        norms = np.linalg.norm(mock_matrix, axis=1, keepdims=True)
        return mock_matrix / norms

    def _build_structure_tree(self):
        lines = self.raw_text.split('\n')
        current_section = {"id": "root", "title": "Document Start", "level": 0, "content": "", "children": [], "pages": set()}
        self.sections["root"] = current_section
        self.tree.append(current_section)
        
        stack = [current_section]
        section_counter = 1
        current_page = 1 # Default

        for line in lines:
            # Check if this line is a page marker
            page_match = re.search(r'\[PAGE:\s*(\d+)\]', line)
            if page_match:
                current_page = int(page_match.group(1))
                continue # Skip adding the raw marker to the content

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
                    current_section["pages"].add(current_page) # Tag content with current page

    def _build_search_indices(self):
        self.section_ids_ordered = [sec_id for sec_id, data in self.sections.items() if data["content"].strip()]
        texts = [self.sections[sec_id]["content"] for sec_id in self.section_ids_ordered]
        if not texts: return
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        self.section_embeddings = self._get_embeddings(texts)

    # --- LLM TOOLS ---
    def get_overview(self) -> str:
        def _section_chars(node: Dict) -> int:
            total = len(node.get("content", ""))
            for child in node.get("children", []):
                total += _section_chars(child)
            return total

        def build_toc(nodes: List[Dict], indent=""):
            toc = []
            for node in nodes:
                if node["id"] == "root" and node["title"] == "Document Start":
                    toc.extend(build_toc(node["children"], indent))
                    continue
                pages = sorted(node.get("pages", set()))
                page_str = (
                    f"p.{pages[0]}" if len(pages) == 1
                    else f"p.{pages[0]}-{pages[-1]}" if pages
                    else ""
                )
                sec_tok = max(1, _section_chars(node) // 4)
                tok_str = f"~{sec_tok/1000:.1f}k tok" if sec_tok >= 1000 else f"~{sec_tok} tok"
                loc = f" ({page_str}, {tok_str})" if page_str else f" ({tok_str})"
                toc.append(f"{indent}- [{node['id']}] {node['title']}{loc}")
                toc.extend(build_toc(node["children"], indent + "  "))
            return toc

        toc_lines = build_toc(self.tree)
        # Total across all non-root sections
        all_chars = sum(
            len(sec.get("content", ""))
            for sec_id, sec in self.sections.items()
            if sec_id != "root"
        )
        total_tok = max(1, all_chars // 4)
        grand_str = f"~{total_tok/1000:.1f}k" if total_tok >= 1000 else f"~{total_tok}"
        toc_lines.append("")
        toc_lines.append(f"📊 Total document content: {grand_str} tokens across {len(self.sections)} sections.")
        if total_tok <= 4000:
            toc_lines.append(
                "✅ Strategy: Document is SHORT. Use parallel tool calls — include multiple "
                "`read_document_section` calls in a single response."
            )
        elif total_tok <= 12000:
            toc_lines.append(
                "⚠️  Strategy: Document is MEDIUM. Group sections into 2-3 batches of parallel "
                "`read_document_section` calls."
            )
        else:
            toc_lines.append(
                "🚨 Strategy: Document is LARGE (risk of context overflow). "
                "You MUST use `execute_parallel_branches` to divide reading across branches. "
                "Do NOT read sections sequentially — context WILL overflow."
            )
        return "Document Overview:\n" + "\n".join(toc_lines)

    def read_details(self, section_id: str) -> str:
        """TOOL 2: Retrieves the detailed text for a specific section ID."""
        if section_id not in self.sections: return "Error: Section ID not found."
        section = self.sections[section_id]
        
        # Format the page numbers nicely (e.g., "Pages 15-17" or "Page 15")
        pages = sorted(list(section["pages"]))
        page_str = f"Page {pages[0]}" if len(pages) == 1 else f"Pages {pages[0]}-{pages[-1]}"
        
        result = f"--- SECTION: {section['title']} ({section_id}) | Found on {page_str} ---\n\n"
        result += section['content'].strip()
        
        if section["children"]:
            result += "\n\nSub-sections available:\n" + "\n".join([f"- {c['title']} ({c['id']})" for c in section["children"]])
        return result

    def search(self, query: str, top_k: int = 3) -> str:
        """TOOL 3: Performs a hybrid search across the document."""
        if not self.section_ids_ordered: return "Error: Document is empty."
        
        # TF-IDF & Semantic
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
    """Wrapper that extracts PDF to MD and initializes the Agent."""
    if not os.path.exists(pdf_filepath):
        raise FileNotFoundError(f"Could not find PDF at: {pdf_filepath}")
        
    extractor = HybridPDFExtractor(pdf_filepath)
    markdown_text = extractor.extract_to_markdown()
    
    md_filename = pdf_filepath.replace('.pdf', '_extracted.md')
    with open(md_filename, 'w', encoding='utf-8') as f:
        f.write(markdown_text)
    print(f"💾 Saved intermediate markdown to {md_filename}")
    
    return StructuredDocumentAgent(markdown_text, embedding_type=embedding_type, local_model_name=local_model_name)

# ==========================================
# 4. TEST CASES
# ==========================================
def test_pdf(args):
    print("\n" + "="*50)
    print(f"🚀 RUNNING PDF TEST on: {args.pdf_path}")
    print("="*50)
    
    if os.path.exists(args.pdf_path):
        agent = create_agent_from_pdf(
            pdf_filepath=args.pdf_path, 
            embedding_type=args.embedding_type, 
            local_model_name=args.local_model
        )
        print("\n--- TEST: GET OVERVIEW ---")
        print(agent.get_overview())
        
        print("\n--- TEST: SEARCH ---")
        print(agent.search("methodology or implementation details"))
    else:
        print(f"⚠️ PDF not found: '{args.pdf_path}'. Please check the path.")

def test_agent(args):
    print("\n" + "="*50)
    print("🚀 RUNNING AGENT MOCK TEXT TEST")
    print("="*50)
    
    sample_markdown = """
# Abstract
This paper introduces a novel framework for analyzing data.

# 1. Introduction
Data analysis is hard. We present a new method.
## 1.1 Background
Historically, people used abacuses.
## 1.2 Motivation
We need faster tools because data is growing.

# 2. Methodology
Our method relies on hybrid extraction.
## 2.1 The OpenDataLoader Approach
We use bounding boxes and layout analysis.
## 2.2 The PaperIndex Structure
We build trees out of documents so LLMs can read them.

# 3. Results
It works very well.
    """
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


# ==========================================
# MAIN EXECUTION & ARGPARSE
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Structured Document Agent Pipeline")
    
    # Test Execution Options
    parser.add_argument("--test", type=str, choices=["pdf", "agent", "both"], default="both", 
                        help="Which test to run: 'pdf', 'agent', or 'both'")
    parser.add_argument("--pdf_path", type=str, default="data/NVIDIA-Nemotron-3-Super-Technical-Report.pdf",
                        help="Path to the PDF file to process")
    
    # Embedding Configuration
    parser.add_argument("--embedding_type", type=str, choices=["local", "api"], default="local",
                        help="Use 'local' SentenceTransformers or external 'api'")
    parser.add_argument("--local_model", type=str, default="all-MiniLM-L6-v2",
                        help="HuggingFace model name for local embeddings")
    parser.add_argument("--api_url", type=str, default="http://100.83.246.7:8003/v1",
                        help="Base URL for the embedding API")
    parser.add_argument("--api_key", type=str, default="EMPTY",
                        help="API Key for the embedding API")
    parser.add_argument("--api_model", type=str, default="custom_bge_gpu1",
                        help="Model name to pass to the API")

    args = parser.parse_args()

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
python document_agent.py --test both
python BatchAgent/document_agent.py --test pdf --pdf_path "data/pdftest/NVIDIA-Nemotron-3-Super-Technical-Report.pdf" --embedding_type api --api_url "http://100.81.148.35:8003/v1" --api_model "BAAI/bge-large-zh-v1.5"
"""
