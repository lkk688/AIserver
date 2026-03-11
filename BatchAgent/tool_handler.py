import re
import json
import urllib.request
import traceback
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
from rich.console import Console
#pip install beautifulsoup4 markdownify httpx pypdf playwright
import httpx
from bs4 import BeautifulSoup
import markdownify

# Optional: pypdf for PDF extraction
try:
    from pypdf import PdfReader
    import io
    _PYPDF_AVAILABLE = True
except ImportError:
    _PYPDF_AVAILABLE = False

# Optional: playwright for JS-heavy page fallback
try:
    from playwright.sync_api import sync_playwright
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    _PLAYWRIGHT_AVAILABLE = False

import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import necessary components from your codebase
# Make sure these are accessible or adjust imports as needed
from BatchAgent.mini_batch_agent_libs import (
    is_git_repo, apply_patch_guarded, apply_write_files, extract_all_diffs,
    extract_write_file_actions_v2, apply_fuzzy_patch, extract_files_from_diff,
    resolve_path, search_code, find_file, list_directory, read_file_chunk,
    run_bash_command, run_shell
)
from BatchAgent.mini_batch_agent import (
    AgentAction, ActionToolCall, ActionWriteFile, ActionApplyDiff, ActionReplaceText
)

console = Console()

# ---- Content cleaning constants ----
_NOISE_TAGS = ["script", "style", "nav", "footer", "header", "aside",
               "noscript", "iframe", "svg", "form", "button"]
_NOISE_PATTERNS = [
    re.compile(r'(cookie|gdpr|privacy|consent|subscribe|newsletter)', re.I),
    re.compile(r'^\s{0,4}[\|\-\–\•→✓✗★☆©®™]{1,3}\s*$'),  # icon-only lines
]
_MAX_CONTENT_CHARS = 15000
_PLAYWRIGHT_THRESHOLD = 300  # chars — below this, try playwright fallback


def _clean_text_content(text: str) -> str:
    """Shared post-processing: collapse blank lines, filter noise lines, truncate."""
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Drop very short lines that are likely nav/menu artifacts (< 3 words AND < 25 chars)
        if len(stripped) < 25 and len(stripped.split()) < 3 and stripped:
            continue
        # Drop lines matching noise patterns (cookie banners, GDPR boilerplate)
        if any(p.search(stripped) for p in _NOISE_PATTERNS):
            continue
        cleaned.append(line)

    text = '\n'.join(cleaned)
    # Collapse 3+ consecutive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    # Truncate
    if len(text) > _MAX_CONTENT_CHARS:
        text = text[:_MAX_CONTENT_CHARS] + "\n\n... [Content Truncated] ..."
    return text


def _extract_pdf_text(raw_bytes: bytes) -> str:
    """Extract text from a PDF binary using pypdf."""
    if not _PYPDF_AVAILABLE:
        return "Error: pypdf is not installed. Run: pip install pypdf"
    try:
        reader = PdfReader(io.BytesIO(raw_bytes))
        pages = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(f"--- Page {i+1} ---\n{page_text.strip()}")
        if not pages:
            return "Error: PDF appears to have no extractable text (may be image-only/scanned)."
        full_text = "\n\n".join(pages)
        return _clean_text_content(full_text)
    except Exception as e:
        return f"Error extracting PDF text: {e}"


def _fetch_via_playwright(url: str) -> str:
    """Fallback: use a headless browser to render JS-heavy pages."""
    if not _PLAYWRIGHT_AVAILABLE:
        return ""
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/120.0.0.0 Safari/537.36"
            )
            page.goto(url, timeout=20000, wait_until="domcontentloaded")
            # Extract visible text from body after JS execution
            text = page.evaluate("() => document.body.innerText")
            browser.close()
        return _clean_text_content(text or "")
    except Exception as e:
        console.print(f"[dim]Playwright fallback failed: {e}[/dim]")
        return ""


def fetch_and_parse_url(url: str) -> str:
    """
    Fetches a URL and returns clean text content.
    Handles: HTML pages, PDFs, plain text files.
    Falls back to Playwright for JS-heavy pages with thin content.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml,application/pdf;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        with httpx.Client(timeout=20.0, follow_redirects=True) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            raw_bytes = response.content

        # ── Layer 1: File-type detection ──────────────────────────────────
        url_path = url.split("?")[0].lower()
        is_pdf = (
            url_path.endswith(".pdf")
            or "application/pdf" in content_type
            or "pdf" in content_type
        )
        is_plaintext = any(url_path.endswith(ext) for ext in (".txt", ".md", ".csv", ".log", ".rst"))

        if is_pdf:
            console.print(f"[dim]Detected PDF, extracting text via pypdf...[/dim]")
            return _extract_pdf_text(raw_bytes)

        if is_plaintext:
            console.print(f"[dim]Detected plain text file, decoding...[/dim]")
            text = raw_bytes.decode("utf-8", errors="replace")
            return _clean_text_content(text)

        # ── Layer 2: HTML parsing ─────────────────────────────────────────
        try:
            html_content = raw_bytes.decode("utf-8", errors="replace")
        except Exception:
            html_content = response.text

        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove noisy elements
        for tag in _NOISE_TAGS:
            for el in soup.find_all(tag):
                el.decompose()
        # Also remove elements with cookie/ad-related classes or IDs
        for el in soup.find_all(True, attrs={"class": re.compile(r'cookie|consent|banner|popup|overlay|ad-|ads-', re.I)}):
            el.decompose()
        for el in soup.find_all(True, attrs={"id": re.compile(r'cookie|consent|banner|popup|overlay', re.I)}):
            el.decompose()

        # Find the richest content container
        main_content = (
            soup.find('article')
            or soup.find('main')
            or soup.find(id=re.compile(r'content|main|body|post', re.I))
            or soup.find(class_=re.compile(r'content|main|article|post|entry', re.I))
            or soup.body
        )

        if main_content:
            md_text = markdownify.markdownify(
                str(main_content),
                heading_style="ATX",
                strip=['img', 'a'],
            )
        else:
            # Last-resort: raw text extraction from soup
            md_text = soup.get_text(separator='\n', strip=True)

        md_text = _clean_text_content(md_text)

        # ── Layer 3: Playwright fallback for thin/empty content ───────────
        if len(md_text) < _PLAYWRIGHT_THRESHOLD and _PLAYWRIGHT_AVAILABLE:
            console.print(f"[dim]Content too thin ({len(md_text)} chars), trying Playwright fallback...[/dim]")
            playwright_text = _fetch_via_playwright(url)
            if len(playwright_text) > len(md_text):
                console.print(f"[dim]Playwright returned {len(playwright_text)} chars, using that.[/dim]")
                return playwright_text

        if not md_text:
            return "Error: Could not extract any meaningful content from this page."

        return md_text

    except httpx.HTTPStatusError as e:
        # For PDFs behind auth walls, try playwright
        if _PLAYWRIGHT_AVAILABLE:
            console.print(f"[dim]HTTP {e.response.status_code} error, trying Playwright...[/dim]")
            result = _fetch_via_playwright(url)
            if result:
                return result
        return f"HTTP Error {e.response.status_code} while fetching URL: {url}"
    except httpx.HTTPError as e:
        return f"Network error while fetching URL: {str(e)}"
    except Exception as e:
        return f"Failed to parse URL: {str(e)}\n{traceback.format_exc(limit=3)}"
    
# ==========================================
# 1. Action Parsers
# ==========================================

def parse_text_actions(content: str, allowlist: List[str]) -> List[AgentAction]:
    """
    Fallback text parser to convert plain text markdown into AgentAction protocol.
    Optimized for robustness.
    """
    actions = []
    
    # 1. WRITE_FILE (Format B)
    write_actions = extract_write_file_actions_v2(content)
    if write_actions:
        for path, text in write_actions:
            target_path = resolve_path(path, allowlist)
            if target_path:
                actions.append(ActionWriteFile(path=str(target_path), content=text))
                
    # 2. Unified Diff (Format A)
    diff = extract_all_diffs(content)
    if diff:
        actions.append(ActionApplyDiff(diff_text=diff))
        
    # 3. Interactive Tool Tags (Format C fallback)
    tool_patterns = {
        "search_code": r'(?:<tool_call>\s*)?<search_code>(.*?)</search_code>(?:\s*</tool_call>)?',
        "find_file": r'(?:<tool_call>\s*)?<find_file>(.*?)</find_file>(?:\s*</tool_call>)?',
        "list_directory": r'(?:<tool_call>\s*)?<list_directory>\s*<dir_path>(.*?)</dir_path>\s*</list_directory>(?:\s*</tool_call>)?',
        "run_bash_command": r'(?:<tool_call>\s*)?<run_bash_command>\s*<command>(.*?)</command>\s*</run_bash_command>(?:\s*</tool_call>)?',
    }
    
    for tool_name, pattern in tool_patterns.items():
        for match in re.finditer(pattern, content, re.DOTALL):
            # For list_directory, handle empty inner tag
            arg_val = match.group(1).strip() if match.group(1) else "." 
            actions.append(ActionToolCall(name=tool_name, args={list(BASE_TOOLS_SCHEMA[tool_name].keys())[0]: arg_val}))

    # Special handling for read_file_chunk (multi-args)
    for match in re.finditer(r'(?:<tool_call>\s*)?<read_file_chunk>\s*<filepath>(.*?)</filepath>(.*?)</read_file_chunk>(?:\s*</tool_call>)?', content, re.DOTALL):
        fpath = match.group(1).strip()
        rest = match.group(2)
        m_s = re.search(r'<start_line>(\d+)</start_line>', rest)
        m_e = re.search(r'<end_line>(\d+)</end_line>', rest)
        actions.append(ActionToolCall(
            name="read_file_chunk", 
            args={"filepath": fpath, "start_line": int(m_s.group(1)) if m_s else 1, "end_line": int(m_e.group(1)) if m_e else 1000}
        ))

    # Special handling for web_search (supports category)
    for match in re.finditer(r'(?:<tool_call>\s*)?<web_search>\s*<query>(.*?)</query>(.*?)</web_search>(?:\s*</tool_call>)?', content, re.DOTALL):
        query = match.group(1).strip()
        rest = match.group(2)
        cat_match = re.search(r'<category>(.*?)</category>', rest)
        cat = cat_match.group(1).strip() if cat_match else "general"
        actions.append(ActionToolCall(name="web_search", args={"query": query, "category": cat}))

    # Fallback to simple <web_search>query</web_search>
    for match in re.finditer(r'(?:<tool_call>\s*)?<web_search>(?![\s\S]*<query>)(.*?)</web_search>(?:\s*</tool_call>)?', content, re.DOTALL):
        actions.append(ActionToolCall(name="web_search", args={"query": match.group(1).strip(), "category": "general"}))

    # 4. Extreme Fallbacks if no explicit format found and we only have 1 target file
    # if not actions and len(allowlist) == 1:
    #     code_blocks = re.findall(r'```(?:python)?\s*(.*?)```', content, re.DOTALL)
    #     if len(code_blocks) == 1:
    #         block = code_blocks[0].strip()
    #         if "def " in block or "import " in block:
    #             actions.append(ActionWriteFile(path=allowlist[0], content=block))
    #     elif "def " in content or "import " in content:
    #         clean = content.strip()
    #         if clean.startswith("```python"): clean = clean[len("```python"):].strip()
    #         elif clean.startswith("```"): clean = clean[3:].strip()
    #         if clean.endswith("```"): clean = clean[:-3].strip()
    #         actions.append(ActionWriteFile(path=allowlist[0], content=clean))
            
    return actions


# ==========================================
# 2. Advanced Web Search (Domain-Aware)
# ==========================================

def perform_domain_aware_search(query: str, category: str, api_key: str) -> str:
    """
    Advanced search that routes queries to specific domains based on the category.
    Category values match DOMAIN_REGISTRY keys in domain_tools.py plus common aliases.
    """
    if not api_key or api_key == "EMPTY":
        return "System Error: SERPER_API_KEY is not configured."

    # ── Domain filter map (covers all DOMAIN_REGISTRY keys + common aliases) ──
    # Filters are deliberately broad — multiple high-quality sources, not single-site
    domain_filters: Dict[str, str] = {
        # Core DOMAIN_REGISTRY keys
        "news":         "site:reuters.com OR site:apnews.com OR site:bbc.com OR site:bloomberg.com OR site:theguardian.com",
        "academic":     "site:arxiv.org OR site:scholar.google.com OR site:semanticscholar.org OR site:pubmed.ncbi.nlm.nih.gov OR site:researchgate.net",
        "medical":      "site:pubmed.ncbi.nlm.nih.gov OR site:medlineplus.gov OR site:nih.gov OR site:mayoclinic.org OR site:webmd.com",
        "software_eng": "site:stackoverflow.com OR site:github.com OR site:dev.to OR site:docs.python.org OR site:pypi.org",
        "math":         "site:math.stackexchange.com OR site:artofproblemsolving.com OR site:mathworld.wolfram.com OR site:khanacademy.org OR site:brilliant.org",
        "science":      "site:nature.com OR site:sciencedirect.com OR site:phys.org OR site:science.org OR site:wolframalpha.com",
        "language":     "site:en.wiktionary.org OR site:languageguide.org OR site:bbc.co.uk/languages OR site:italki.com",
        "business":     "site:sec.gov OR site:finance.yahoo.com OR site:bloomberg.com OR site:investopedia.com OR site:marketwatch.com",
        "assistant":    "site:superuser.com OR site:askubuntu.com OR site:serverfault.com OR site:apple.stackexchange.com",
        "sales_support":"site:zendesk.com OR site:hubspot.com OR site:salesforce.com OR site:freshdesk.com",
        # Common aliases
        "code":         "site:github.com OR site:stackoverflow.com OR site:docs.python.org OR site:pypi.org OR site:realpython.com",
        "finance":      "site:sec.gov OR site:finance.yahoo.com OR site:bloomberg.com OR site:investopedia.com",
        "health":       "site:pubmed.ncbi.nlm.nih.gov OR site:nih.gov OR site:mayoclinic.org OR site:webmd.com",
        "programming":  "site:stackoverflow.com OR site:github.com OR site:realpython.com OR site:docs.python.org",
        "research":     "site:arxiv.org OR site:semanticscholar.org OR site:scholar.google.com OR site:jstor.org",
        # Default fallback
        "general":      "",
    }

    # ── Category alias normalisation ──────────────────────────────────────────
    _aliases: Dict[str, str] = {
        "software":            "software_eng",
        "software_engineering":"software_eng",
        "engineering":         "software_eng",
        "medicine":            "medical",
        "biology":             "medical",
        "physics":             "science",
        "chemistry":           "science",
        "mathematics":         "math",
        "maths":               "math",
        "statistics":          "math",
        "economics":           "business",
        "stock":               "business",
        "stocks":              "business",
        "investment":          "business",
        "support":             "sales_support",
        "crm":                 "sales_support",
        "system":              "assistant",
        "computer":            "assistant",
        "paper":               "academic",
        "papers":              "academic",
        "python":              "code",
        "javascript":          "code",
        "js":                  "code",
    }

    cat = category.lower().strip()
    cat = _aliases.get(cat, cat)  # resolve alias first
    if cat not in domain_filters:
        cat = "general"  # safe fallback

    filter_str = domain_filters[cat]
    # Only append site filter when non-empty (avoids polluting general queries)
    final_query = f"{query} {filter_str}".strip() if filter_str else query

    console.print(f"[dim]Routing search -> Category: '{cat}', Final Query: '{final_query}'[/dim]")

    url = "https://google.serper.dev/search"
    req = urllib.request.Request(url, method="POST")
    req.add_header("X-API-KEY", api_key)
    req.add_header("Content-Type", "application/json")

    # Tune result count by category — dense domains get more results
    if cat == "general":
        num_results = 8
    elif cat in ("math", "academic", "science", "medical"):
        num_results = 6  # more results for research-heavy domains
    else:
        num_results = 5

    data = json.dumps({"q": final_query, "num": num_results}).encode("utf-8")

    try:
        with urllib.request.urlopen(req, data=data, timeout=15) as response:
            res_data = json.loads(response.read().decode("utf-8"))

            organic = res_data.get("organic", [])
            answer_box = res_data.get("answerBox", {})
            knowledge_graph = res_data.get("knowledgeGraph", {})

            results = []

            # Prioritize Google's direct answer box
            if answer_box and "snippet" in answer_box:
                results.append(f"⭐ [Direct Answer]: {answer_box['snippet']}\n")

            # Surface knowledge graph description when available (great for math/science)
            if knowledge_graph and "description" in knowledge_graph:
                results.append(f"📚 [Knowledge Panel]: {knowledge_graph['description']}\n")

            for i, item in enumerate(organic):
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                link = item.get("link", "")
                date = item.get("date", "")
                date_str = f" ({date})" if date else ""
                results.append(f"[{i+1}] {title}{date_str}\n{snippet}\nURL: {link}\n")

            return "\n".join(results) if results else f"No results found for '{final_query}'."

    except urllib.error.URLError as e:
        return f"Network Error during search: {str(e)}"
    except Exception as e:
        return f"Search processing failed: {str(e)}"


# ==========================================
# 3. Universal Tool Handler Class
# ==========================================

class UniversalToolHandler:
    """
    Central dispatcher that takes AgentActions (parsed from JSON or Text)
    and executes them in the local environment or via external APIs.
    """
    def __init__(self, config, turn_dir: Path, allowlist: List[str], dynamic_tools_mapping: dict = None):
        self.config = config
        self.turn_dir = turn_dir
        self.allowlist = allowlist
        self.dynamic_tools_mapping = dynamic_tools_mapping or {}

    def _execute_code_mutations(self, actions: List[AgentAction], full_content: str) -> bool:
        """
        Executes file modification actions. 
        Returns True if any file was successfully changed.
        """
        changes_applied = False
        
        for action in actions:
            if isinstance(action, ActionApplyDiff):
                diff = action.diff_text
                (self.turn_dir / "patch.diff").write_text(diff, encoding="utf-8")
                
                # Strategy 1: Strict Git Apply
                if is_git_repo():
                    applied = apply_patch_guarded(diff, self.turn_dir, auto_approve=self.config.auto_approve)
                    if applied:
                        changes_applied = True
                        console.print("[green]Strict diff applied successfully.[/green]")
                        continue
                
                # Strategy 2: Fuzzy Patch
                file_diffs = re.split(r'(?=^diff --git )', diff, flags=re.MULTILINE)
                fuzzy_successes, fuzzy_total = 0, 0
                fuzzy_logs = ["\n--- Fuzzy Patch Attempt ---"]
                
                for fd in file_diffs:
                    if not fd.strip().startswith("diff --git"): continue
                    fuzzy_total += 1
                    match = re.search(r'diff --git a/\S+ b/(\S+)', fd)
                    if match:
                        raw_path = match.group(1)
                        fuzzy_logs.append(f"Processing diff for: {raw_path}")
                        target_path = resolve_path(raw_path, self.allowlist)
                        if target_path:
                            if apply_fuzzy_patch(target_path, fd, log_buffer=fuzzy_logs):
                                fuzzy_successes += 1
                                fuzzy_logs.append(">> Success")
                            else:
                                fuzzy_logs.append(">> Failed")
                        else:
                            fuzzy_logs.append(f"[red]Skipping diff for unresolved path: {raw_path}[/red]")
                
                try:
                    with open(self.turn_dir / "apply.log", "a", encoding="utf-8") as f:
                        f.write("\n".join(fuzzy_logs) + "\n")
                except Exception as e:
                    console.print(f"Failed to append to apply.log: {e}")

                if fuzzy_successes > 0:
                    changes_applied = True
                    console.print(f"[green]Fuzzy patch applied ({fuzzy_successes}/{fuzzy_total} files).[/green]")
                else:
                    console.print(f"[red]Patch failed ({fuzzy_total} hunks). See patch.diff and apply.log.[/red]")
                    
            elif isinstance(action, ActionWriteFile):
                target_path = resolve_path(action.path, self.allowlist)
                if target_path:
                    if apply_write_files([(str(target_path), action.content)], self.allowlist, self.turn_dir):
                        changes_applied = True
                else:
                    console.print(f"[red]Skipping WRITE_FILE for unresolved path: {action.path}[/red]")
                    
            elif isinstance(action, ActionReplaceText):
                target_path = resolve_path(action.path, self.allowlist)
                if target_path and target_path.exists():
                    file_text = target_path.read_text(encoding="utf-8")
                    if action.old_text in file_text:
                        new_text = file_text.replace(action.old_text, action.new_text, 1)
                        target_path.write_text(new_text, encoding="utf-8")
                        console.print(f"[green]Replaced text in {target_path}[/green]")
                        changes_applied = True
                    else:
                        console.print(f"[red]search_and_replace failed: 'old_text' not found in {target_path}[/red]")
                else:
                    console.print(f"[red]search_and_replace skipped: unresolved or missing file {action.path}[/red]")

        # Fallback: Check for extractable new files if diff methods failed entirely
        if not changes_applied and full_content and extract_all_diffs(full_content):
            console.print("[yellow]All patch methods failed. Checking for extractable new files in diff...[/yellow]")
            diff_files = extract_files_from_diff(extract_all_diffs(full_content))
            if diff_files and apply_write_files(diff_files, self.allowlist, self.turn_dir):
                changes_applied = True
                console.print("[green]Wrote new files extracted from diff.[/green]")

        return changes_applied

    def execute(self, actions: List[AgentAction], full_llm_content: str) -> Tuple[bool, str]:
        """
        Executes all actions.
        Returns: (has_mutation_occurred: bool, observation_string: str)
        """
        tool_results = []
        has_mutation = False
        mutation_actions = []

        for action in actions:
            try:
                # -----------------------------------------
                # 1. Read / Search Tools (The "Eyes")
                # -----------------------------------------
                if isinstance(action, ActionToolCall):
                    res = ""
                    name = action.name
                    args = action.args
                    
                    # === [防洪补丁]：终端显示截断 ===
                    # 遍历 args，把超过 80 字符的超长字符串截断，保护终端 UI
                    safe_args = {}
                    for k, v in args.items():
                        val_str = str(v)
                        safe_args[k] = val_str[:80] + "... [TRUNCATED]" if len(val_str) > 80 else v
                    
                    console.print(f"[bold magenta]🛠️ Tool Execution:[/bold magenta] {name}({safe_args})")
                    
                    # === [防线补丁 1]：拦截捏造的 XML 写入工具 ===
                    if name.upper() in ["WRITE_FILE", "WRITE", "CREATE_FILE"]:
                        res = "System Guardrail: Do NOT use XML tool calls for writing files. You MUST use the pure Markdown Format B (`WRITE_FILE: filepath\\n<<<CONTENT...`) outside of any XML tags."
                        tool_results.append(f"### Result for {name}\n```text\n{res}\n```")
                        continue

                    # --- 正常工具路由 ---
                    if name == "web_search":
                        query = args.get("query", "")
                        category = args.get("category", "general")
                        res = perform_domain_aware_search(query, category, getattr(self.config, 'serper_api_key', ''))
                    
                    elif name == "read_url":
                        url = args.get("url", "")
                        console.print(f"[dim]Fetching webpage: {url}[/dim]")
                        # 如果没有 fetch_and_parse_url 函数，请确保导入或替换
                        res = fetch_and_parse_url(url) if 'fetch_and_parse_url' in globals() else f"read_url not implemented for {url}"
                        
                    elif name == "search_code":
                        res = search_code(args.get("query", ""))
                        
                    elif name == "find_file":
                        res = find_file(args.get("pattern", ""))
                        
                    elif name == "read_file_chunk":
                        res = read_file_chunk(args.get("filepath", ""), args.get("start_line", 1), args.get("end_line", 1000))
                        
                    elif name == "list_directory":
                        res = list_directory(args.get("dir_path", "."))
                        
                    elif name == "run_bash_command":
                        cmd = args.get("command", "")
                        # === [防线补丁 2]：封杀使用 Bash 写长代码（保护上下文） ===
                        if len(cmd) > 300 and ("cat >" in cmd or "echo " in cmd):
                            res = "System Guardrail: Command too long. Writing files via bash is strictly forbidden. Use Markdown Format B (`WRITE_FILE: ...`) instead."
                        else:
                            res = run_bash_command(cmd)
                    
                    elif name == "json_parse_error":
                        res = args.get("error", "JSON Parse Error")
                    
                    # =========================================================
                    # [NEW] 动态执行自定义工具 (沙盒执行，保障绝对安全)
                    # =========================================================
                    elif name in self.dynamic_tools_mapping:
                        script_path = self.dynamic_tools_mapping[name]
                        target_script = self.config.workspace_dir / script_path
                        
                        if not target_script.exists():
                            res = f"System Error: Cannot find the script '{script_path}'. Did you write the file first?"
                        else:
                            import json
                            # 将大模型传来的 JSON arguments 序列化，作为命令行参数传给脚本
                            args_json = json.dumps(args).replace("'", "'\\''") # 简单的 Bash 单引号转义
                            cmd = f"python3 {script_path} '{args_json}'"
                            
                            # 调用系统的沙盒命令行执行，完美杜绝危险代码直接污染母舰内存
                            code, out = run_shell(cmd, cap=10000, sandbox_container=getattr(self.config, 'sandbox_container', None))
                            
                            if code == 0:
                                res = out.strip()
                            else:
                                res = f"[Custom Tool Execution Error (Exit {code})]:\n{out}"
                                
                    # [NEW] 拦截并执行领域特定工具
                    else:
                        from BatchAgent.domain_tools import DOMAIN_FUNCTIONS
                        if name in DOMAIN_FUNCTIONS:
                            try:
                                func = DOMAIN_FUNCTIONS[name]
                                # 动态解包字典参数并调用函数
                                res = func(**args) 
                            except Exception as e:
                                res = f"Error executing domain tool {name}: {e}"
                        else:
                            res = f"Error: Unknown tool '{name}'"
                        
                    # Format the observation block for the LLM
                    tool_results.append(f"### Result for {name}\n```text\n{res}\n```")

                # -----------------------------------------
                # 2. Write / Patch Tools (The "Hands")
                # -----------------------------------------
                elif isinstance(action, (ActionWriteFile, ActionApplyDiff, ActionReplaceText)):
                    has_mutation = True
                    mutation_actions.append(action)

            except Exception as e:
                error_msg = f"Error executing tool {getattr(action, 'name', 'mutation')}: {str(e)}"
                tool_results.append(error_msg)
                console.print(f"[red]{error_msg}[/red]")

        # Process all code mutations atomically
        if mutation_actions:
            console.print("[cyan]📝 Applying Code Mutations to Disk...[/cyan]")
            success = self._execute_code_mutations(mutation_actions, full_llm_content)
            if success:
                tool_results.append("### System Action\nFile modifications were successfully applied to the disk. Please proceed to verify or conclude.")
            else:
                tool_results.append("### System Error\nFailed to apply file modifications. The diff may be malformed or the file path is incorrect. Please try again using WRITE_FILE format.")

        return has_mutation, "\n\n".join(tool_results)


# --- Helper for text parser ---
# We keep a minimal mock schema here just for text parsing keys
BASE_TOOLS_SCHEMA = {
    "search_code": {"query": ""},
    "find_file": {"pattern": ""},
    "list_directory": {"dir_path": ""},
    "run_bash_command": {"command": ""}
}