import re
import json
import urllib.request
import traceback
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
from rich.console import Console
#pip install beautifulsoup4 markdownify httpx
import httpx
from bs4 import BeautifulSoup
import markdownify

# Import necessary components from your codebase
# Make sure these are accessible or adjust imports as needed
from BatchAgent.mini_batch_agent_libs import (
    is_git_repo, apply_patch_guarded, apply_write_files, extract_all_diffs,
    extract_write_file_actions, apply_fuzzy_patch, extract_files_from_diff,
    resolve_path, search_code, find_file, list_directory, read_file_chunk,
    run_bash_command
)
from BatchAgent.mini_batch_agent import (
    AgentAction, ActionToolCall, ActionWriteFile, ActionApplyDiff, ActionReplaceText
)

console = Console()

def fetch_and_parse_url(url: str) -> str:
    """
    Fetches a webpage and converts its main content to clean Markdown.
    """
    try:
        # We use a browser-like User-Agent to avoid basic anti-bot blocks
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        # Use httpx for a robust synchronous request (or async if you prefer)
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            
        html_content = response.text
        
        # Parse with BeautifulSoup to remove junk
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove scripts, styles, footers, navbars
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            element.decompose()
            
        # Try to find the main content area to reduce noise
        main_content = soup.find('main') or soup.find('article') or soup.find(id=re.compile('content|main', re.I)) or soup.body
        
        if not main_content:
            return "Error: Could not parse the main content of the page."

        # Convert the cleaned HTML to Markdown
        # We turn off image links and truncate very long lines
        md_text = markdownify.markdownify(
            str(main_content), 
            heading_style="ATX", 
            strip=['img', 'a'] # Strip links and images to save tokens, we only want text
        )
        
        # Clean up excessive blank lines
        md_text = re.sub(r'\n{3,}', '\n\n', md_text).strip()
        
        # Truncate to a reasonable length to prevent context overflow (e.g., 10,000 chars)
        max_chars = 15000 
        if len(md_text) > max_chars:
            md_text = md_text[:max_chars] + "\n\n... [Content Truncated due to length] ..."
            
        return md_text
        
    except httpx.HTTPError as e:
        return f"Network error while fetching URL: {str(e)}"
    except Exception as e:
        return f"Failed to parse URL: {str(e)}"
    
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
    write_actions = extract_write_file_actions(content)
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
    if not actions and len(allowlist) == 1:
        code_blocks = re.findall(r'```(?:python)?\s*(.*?)```', content, re.DOTALL)
        if len(code_blocks) == 1:
            block = code_blocks[0].strip()
            if "def " in block or "import " in block:
                actions.append(ActionWriteFile(path=allowlist[0], content=block))
        elif "def " in content or "import " in content:
            clean = content.strip()
            if clean.startswith("```python"): clean = clean[len("```python"):].strip()
            elif clean.startswith("```"): clean = clean[3:].strip()
            if clean.endswith("```"): clean = clean[:-3].strip()
            actions.append(ActionWriteFile(path=allowlist[0], content=clean))
            
    return actions


# ==========================================
# 2. Advanced Web Search (Domain-Aware)
# ==========================================

def perform_domain_aware_search(query: str, category: str, api_key: str) -> str:
    """
    Advanced search that routes queries to specific domains based on the category
    suggested by the LLM.
    """
    if not api_key or api_key == "EMPTY":
        return "System Error: SERPER_API_KEY is not configured."

    # Google Dorks specific to domains
    domain_filters = {
        "news": "site:reuters.com OR site:apnews.com OR site:bloomberg.com OR site:bbc.com/news",
        "code": "site:github.com OR site:stackoverflow.com OR site:docs.python.org",
        "academic": "site:arxiv.org OR site:nature.com OR site:sciencedirect.com",
        "general": ""
    }
    
    # Normalize category, default to general
    cat = category.lower().strip()
    if cat not in domain_filters:
        cat = "general"
        
    filter_str = domain_filters[cat]
    final_query = f"{query} {filter_str}".strip()
    
    console.print(f"[dim]Routing search -> Category: '{cat}', Final Query: '{final_query}'[/dim]")
    
    url = "https://google.serper.dev/search"
    req = urllib.request.Request(url, method="POST")
    req.add_header("X-API-KEY", api_key)
    req.add_header("Content-Type", "application/json")
    
    # Fetch 8 results if it's general, fewer if it's highly specific
    num_results = 8 if cat == "general" else 5
    data = json.dumps({"q": final_query, "num": num_results}).encode("utf-8")
    
    try:
        with urllib.request.urlopen(req, data=data, timeout=15) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            
            organic = res_data.get("organic", [])
            answer_box = res_data.get("answerBox", {})
            
            results = []
            
            # Prioritize Google's direct answer box if available
            if answer_box and "snippet" in answer_box:
                results.append(f"⭐ [Direct Answer]: {answer_box['snippet']}\n")
                
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
    def __init__(self, config, turn_dir: Path, allowlist: List[str]):
        self.config = config
        self.turn_dir = turn_dir
        self.allowlist = allowlist

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
                # --- Read / Search Tools ---
                if isinstance(action, ActionToolCall):
                    res = ""
                    name = action.name
                    args = action.args
                    
                    console.print(f"[bold magenta]🛠️ Tool Execution:[/bold magenta] {name}({args})")
                    
                    if name == "web_search":
                        # Now uses the advanced domain-aware search
                        query = args.get("query", "")
                        category = args.get("category", "general")
                        res = perform_domain_aware_search(query, category, self.config.serper_api_key)
                    
                    # [NEW] Add the read_url handler here
                    elif name == "read_url":
                        url = args.get("url", "")
                        console.print(f"[dim]Fetching webpage: {url}[/dim]")
                        # Assuming you put fetch_and_parse_url in this file
                        res = fetch_and_parse_url(url)
                        
                    elif name == "search_code":
                        res = search_code(args.get("query", ""))
                        
                    elif name == "find_file":
                        res = find_file(args.get("pattern", ""))
                        
                    elif name == "read_file_chunk":
                        res = read_file_chunk(args.get("filepath", ""), args.get("start_line", 1), args.get("end_line", 1000))
                        
                    elif name == "list_directory":
                        res = list_directory(args.get("dir_path", "."))
                        
                    elif name == "run_bash_command":
                        res = run_bash_command(args.get("command", ""))
                        
                    else:
                        res = f"Error: Unknown tool '{name}'"
                        
                    # Format the observation block for the LLM
                    tool_results.append(f"### Result for {name}\n```text\n{res}\n```")

                # --- Write / Patch Tools ---
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