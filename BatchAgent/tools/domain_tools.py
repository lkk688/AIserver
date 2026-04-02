# domain_tools.py
from typing import List, Dict, Any
import datetime
import platform
import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import os
import subprocess
import tempfile
import random

# ==========================================
# Feature flag — set to True to re-enable domain-specific plugin tools.
# When False, DOMAIN_REGISTRY returns empty lists so no extra agent tools are loaded.
# Domain-specific search is handled by online_search_toolkit via the web_search
# category parameter (e.g. web_search(query="...", category="medical")).
# ==========================================
DOMAIN_TOOLS_ENABLED = False

# ==========================================
# 1. 医疗健康领域 (Medical)
# ==========================================
MEDICAL_TOOLS = [
    {
        "name": "search_pubmed",
        "description": "Search PubMed for medical and life sciences literature. Returns abstract summaries.",
        "properties": {
            "query": {"type": "string", "description": "Medical search query (e.g., 'COVID-19 mRNA vaccine efficacy')."},
            "max_results": {"type": "integer", "description": "Number of results to return."}
        },
        "required": ["query"]
    },
    {
        "name": "check_drug_interactions",
        "description": "Check for known interactions between two or more drugs using medical databases.",
        "properties": {
            "drugs": {"type": "array", "items": {"type": "string"}, "description": "List of drug names."}
        },
        "required": ["drugs"]
    }
]

def search_pubmed(query: str, max_results: int = 3) -> str:
    """
    API: NCBI E-utilities (PubMed)
    Process to apply API: 
    1. Register at NCBI to get an API key for higher rate limits (optional but recommended): https://www.ncbi.nlm.nih.gov/account/
    2. Pass the API key using '&api_key=YOUR_API_KEY' in the URL.
    """
    try:
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {"db": "pubmed", "term": query, "retmode": "json", "retmax": max_results}
        qs = urllib.parse.urlencode(params)
        with urllib.request.urlopen(f"{base_url}?{qs}") as response:
            data = json.loads(response.read().decode())
        ids = data.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return f"No PubMed results found for '{query}'."
        
        summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        summary_params = {"db": "pubmed", "id": ",".join(ids), "retmode": "json"}
        sqs = urllib.parse.urlencode(summary_params)
        with urllib.request.urlopen(f"{summary_url}?{sqs}") as response:
            s_data = json.loads(response.read().decode())
        
        results = []
        for uid in ids:
            item = s_data.get("result", {}).get(uid, {})
            title = item.get("title", "")
            results.append(f"Title: {title} (PMID: {uid})")
        return "\n".join(results)
    except Exception as e:
        return f"Error querying PubMed: {e}"

def check_drug_interactions(drugs: list) -> str:
    """
    API: NIH RxNorm / REST APIs
    Process to apply API:
    1. The National Library of Medicine (NLM) provides free APIs (e.g., RxClass, Interaction API). No API key is needed.
    2. Documentation: https://lhncbc.nlm.nih.gov/RxNav/APIs/InteractionAPIs.html
    """
    try:
        if len(drugs) < 2:
            return "Need at least two drugs to check for interactions."
        rxcuis = []
        for drug in drugs:
            url = f"https://rxnav.nlm.nih.gov/REST/rxcui.json?name={urllib.parse.quote(drug)}"
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode())
                group = data.get("idGroup", {})
                if "rxnormId" in group:
                    rxcuis.append(group["rxnormId"][0])
                else:
                    return f"Drug '{drug}' not found in RxNorm."
        
        sources = "+".join(rxcuis)
        url = f"https://rxnav.nlm.nih.gov/REST/interaction/list.json?rxcuis={sources}"
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
            if "fullInteractionTypeGroup" in data:
                return f"Interactions found for {drugs}. Please consult a medical professional for details."
            else:
                return f"No known major interactions found for {drugs} in NLM database."
    except Exception as e:
        return f"Error checking interactions: {e}"

# ==========================================
# 2. 学术研究领域 (Academic)
# ==========================================
ACADEMIC_TOOLS = [
    {
        "name": "search_arxiv",
        "description": "Search ArXiv for pre-print papers in physics, math, and computer science.",
        "properties": {
            "query": {"type": "string"},
            "category": {"type": "string", "enum": ["cs.AI", "cs.CL", "math", "physics"]}
        },
        "required": ["query"]
    },
    {
        "name": "extract_paper_summary",
        "description": "Given a DOI or ArXiv ID, extract the core methodology and results of the paper.",
        "properties": {
            "paper_id": {"type": "string"}
        },
        "required": ["paper_id"]
    }
]

def search_arxiv(query: str, category: str = "cs.AI") -> str:
    """
    API: ArXiv API
    Process to apply API:
    1. ArXiv API is free and does not require an API key for reasonable usage.
    2. Documentation: https://info.arxiv.org/help/api/user-manual.html
    """
    try:
        search_query = f"cat:{category} AND all:{urllib.parse.quote(query)}"
        url = f"http://export.arxiv.org/api/query?search_query={search_query}&start=0&max_results=3"
        with urllib.request.urlopen(url) as response:
            xml_data = response.read()
        root = ET.fromstring(xml_data)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)
        if not entries:
            return f"No ArXiv results found for '{query}' in {category}."
        
        results = []
        for entry in entries:
            title = entry.find("atom:title", ns).text.replace("\n", " ").strip()
            summary = entry.find("atom:summary", ns).text.replace("\n", " ").strip()[:200]
            results.append(f"Title: {title}\nSummary: {summary}...")
        return "\n\n".join(results)
    except Exception as e:
        return f"Error querying ArXiv: {e}"

def extract_paper_summary(paper_id: str) -> str:
    """
    API: ArXiv API (or CrossRef for DOIs)
    Process to apply API: no key required for ArXiv.
    """
    try:
        url = f"http://export.arxiv.org/api/query?id_list={paper_id}"
        with urllib.request.urlopen(url) as response:
            xml_data = response.read()
        root = ET.fromstring(xml_data)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entry = root.find("atom:entry", ns)
        if entry is None:
            return f"Paper ID '{paper_id}' not found on ArXiv."
        title = entry.find("atom:title", ns).text.replace("\n", " ").strip()
        summary = entry.find("atom:summary", ns).text.strip()
        return f"Title: {title}\nSummary for {paper_id}:\n{summary}"
    except Exception as e:
        return f"Error fetching paper summary: {e}"

# ==========================================
# 3. 新闻与时效领域 (News)
# ==========================================
NEWS_TOOLS = [
    {
        "name": "fetch_latest_news",
        "description": "Fetch the latest news headlines from major RSS feeds based on a topic.",
        "properties": {
            "topic": {"type": "string", "description": "e.g., 'technology', 'finance', 'politics'"},
            "region": {"type": "string", "description": "e.g., 'US', 'China', 'Global'"}
        },
        "required": ["topic"]
    }
]

def fetch_latest_news(topic: str, region: str = "Global") -> str:
    """
    API: NewsAPI
    Process to apply API:
    1. Go to https://newsapi.org/
    2. Register for a free account to get an API key. 
    3. Set the environment variable NEWS_API_KEY.
    """
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        return f"[News API Mock] NEWS_API_KEY is not set. Mock {region} {topic} news for {today}: 1. AI models achieve new breakthrough. 2. Global markets rally."
    try:
        url = f"https://newsapi.org/v2/everything?q={urllib.parse.quote(topic)}&sortBy=publishedAt&pageSize=3&apiKey={api_key}"
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
        articles = data.get("articles", [])
        if not articles:
            return f"No news found for topic '{topic}'."
        
        results = []
        for idx, article in enumerate(articles, 1):
            title = article.get("title", "")
            results.append(f"{idx}. {title}")
        return f"Latest news on {topic} ({region}):\n" + "\n".join(results)
    except Exception as e:
        return f"Error fetching news: {e}"

# ==========================================
# 4. 软件工程领域 (Software Engineering)
# ==========================================
SOFTWARE_ENG_TOOLS = [
    {
        "name": "lint_code_snippet",
        "description": "Run a linter (like flake8 or pylint) on a code snippet to find syntax and style errors.",
        "properties": {
            "code": {"type": "string", "description": "The raw code to lint."},
            "language": {"type": "string", "description": "Programming language (e.g., 'python', 'javascript')"}
        },
        "required": ["code", "language"]
    },
    {
        "name": "query_stack_overflow",
        "description": "Search StackOverflow API for solutions to specific programming errors.",
        "properties": {
            "error_message": {"type": "string", "description": "The exact error traceback or description."}
        },
        "required": ["error_message"]
    }
]

def lint_code_snippet(code: str, language: str) -> str:
    """
    Runs a local linter via subprocess.
    Process: Ensure 'pylint' is installed for Python, or 'eslint' for JavaScript.
    """
    language = language.lower()
    try:
        if language == "python":
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode='w') as f:
                f.write(code)
                temp_path = f.name
            try:
                result = subprocess.run(["pylint", temp_path], capture_output=True, text=True)
                return result.stdout if result.stdout else "No linting issues found."
            finally:
                os.remove(temp_path)
        elif language in ["javascript", "js"]:
            with tempfile.NamedTemporaryFile(suffix=".js", delete=False, mode='w') as f:
                f.write(code)
                temp_path = f.name
            try:
                result = subprocess.run(["eslint", temp_path], capture_output=True, text=True)
                return result.stdout if result.stdout else "No linting issues found."
            finally:
                os.remove(temp_path)
        else:
            return f"Linting for language '{language}' is not currently supported."
    except FileNotFoundError:
        return f"Linter for {language} not found. Please 'pip install pylint' or 'npm install -g eslint'."
    except Exception as e:
        return f"Error linting code: {e}"

def query_stack_overflow(error_message: str) -> str:
    """
    API: StackExchange API
    Process to apply API:
    1. Go to https://stackapps.com/apps/oauth/register to register an application and get a 'key'
    2. Optional for low volume. Set STACKEXCHANGE_KEY env var if you have one.
    """
    try:
        key_param = ""
        se_key = os.getenv("STACKEXCHANGE_KEY")
        if se_key:
            key_param = f"&key={se_key}"
            
        url = f"https://api.stackexchange.com/2.3/search/advanced?order=desc&sort=relevance&q={urllib.parse.quote(error_message)}&site=stackoverflow&filter=withbody{key_param}"
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
        items = data.get("items", [])
        if not items:
            return f"No StackOverflow answers found for error: {error_message[:30]}..."
        
        top_item = items[0]
        title = top_item.get("title", "")
        link = top_item.get("link", "")
        is_answered = top_item.get("is_answered", False)
        return f"StackOverflow Top Result: {title}\nAnswered: {is_answered}\nLink: {link}"
    except Exception as e:
        return f"Error querying StackOverflow: {e}"

# ==========================================
# 5. 数学领域 (Mathematics)
# ==========================================
MATH_TOOLS = [
    {
        "name": "evaluate_math_expression",
        "description": "Evaluate complex mathematical expressions, calculus, or algebra using a symbolic math engine (like SymPy).",
        "properties": {
            "expression": {"type": "string", "description": "Math expression (e.g., 'integrate(x**2, x)', 'solve(x**2 - 4, x)')."}
        },
        "required": ["expression"]
    }
]

def evaluate_math_expression(expression: str) -> str:
    """
    Evaluates math using SymPy locally.
    Process: pip install sympy
    """
    try:
        import sympy
        from sympy.parsing.sympy_parser import parse_expr
        parsed = parse_expr(expression)
        result = sympy.simplify(parsed)
        return f"[Math Engine] Evaluation of '{expression}' resulted in: {result}"
    except ImportError:
        return "SymPy is not installed. Please 'pip install sympy'."
    except Exception as e:
        return f"Error evaluating expression: {e}"

# ==========================================
# 6. 科学领域 (Science)
# ==========================================
SCIENCE_TOOLS = [
    {
        "name": "query_wolfram_alpha",
        "description": "Query Wolfram Alpha for scientific facts, chemical properties, or physics computations.",
        "properties": {
            "query": {"type": "string", "description": "Scientific query (e.g., 'boiling point of tungsten', 'distance to mars')."}
        },
        "required": ["query"]
    }
]

def query_wolfram_alpha(query: str) -> str:
    """
    API: Wolfram Alpha LLM API or Short Answer API
    Process to apply API:
    1. Go to https://developer.wolframalpha.com/
    2. Sign up and get an AppID.
    3. Set environment variable WOLFRAM_ALPHA_APPID.
    """
    app_id = os.getenv("WOLFRAM_ALPHA_APPID")
    if not app_id:
        return f"[Wolfram API Mock] WOLFRAM_ALPHA_APPID is not set. Mock answer to '{query}': Approximately 42."
    try:
        url = f"http://api.wolframalpha.com/v1/result?appid={app_id}&i={urllib.parse.quote(query)}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            result = response.read().decode()
        return f"[Wolfram API] Answer to '{query}': {result}"
    except urllib.error.HTTPError as e:
        if e.code == 501:
            return f"Wolfram Alpha could not understand the query: '{query}'."
        return f"HTTP error {e.code}"
    except Exception as e:
        return f"Error querying Wolfram Alpha: {e}"

# ==========================================
# 7. 语言教育领域 (Language Education)
# ==========================================
LANGUAGE_TOOLS = [
    {
        "name": "explain_grammar",
        "description": "Provide a detailed linguistic and grammatical breakdown of a sentence in a target language.",
        "properties": {
            "sentence": {"type": "string"},
            "language": {"type": "string", "description": "Language of the sentence (e.g., 'French', 'Japanese')"}
        },
        "required": ["sentence", "language"]
    },
    {
        "name": "generate_vocabulary_quiz",
        "description": "Generate a short vocabulary quiz for a specific CEFR level (A1-C2).",
        "properties": {
            "language": {"type": "string"},
            "cefr_level": {"type": "string", "enum": ["A1", "A2", "B1", "B2", "C1", "C2"]}
        },
        "required": ["language", "cefr_level"]
    }
]

def explain_grammar(sentence: str, language: str) -> str:
    """
    API: OpenAI API (or any LLM)
    Process to apply API:
    1. Register at platform.openai.com
    2. Get API key and set OPENAI_API_KEY environment variable.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return f"[Language API Mock] OPENAI_API_KEY not set. Mock grammar for {language}: '{sentence}' uses subject-verb-object structure."
    
    try:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": f"Explain the grammar of this {language} sentence: '{sentence}'"}]
        }
        req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=json.dumps(payload).encode(), headers=headers)
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode())
        return res_data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error connecting to LLM API: {e}"

def generate_vocabulary_quiz(language: str, cefr_level: str) -> str:
    """
    API: OpenAI API (or any LLM)
    Process to apply API:
    Same as above, needs OPENAI_API_KEY environment variable.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return f"[Language API Mock] OPENAI_API_KEY not set. Mock quiz for {language} {cefr_level}: 1. Hello 2. Goodbye 3. Thank you"
    
    try:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        prompt = f"Generate a short 3-question multiple choice vocabulary quiz for {language} at {cefr_level} level. Provide answers at the end."
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}]
        }
        req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=json.dumps(payload).encode(), headers=headers)
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode())
        return res_data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error connecting to LLM API: {e}"

# ==========================================
# 8. 商业与金融领域 (Business & Finance)
# ==========================================
BUSINESS_TOOLS = [
    {
        "name": "get_stock_quote",
        "description": "Get real-time or historical stock market data for a given ticker symbol.",
        "properties": {
            "ticker": {"type": "string", "description": "Stock ticker (e.g., 'AAPL', 'TSLA')"},
            "period": {"type": "string", "enum": ["current", "1mo", "1y"]}
        },
        "required": ["ticker"]
    },
    {
        "name": "fetch_company_filings",
        "description": "Fetch recent SEC filings (like 10-K, 10-Q) for a public company.",
        "properties": {
            "ticker": {"type": "string"}
        },
        "required": ["ticker"]
    }
]

def get_stock_quote(ticker: str, period: str = "current") -> str:
    """
    Requires 'yfinance' library to fetch data from Yahoo Finance. No API key needed.
    Process: pip install yfinance
    """
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        query_period = "1d" if period == "current" else period
        hist = stock.history(period=query_period)
        if hist.empty:
            return f"Could not fetch data for ticker {ticker}."
        last_close = hist['Close'].iloc[-1]
        return f"[Finance API] {ticker}: Latest close price is ${last_close:.2f} (period: {period})"
    except ImportError:
        return "yfinance is not installed. Please 'pip install yfinance'."
    except Exception as e:
        return f"Error fetching stock data: {e}"

def fetch_company_filings(ticker: str) -> str:
    """
    API: SEC EDGAR API
    Process to apply API:
    1. The SEC EDGAR API is free but requires a descriptive User-Agent header (Name Email).
    2. Ensure SEC_USER_AGENT env var is set, or use a default one.
    """
    try:
        user_agent = os.getenv("SEC_USER_AGENT", "MyBot/1.0 (info@example.com)")
        headers = {"User-Agent": user_agent}
        
        tickers_url = "https://www.sec.gov/files/company_tickers.json"
        req = urllib.request.Request(tickers_url, headers=headers)
        with urllib.request.urlopen(req) as response:
            ticker_map = json.loads(response.read().decode())
        
        cik = None
        for key, company in ticker_map.items():
            if company["ticker"].upper() == ticker.upper():
                cik = str(company["cik_str"]).zfill(10)
                break
        
        if not cik:
            return f"Ticker {ticker} not found in SEC EDGAR database."
            
        sub_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        req = urllib.request.Request(sub_url, headers=headers)
        with urllib.request.urlopen(req) as response:
            sub_data = json.loads(response.read().decode())
        
        recent = sub_data.get("filings", {}).get("recent", {})
        if not recent or "form" not in recent:
            return f"No recent filings found for {ticker}."
            
        forms = recent["form"][:3]
        dates = recent["filingDate"][:3]
        
        filing_info = [f"{f} filed on {d}" for f, d in zip(forms, dates)]
        return f"Recent SEC filings for {ticker} (CIK: {cik}):\n" + "\n".join(filing_info)
    except Exception as e:
        return f"Error fetching SEC filings: {e}"

# ==========================================
# 9. 电脑助手/系统控制领域 (Computer Assistant)
# ==========================================
COMPUTER_ASSISTANT_TOOLS = [
    {
        "name": "get_system_status",
        "description": "Get current OS, CPU, RAM, and Disk usage of the host machine.",
        "properties": {},
        "required": []
    },
    {
        "name": "manage_process",
        "description": "List top processes or kill a specific process by name/PID. Use with extreme caution.",
        "properties": {
            "action": {"type": "string", "enum": ["list_top", "kill"]},
            "process_name": {"type": "string", "description": "Target process name if action is kill."}
        },
        "required": ["action"]
    }
]

def get_system_status() -> str:
    """
    Uses psutil to get actual system status.
    Process: pip install psutil
    """
    try:
        import psutil
        os_info = f"{platform.system()} {platform.release()}"
        cpu = psutil.cpu_percent(interval=0.5)
        ram = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        ram_used_gb = ram.used / (1024**3)
        ram_total_gb = ram.total / (1024**3)
        disk_free_gb = disk.free / (1024**3)
        
        return (f"[System API] OS: {os_info}. CPU Usage: {cpu}%. "
                f"RAM: {ram_used_gb:.1f}GB/{ram_total_gb:.1f}GB used. "
                f"Disk: {disk_free_gb:.1f}GB free on root.")
    except ImportError:
        return "psutil is not installed. Please 'pip install psutil'."
    except Exception as e:
        return f"Error getting system status: {e}"

def manage_process(action: str, process_name: str = "") -> str:
    """
    Uses psutil to list or kill processes.
    Process: pip install psutil
    """
    try:
        import psutil
        if action == "list_top":
            procs = []
            for p in psutil.process_iter(['name', 'memory_info']):
                try:
                    procs.append(p.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            procs = sorted(procs, key=lambda p: p.get('memory_info').rss if p.get('memory_info') else 0, reverse=True)[:5]
            res = "[System API] Top processes by memory:\n"
            for idx, p in enumerate(procs, 1):
                mem_mb = p['memory_info'].rss / (1024**2) if p['memory_info'] else 0
                res += f"{idx}. {p['name']} ({mem_mb:.1f} MB)\n"
            return res.strip()
            
        elif action == "kill" and process_name:
            killed = 0
            for p in psutil.process_iter(['name']):
                if p.info['name'] == process_name:
                    try:
                        p.terminate()
                        killed += 1
                    except psutil.AccessDenied:
                        return f"Access denied to kill {process_name}."
            if killed > 0:
                return f"[System API] {killed} instances of {process_name} terminated."
            else:
                return f"[System API] Process {process_name} not found."
        else:
            return "Invalid action or missing process name."
    except ImportError:
        return "psutil is not installed. Please 'pip install psutil'."
    except Exception as e:
        return f"Error managing process: {e}"

# ==========================================
# 10. 公司客服/销售专属 (Company Customer Service/Sales)
# ==========================================
COMPANY_SUPPORT_TOOLS = [
    {
        "name": "query_internal_knowledge_base",
        "description": "Search the company's internal Confluence/Notion for policies, product specs, or troubleshooting guides.",
        "properties": {
            "query": {"type": "string"}
        },
        "required": ["query"]
    },
    {
        "name": "check_customer_crm",
        "description": "Look up a customer's profile, purchase history, and active tickets using their email or ID.",
        "properties": {
            "customer_id": {"type": "string", "description": "Email or Customer ID"}
        },
        "required": ["customer_id"]
    },
    {
        "name": "create_support_ticket",
        "description": "Create a formal support ticket for a user issue in Jira/Zendesk.",
        "properties": {
            "customer_id": {"type": "string"},
            "issue_summary": {"type": "string"},
            "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"]}
        },
        "required": ["customer_id", "issue_summary", "priority"]
    }
]

def query_internal_knowledge_base(query: str) -> str:
    """
    API: Internal Confluence/Notion/Zendesk API
    Process to apply API:
    1. Assuming an internal REST API endpoint e.g., https://internal.company.com/api/kb/search
    2. Authentication via internal API Token passed in headers.
    """
    try:
        base_url = os.getenv("INTERNAL_KB_URL")
        token = os.getenv("INTERNAL_API_TOKEN")
        if not base_url or not token:
            return f"[Knowledge Base Mock] Search for '{query}': Our enterprise policy states refunds are allowed within 30 days of purchase."
            
        headers = {"Authorization": f"Bearer {token}"}
        req = urllib.request.Request(f"{base_url}?q={urllib.parse.quote(query)}", headers=headers)
        with urllib.request.urlopen(req) as response:
            return json.dumps(json.loads(response.read().decode()))
    except Exception as e:
        return f"Error querying internal KB: {e}"

def check_customer_crm(customer_id: str) -> str:
    """
    API: Salesforce/HubSpot or internal CRM API
    Process to apply API:
    1. Authenticate with OAuth2 or API token provided by the CRM vendor.
    2. Set CRM_API_URL and CRM_API_TOKEN.
    """
    try:
        base_url = os.getenv("CRM_API_URL")
        token = os.getenv("CRM_API_TOKEN")
        if not base_url or not token:
            return f"[CRM API Mock] Customer {customer_id}: VIP Tier. Last purchase: 'Enterprise Server Rack'. Active tickets: 0."
            
        headers = {"Authorization": f"Bearer {token}"}
        req = urllib.request.Request(f"{base_url}/customers/{customer_id}", headers=headers)
        with urllib.request.urlopen(req) as response:
            return json.dumps(json.loads(response.read().decode()))
    except Exception as e:
        return f"Error querying CRM: {e}"

def create_support_ticket(customer_id: str, issue_summary: str, priority: str) -> str:
    """
    API: Jira/Zendesk API
    Process to apply API:
    1. Generate an API token from the account settings in Jira or Zendesk.
    2. Use Basic Auth (Email:Token) or Bearer token. Set TICKET_API_URL and TICKET_API_TOKEN.
    """
    try:
        base_url = os.getenv("TICKET_API_URL")
        token = os.getenv("TICKET_API_TOKEN")
        if not base_url or not token:
            ticket_id = f"TKT-{random.randint(1000, 9999)}"
            return f"[Ticket System Mock] Created {priority} priority ticket #{ticket_id} for {customer_id}: '{issue_summary}'."
            
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {"customer": customer_id, "summary": issue_summary, "priority": priority}
        req = urllib.request.Request(f"{base_url}/tickets", data=json.dumps(payload).encode(), headers=headers)
        with urllib.request.urlopen(req) as response:
            return "Ticket created successfully: " + response.read().decode()
    except Exception as e:
        return f"Error creating ticket: {e}"


# ==========================================
# 领域路由字典 (DOMAIN_REGISTRY)
# ==========================================
_FULL_DOMAIN_REGISTRY = {
    "medical": MEDICAL_TOOLS,
    "academic": ACADEMIC_TOOLS,
    "news": NEWS_TOOLS,
    "software_eng": SOFTWARE_ENG_TOOLS,
    "math": MATH_TOOLS,
    "science": SCIENCE_TOOLS,
    "language": LANGUAGE_TOOLS,
    "business": BUSINESS_TOOLS,
    "assistant": COMPUTER_ASSISTANT_TOOLS,
    "sales_support": COMPANY_SUPPORT_TOOLS
}

# When DOMAIN_TOOLS_ENABLED is False, domain-specific tools are not injected
# into the agent — search routing is done by online_search_toolkit instead.
DOMAIN_REGISTRY = {
    k: (v if DOMAIN_TOOLS_ENABLED else [])
    for k, v in _FULL_DOMAIN_REGISTRY.items()
}

# ==========================================
# 工具函数映射表 (DOMAIN_FUNCTIONS)
# ==========================================
DOMAIN_FUNCTIONS = {
    # Medical
    "search_pubmed": search_pubmed,
    "check_drug_interactions": check_drug_interactions,
    # Academic
    "search_arxiv": search_arxiv,
    "extract_paper_summary": extract_paper_summary,
    # News
    "fetch_latest_news": fetch_latest_news,
    # Software Eng
    "lint_code_snippet": lint_code_snippet,
    "query_stack_overflow": query_stack_overflow,
    # Math & Science
    "evaluate_math_expression": evaluate_math_expression,
    "query_wolfram_alpha": query_wolfram_alpha,
    # Language
    "explain_grammar": explain_grammar,
    "generate_vocabulary_quiz": generate_vocabulary_quiz,
    # Business
    "get_stock_quote": get_stock_quote,
    "fetch_company_filings": fetch_company_filings,
    # Assistant
    "get_system_status": get_system_status,
    "manage_process": manage_process,
    # Support
    "query_internal_knowledge_base": query_internal_knowledge_base,
    "check_customer_crm": check_customer_crm,
    "create_support_ticket": create_support_ticket
}