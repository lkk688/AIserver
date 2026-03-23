import feedparser
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import schedule
import time
import logging
from typing import List, Dict, Any

# 1. Setup basic logging for the Agent's background tasks
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 2. A curated list of common, high-quality RSS feeds categorized by topic
DEFAULT_RSS_FEEDS = {
    "Technology & Startups": [
        "https://techcrunch.com/feed/",
        "https://news.ycombinator.com/rss",          # Hacker News
        "https://www.wired.com/feed/rss"
    ],
    "Artificial Intelligence": [
        "https://news.google.com/rss/search?q=Artificial+Intelligence&hl=en-US&gl=US&ceid=US:en",
        "https://www.technologyreview.com/feed/"     # MIT Technology Review
    ],
    "Global News": [
        "https://rss.nytimes.com/services/xml/rss/nyt/World.xml", # NYT World News
        "https://feeds.bbci.co.uk/news/world/rss.xml",            # BBC World News
        "https://www.theguardian.com/world/rss"
    ]
}

def fetch_rss_articles(feed_urls: List[str], max_items: int = 3) -> List[Dict[str, Any]]:
    """
    [Agent Tool] Fetches and extracts the latest articles from a list of RSS feed URLs.
    
    Args:
        feed_urls (List[str]): A list of RSS feed URLs to scrape.
        max_items (int): The maximum number of latest articles to retrieve per feed.
        
    Returns:
        List[Dict]: A list of dictionaries containing 'title', 'link', 'summary', and 'source'.
    """
    logging.info(f"Starting to fetch articles from {len(feed_urls)} feeds...")
    collected_articles = []
    
    for url in feed_urls:
        try:
            # Parse the RSS feed
            feed = feedparser.parse(url)
            feed_title = feed.feed.get('title', 'Unknown Source')
            logging.info(f"Successfully fetched feed: {feed_title}")
            
            # Extract the top 'max_items' entries
            for entry in feed.entries[:max_items]:
                # Handle cases where summary/description might be missing or under different tags
                raw_summary = entry.get("summary", entry.get("description", "No summary available."))
                
                article = {
                    "source": feed_title,
                    "title": entry.get("title", "Untitled"),
                    "link": entry.get("link", url),
                    "summary": raw_summary[:300] + "..." if len(raw_summary) > 300 else raw_summary
                }
                collected_articles.append(article)
                
        except Exception as e:
            logging.error(f"Error fetching RSS feed {url}: {e}")
            
    logging.info(f"Total articles collected: {len(collected_articles)}")
    return collected_articles

def discover_rss_feeds(website_url: str) -> List[Dict[str, str]]:
    """
    [Agent Tool] Scans a given website URL to find hidden RSS or Atom feed links.
    Use this when the user asks to subscribe to a generic website.
    
    Args:
        website_url (str): The homepage URL of the website to scan.
        
    Returns:
        List[Dict]: A list of found feeds with their 'title', 'url', and 'type'.
    """
    logging.info(f"Attempting to discover RSS feeds on: {website_url}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    found_feeds = []
    
    try:
        # Request the webpage content with a 10-second timeout
        response = requests.get(website_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for standard RSS/Atom link tags in the <head>
        for link in soup.find_all('link', type=True):
            link_type = link['type'].lower()
            if 'application/rss+xml' in link_type or 'application/atom+xml' in link_type:
                href = link.get('href')
                if href:
                    # Convert relative URLs (e.g., '/feed') to absolute URLs
                    full_url = urljoin(website_url, href)
                    found_feeds.append({
                        "title": link.get('title', 'Unnamed Feed'),
                        "url": full_url,
                        "type": link_type
                    })
                    
        return found_feeds

    except requests.exceptions.RequestException as e:
        logging.error(f"Network error while scanning {website_url}: {e}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error while scanning {website_url}: {e}")
        return []

def scheduled_agent_task():
    """
    The main job that the Agent executes on a schedule.
    """
    logging.info("--- Running Scheduled RSS Aggregation ---")
    
    # Flatten our categorized dictionary into a single list of URLs
    all_urls = [url for category in DEFAULT_RSS_FEEDS.values() for url in category]
    
    # Fetch articles
    articles = fetch_rss_articles(all_urls, max_items=2)
    
    # Example output: printing the results. 
    # In a real Agent, you would pass 'articles' to an LLM for summarization here.
    for i, article in enumerate(articles, 1):
        print(f"\n[{i}] {article['source']} | {article['title']}")
        print(f"Link: {article['link']}")
    
    logging.info("--- Scheduled Task Completed ---")

# ==========================================
# Main Execution & Scheduler
# ==========================================
if __name__ == "__main__":
    # 1. Test the Discovery Tool
    test_site = "https://techcrunch.com"
    feeds = discover_rss_feeds(test_site)
    print(f"\nDiscovered feeds for {test_site}:")
    for f in feeds:
        print(f" -> {f['title']}: {f['url']}")

    # 2. Run the fetch task once immediately
    print("\nRunning initial fetch...")
    scheduled_agent_task()
    
    # 3. Setup the schedule (e.g., run every 6 hours)
    schedule.every(6).hours.do(scheduled_agent_task)
    
    logging.info("Scheduler started. Waiting for the next execution window...")
    
    # Keep the script running
    try:
        while True:
            schedule.run_pending()
            time.sleep(60) # Sleep for 60 seconds before checking the schedule again
    except KeyboardInterrupt:
        logging.info("Scheduler stopped by user.")

"""
pip install feedparser requests beautifulsoup4 schedule
"""
