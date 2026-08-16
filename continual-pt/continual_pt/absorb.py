"""
The absorb phase.

Researches X on the web, generates source-grounded practice examples.

The model does the cognitive work (generating queries, reading pages,
identifying gaps, writing practice). The runtime enforces structure
(trusted domains, source provenance, content hashing).
"""

import requests
import time
import hashlib
import json
import re
from bs4 import BeautifulSoup
from typing import List, Dict
from urllib.parse import urlparse


# --- Known sources for common ML topics ---
# Used as fallback when DuckDuckGo is blocked (e.g., from Modal's network)
KNOWN_SOURCES = {
    "lora": [
        "https://arxiv.org/abs/2106.09685",
        "https://arxiv.org/html/2106.09685v2",
        "https://huggingface.co/docs/peft/en/developer_guides/lora",
        "https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms",
    ],
    "grpo": [
        "https://arxiv.org/abs/2402.03300",
        "https://arxiv.org/html/2402.03300v2",
        "https://huggingface.co/docs/trl/main/en/grpo_trainer",
        "https://deepseekmodel.github.io/DeepSeek-R1-Report/",
    ],
    "diloco": [
        "https://arxiv.org/abs/2311.08105",
        "https://arxiv.org/html/2311.08105v2",
        "https://huggingface.co/blog/diloco",
        "https://www.fedml.ai/paper/diloco-distributed-optimization",
    ],
}


def get_known_sources(x: str) -> List[Dict]:
    """Get known sources for a topic based on keyword matching."""
    x_lower = x.lower()
    for keyword, urls in KNOWN_SOURCES.items():
        if keyword in x_lower:
            return [{"title": f"Source for {keyword}", "url": url, "snippet": ""} for url in urls]
    return []


# --- Web research ---

def duckduckgo_search(query: str, max_results: int = 5, delay: float = 2.0) -> List[Dict]:
    """Search DuckDuckGo HTML endpoint. Free, no API key.
    Falls back gracefully if DuckDuckGo is blocked."""
    print(f"[search] Querying: {query}")
    time.sleep(delay)

    url = "https://html.duckduckgo.com/html/"
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
    }
    data = {"q": query, "b": ""}

    try:
        resp = requests.post(url, headers=headers, data=data, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []

        for item in soup.select(".result"):
            title_elem = item.select_one(".result__title a")
            snippet_elem = item.select_one(".result__snippet")
            if title_elem:
                title = title_elem.get_text(strip=True)
                raw_url = title_elem.get("href", "")
                if "uddg=" in raw_url:
                    from urllib.parse import parse_qs, urlparse
                    parsed = urlparse(raw_url)
                    params = parse_qs(parsed.query)
                    actual_url = params.get("uddg", [raw_url])[0]
                else:
                    actual_url = raw_url
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                results.append({
                    "title": title,
                    "url": actual_url,
                    "snippet": snippet,
                })
            if len(results) >= max_results:
                break

        print(f"[search] Found {len(results)} results")
        return results
    except Exception as e:
        print(f"[search] Error: {e}")
        return []


def fetch_page(url: str, timeout: int = 15) -> Dict:
    """Fetch a web page and extract text content."""
    print(f"[fetch] {url[:80]}...")
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove scripts, styles
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        # Get title
        title = soup.title.string if soup.title else url

        # Get text
        text = soup.get_text(separator="\n", strip=True)
        # Truncate to reasonable length
        max_chars = 10000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n... [truncated]"

        # Hash for provenance
        content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

        return {
            "url": url,
            "title": title.strip() if title else url,
            "text": text,
            "content_hash": content_hash,
            "status": resp.status_code,
        }
    except Exception as e:
        print(f"[fetch] Error: {e}")
        return {"url": url, "title": url, "text": "", "content_hash": "", "status": 0, "error": str(e)}


# --- Research orchestration ---

RESEARCH_PROMPT = """You are researching: {x}

Generate 3 web search queries to find authoritative, independent sources about {x}.
Focus on:
- Official documentation or papers
- Independent analyses or experiments
- Technical breakdowns

Output ONLY a JSON list of 3 search query strings. No explanation.
Example: ["query 1", "query 2", "query 3"]
"""


def research_x(model, tokenizer, x: str, config) -> Dict:
    """Research X on the web. Returns sources, queries, and fetched content."""
    print(f"\n[absorb] Researching: {x}")

    # Step 1: Generate search queries
    from continual_pt.model import generate
    response = generate(model, tokenizer,
                        RESEARCH_PROMPT.format(x=x),
                        max_new_tokens=config.max_new_tokens_research,
                        temperature=config.temperature)

    # Parse queries
    queries = []
    try:
        # Try JSON parse
        queries = json.loads(response.strip())
        if not isinstance(queries, list):
            queries = [response.strip()]
    except json.JSONDecodeError:
        # Fallback: split by newlines or quotes
        import re
        queries = re.findall(r'"([^"]+)"', response)
        if not queries:
            queries = [response.strip()[:200]]

    queries = queries[:3]  # max 3 queries
    print(f"[absorb] Generated {len(queries)} queries: {queries}")

    # Step 2: Execute searches
    all_results = []
    for q in queries:
        results = duckduckgo_search(q, max_results=config.max_search_results,
                                     delay=config.search_delay)
        all_results.extend(results)

    # Fallback: if DuckDuckGo is blocked, use known sources
    if not all_results:
        known = get_known_sources(x)
        if known:
            print(f"[absorb] DuckDuckGo unavailable. Using {len(known)} known sources for {x}.")
            all_results = known

    # Deduplicate by URL
    seen_urls = set()
    unique_results = []
    for r in all_results:
        if r["url"] not in seen_urls:
            seen_urls.add(r["url"])
            unique_results.append(r)

    print(f"[absorb] {len(unique_results)} unique results")

    # Step 3: Fetch top pages
    pages = []
    for r in unique_results[:config.max_pages_to_read]:
        page = fetch_page(r["url"])
        if page.get("text"):
            pages.append(page)

    print(f"[absorb] Fetched {len(pages)} pages")

    return {
        "x": x,
        "queries": queries,
        "search_results": unique_results,
        "pages": pages,
    }


# --- Practice generation ---

PRACTICE_PROMPT = """You are creating practice examples to teach a model about: {x}

Here are source materials from web research:

{sources}

Generate 5 practice examples. Each example must be:
1. A question that tests understanding of {x}
2. A correct, detailed answer
3. The exact source URL and a quoted span from the source that supports the answer

Output as JSON array. Each element:
{{
  "question": "...",
  "answer": "...",
  "source_url": "...",
  "source_span": "exact quote from the source"
}}

Requirements:
- The answer must be grounded in the source span, not invented
- The source_span must be a verbatim quote from the fetched content
- Vary the questions: some about mechanism, some about specifics, some about tradeoffs
- No hallucinated URLs — only use URLs from the sources above

Output ONLY the JSON array. No explanation.
"""


def generate_practice(model, tokenizer, x: str, research: Dict, config) -> List[Dict]:
    """Generate source-grounded practice examples from research."""
    print(f"\n[absorb] Generating practice examples for: {x}")

    # Format sources for the prompt
    sources_text = ""
    for i, page in enumerate(research.get("pages", [])):
        sources_text += f"\n--- Source {i+1} ---\n"
        sources_text += f"URL: {page['url']}\n"
        sources_text += f"Title: {page['title']}\n"
        sources_text += f"Content (first 2000 chars):\n{page['text'][:2000]}\n"

    if not sources_text:
        sources_text = "(No pages fetched — generate examples from your knowledge, but mark source_url as 'model_internal')"

    from continual_pt.model import generate as gen
    response = gen(model, tokenizer,
                   PRACTICE_PROMPT.format(x=x, sources=sources_text),
                   max_new_tokens=config.max_new_tokens_research,
                   temperature=config.temperature)

    # Parse practice examples
    practice = []
    try:
        practice = json.loads(response.strip())
        if not isinstance(practice, list):
            practice = [practice]
    except json.JSONDecodeError:
        # Try to extract JSON array from response
        import re
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            try:
                practice = json.loads(match.group())
            except:
                pass

    # Validate each example has required fields
    valid_practice = []
    for p in practice:
        if isinstance(p, dict) and "question" in p and "answer" in p:
            p.setdefault("source_url", "")
            p.setdefault("source_span", "")
            valid_practice.append(p)

    print(f"[absorb] Generated {len(valid_practice)} valid practice examples")
    return valid_practice
