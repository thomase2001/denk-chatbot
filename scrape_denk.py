
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

print("🚀 Starting full scrape of denksummit.com...")

BASE_URL = "https://www.denksummit.com"
visited = set()
collected_text = []

def is_internal_link(link):
    return link.startswith("/") or urlparse(link).netloc == urlparse(BASE_URL).netloc

def scrape_page(url):
    try:
        if url.lower().endswith((".zip", ".png", ".jpg", ".jpeg", ".svg")):
            print(f"⏭️ Skipping file: {url}")
            return

        print(f"🔍 Scraping: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        if "text/html" not in response.headers.get("Content-Type", ""):
            print(f"⚠️ Skipping non-HTML content: {url}")
            return

        soup = BeautifulSoup(response.text, "html.parser")

        for script in soup(["script", "style", "noscript"]):
            script.decompose()

        text = soup.get_text(separator="\n", strip=True)
        collected_text.append(f"\n---\nURL: {url}\n{text}")

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            full_url = urljoin(BASE_URL, href)
            if is_internal_link(href) and full_url not in visited:
                visited.add(full_url)
                scrape_page(full_url)

    except Exception as e:
        print(f"❌ Failed: {url} — {e}")

# Start scraping
visited.add(BASE_URL)
scrape_page(BASE_URL)

# Save results
print("✅ Saving to denk_live.txt")
with open("denk_live.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(collected_text))

print("🎉 Done! File created: denk_live.txt")
