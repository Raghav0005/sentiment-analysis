import feedparser
import pandas as pd

def fetch_google_news(topic: str, max_result: int = 20) -> pd.DataFrame:
    """
    Fetch news articles from Google News RSS feed for a given topic.
    
    Args:
        topic (str): The search topic/query to fetch news for
        max_result (int, optional): Maximum number of articles to fetch. Defaults to 20.
    
    Returns:
        pd.DataFrame: DataFrame containing news articles with columns:
                     - source: News source (always "Google News")
                     - title: Article title
                     - summary: Article summary/description
                     - link: Article URL
    """
    
    query = topic.replace(" ", "+")
    rss_url = (
        f"https://news.google.com/rss/search?"
        f"q={query}&hl=en-US&gl=US&ceid=US:en"
    )

    feed = feedparser.parse(rss_url)
    entries = feed.entries[:max_result]
    data = []

    for e in entries:
        data.append({
            "source": "Google News",
            "title": e.title,
            "summary": getattr(e, "summary", ""),
            "link": e.link
        })
    
    return pd.DataFrame(data)
