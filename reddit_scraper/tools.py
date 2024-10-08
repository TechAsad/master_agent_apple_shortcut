import praw
import pandas as pd

import os
import re
from dotenv import load_dotenv


import json

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time

from fake_useragent import UserAgent
import pandas as pd

import asyncio
import asyncpraw
import async_timeout


load_dotenv()

# Replace with your Reddit app credentials
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET =  os.getenv("CLIENT_SECRET")
USER_AGENT = os.getenv("USER_AGENT") # You can use any descriptive user agent 




# Initialize PRAW with credentials
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)


## get subreddits
def search_subreddits(query, limit_per_word=5):
    words = query.split(',')
    all_subreddits = []
    
    for word in words:
        try:
            subreddits = reddit.subreddits.search(word, limit=limit_per_word)
            all_subreddits.extend([subreddit.display_name for subreddit in subreddits])
        except Exception as e:
            print(f"Error searching subreddits with query '{word}': {e}")

    return list(set(all_subreddits))


## Async function to fetch subreddits with a timeout
async def get_top_and_new_posts_async(subreddit_name, limit=15, timeout=15):
    subreddit = reddit.subreddit(subreddit_name)
    try:
        with async_timeout.timeout(timeout):
            #top_posts = await asyncio.to_thread(lambda: list(subreddit.top(limit=limit)))
            new_posts = await asyncio.to_thread(lambda: list(subreddit.new(limit=limit)))
            #return top_posts, new_posts
            return new_posts
    except asyncio.TimeoutError:
        print(f"Timeout fetching posts from {subreddit_name}. Skipping...")
        return [], []
    except Exception as e:
        print(f"Error fetching posts from {subreddit_name}: {e}")
        return [], []

## Async function to get top-level comments (no nested comments)
async def get_comments_from_post_async(post, limit=30):
    try:
        post.comments.replace_more(limit=0)  # Fetch only top-level comments
        comments = await asyncio.to_thread(lambda: [comment.body for comment in post.comments[:limit]])
        return comments
    except Exception as e:
        print(f"Error fetching comments from post {post.id}: {e}")
        return []

## Async function to scrape comments for multiple subreddits concurrently
async def scrape_reddit_comments_async(subreddits, timeout=15):
    all_comments = []
    failed_subreddits = []
    
    # Create tasks for concurrent subreddit scraping
    tasks = []
    for subreddit_name in subreddits:
        tasks.append(scrape_subreddit_async(subreddit_name, timeout=timeout))
    
    # Await and process results of all tasks
    results = await asyncio.gather(*tasks)
    for result in results:
        subreddit_name, comments = result
        if comments:
            all_comments.extend(comments)
        else:
            failed_subreddits.append(subreddit_name)
    
    if failed_subreddits:
        print(f"Failed to retrieve data from these subreddits: {', '.join(failed_subreddits)}")
    
    return all_comments

## Async function to scrape a single subreddit
async def scrape_subreddit_async(subreddit_name, timeout=15):
    print(f"Scraping subreddit: {subreddit_name}")
    #top_posts, 
    new_posts = await get_top_and_new_posts_async(subreddit_name, timeout=timeout)
    
    #if not top_posts and not new_posts:
    if not new_posts:# Skip if no posts were retrieved
        return subreddit_name, []
    
    # Process top and new posts
    all_comments = []
    for post in new_posts: # top_posts + new_posts:
        comments = await get_comments_from_post_async(post, limit=60)  # Fetch up to 60 comments
        all_comments.extend(comments)
    
    return subreddit_name, all_comments

## Main function to scrape reddit comments asynchronously
async def reddit_comments_async(sub_reddits):
    subreddits = sub_reddits.split(',')
    subreddits = [sub.strip() for sub in subreddits]
    
    comments = await scrape_reddit_comments_async(subreddits)
    print(f"Total comments scraped: {len(comments)}")
    
    if comments:
        print(f"Comments for {subreddits} scraped")
    else:
        print(f"No comments scraped from {subreddits}.")
    
    return comments[:250]

# Running the async scraper
def reddit_comments(sub_reddits):
    return asyncio.run(reddit_comments_async(sub_reddits))





### filter comments tool


from fuzzywuzzy import fuzz

def filter_comments(comments):
    unwanted_keywords = ['https']
    # Filter out comments containing unwanted keywords and comments with less than 4 words
    filtered_comments = []
    for comment in comments[:250]:
        if len(comment.split()) >= 4 and not any(keyword.lower() in comment.lower() for keyword in unwanted_keywords):
            filtered_comments.append(comment)

    # Remove comments with links
    filtered_comments = [comment for comment in filtered_comments if 'http' not in comment and 'https' not in comment]
    
    # Remove duplicate comments that are 80% similar or more
    def is_duplicate(comment, existing_comments, threshold=80):
        for existing_comment in existing_comments:
            if fuzz.ratio(comment, existing_comment) >= threshold:
                return True
        return False
    
    unique_comments = []
    for comment in filtered_comments:
        if not is_duplicate(comment, unique_comments):
            unique_comments.append(comment)
    
    return unique_comments



def detect_and_scrape_url(message):
    # Regular expression to detect URLs
    url_pattern = re.compile(r'(https?://[^\s]+)')
    
    # Search for URLs in the message
    match = url_pattern.search(message)

    # Check if a URL was found
    if match:
        url = match.group(0)
        
        # Check if the URL has already been scraped
        
        print(f"\nScraping {url}")
        website_text = get_links_and_text(url)
        # Store the scraped content
        
    
        result = {"URL": url, "text": website_text}
    else:
        result = {}

    # Convert to JSON format
    result_json = json.dumps(result)
    print(result_json)
    return result_json




def get_random_user_agent():
    ua = UserAgent()
    return ua.random

def extract_url(web_str):
    # Use regular expression to find the URL
    url_match = re.search(r'http[s]?://\S+', web_str)
    if url_match:
        return url_match.group()
    else:
        return None

def scrape_new_website(url: str, base_domain: str, max_retries: int = 3, backoff_factor: float = 0.3, timeout: int = 10) -> dict:
    headers = {'User-Agent': get_random_user_agent()}
    session = requests.Session()
    # Extract the URL
    url = extract_url(url)
    for attempt in range(max_retries):
        try:
            response = session.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text(separator=' ', strip=True)
            
            # Basic content cleaning
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = 'joined '.join(chunk for chunk in chunks if chunk)
            
            # Extract links within the same domain
            links = [urljoin(url, a.get('href')) for a in soup.find_all('a', href=True)]
            sublinks = [link for link in links if urlparse(link).netloc == base_domain]
            
            print(len(sublinks))
            return {
                "source": url,
                "content": text,
                "links": sublinks[:1]
            }
        
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                return {
                    "source": url,
                    "error": f"Failed to scrape website after {max_retries} attempts: {str(e)}"
                }
            else:
                time.sleep(backoff_factor * (2 ** attempt))
                continue

def get_links_and_text(url: str, max_depth: int = 1, max_retries: int = 3, backoff_factor: float = 0.3, timeout: int = 10):
    visited_urls = set()
    results = []

    def scrape_recursive(url: str, depth: int):
        if depth > max_depth or url in visited_urls:
            return

        visited_urls.add(url)
        base_domain = urlparse(url).netloc
        result = scrape_new_website(url, base_domain, max_retries, backoff_factor, timeout)
        
        if "error" not in result:
            results.append({"source": result["source"], "content": result["content"]})
            for link in result.get("links", []):
                scrape_recursive(link, depth + 1)

    scrape_recursive(url, 0)
    return results
