import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from tqdm import tqdm

books = []

for page in tqdm(range(1, 670)):  # 100 pages × 100 books ≈ 10,000
    url = f"https://www.goodreads.com/list/show/1.Best_Books_Ever?page={page}"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "lxml")
    
    for book in soup.select("tr"):
        title_tag = book.select_one("a.bookTitle span")
        author_tag = book.select_one("a.authorName span")
        rating_tag = book.select_one("span.minirating")
        
        if title_tag and author_tag and rating_tag:
            books.append({
                "title": title_tag.text.strip(),
                "author": author_tag.text.strip(),
                "rating_info": rating_tag.text.strip()
            })
    
    time.sleep(2)  # Avoid blocking

df = pd.DataFrame(books)
df.to_csv("goodreads_books.csv", index=False)
print(f"Scraped {len(df)} books")
