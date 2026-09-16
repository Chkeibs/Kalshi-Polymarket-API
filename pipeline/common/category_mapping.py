"""
Category mapping dictionary and utility script to check category distribution.
Run this script directly to see category counts from the generated CSVs.
"""

import pandas as pd
import os

CATEGORY_MAPPING = {
    # Politics
    "politics": "Politics",
    "elections": "Politics",
    "world elections": "Politics",
    "global elections": "Politics",
    "midterms": "Politics",
    "us election": "Politics",
    "political": "Politics",
    "trump": "Politics",
    "biden": "Politics",
    "white house": "Politics",
    "congress": "Politics",
    "senate": "Politics",
    "house": "Politics",
    "government": "Politics",
    "policy": "Politics",
    "uk politics": "Politics",
    "geopolitics": "Politics",
    "bernie sanders": "Politics",
    "mayor": "Politics",
    "supreme court": "Politics",
    "courts": "Politics",
    "israel": "Politics",
    "ukraine": "Politics",
    "gaza": "Politics",
    "brazil": "Politics",
    
    # Economics / Finance
    "economics": "Economics",
    "economy": "Economics",
    "financials": "Economics",
    "companies": "Economics",
    "business": "Economics",
    "finance": "Economics",
    "ipo": "Economics",
    "fed rates": "Economics",
    "fed": "Economics",
    "inflation": "Economics",
    "recession": "Economics",
    "stocks": "Economics",
    "interest rates": "Economics",
    "economic policy": "Economics",
    "deficit": "Economics",
    "jerome powell": "Economics",
    "macro indicators": "Economics",
    "trade war": "Economics",
    "exchange": "Economics",
    "pre-market": "Economics",
    
    # Crypto
    "crypto": "Crypto",
    "cryptocurrency": "Crypto",
    "bitcoin": "Crypto",
    "ethereum": "Crypto",
    "doge": "Crypto",
    "nft": "Crypto",
    "airdrops": "Crypto",
    "consensys": "Crypto",
    
    # Sports
    "sports": "Sports",
    "nba": "Sports",
    "nfl": "Sports",
    "nhl": "Sports",
    "mlb": "Sports",
    "soccer": "Sports",
    "football": "Sports",
    "basketball": "Sports",
    "hockey": "Sports",
    "tennis": "Sports",
    "golf": "Sports",
    "ufc": "Sports",
    "boxing": "Sports",
    "formula 1": "Sports",
    "ncaab": "Sports",
    "ncaa": "Sports",
    "big 12": "Sports",
    "super bowl lx": "Sports",
    "serie a": "Sports",
    "la liga": "Sports",
    "premier league": "Sports",
    "champions league": "Sports",
    "cwbb": "Sports",
    "efl cup": "Sports",
    "athletes": "Sports",
    
    # Science / Tech
    "science": "Science",
    "technology": "Science",
    "science and technology": "Science",
    "tech": "Science",
    "ai": "Science",
    "artificial intelligence": "Science",
    "space": "Science",
    "spacex": "Science",
    "climate": "Science",
    "climate and weather": "Science",
    "weather": "Science",
    "internet": "Science",
    "big tech": "Science",
    "tiktok": "Science", 
    "openai": "Science",
    "ai technology": "Science",
    
    # Culture / Entertainment
    "culture": "Culture",
    "entertainment": "Culture",
    "pop culture": "Culture",
    "movies": "Culture",
    "cinema": "Culture",
    "music": "Culture",
    "music & arts": "Culture",
    "celebrity": "Culture",
    "awards": "Culture",
    "oscars": "Culture",
    "grammy": "Culture",
    "emmy": "Culture",
    "golden globes": "Culture",
    "events": "Culture",
    "box office": "Culture",
    "games": "Culture",
    "video games": "Culture",
    "chess": "Culture",
    "taylor swift": "Culture",

    # World
    "world": "World",
    "global": "World",
    "international": "World",
    
    # Other
    "uncategorized": "Other",
    "hide from new": "Other",
    "other": "Other",
    
    # Stragglers
    "trump presidency": "Politics",
    "trump-putin": "Politics",
    "foreign policy": "Politics",
    "starmer": "Politics",
    "u.s. politics": "Politics",
    "france": "Politics",
    "iran": "Politics",
    "putin": "Politics",
    "mamdani": "Politics",
    "argentina": "Politics",
    "south korea": "Politics",
    "dutch": "Politics",
    "defense": "Politics",
    "hamas": "Politics",
    "middle east": "Status", 
    "india": "Politics",
    "pierre": "Politics",
    "epstein": "Politics",
}

def print_cats(file):
    try:
        if not os.path.exists(file):
            print(f"File not found: {file}")
            return
            
        df = pd.read_csv(file)
        if "Category" in df.columns:
            print(f"--- {file} ---")
            print(df["Category"].value_counts().head(50))
    except Exception as e:
        print(f"Error reading {file}: {e}")

if __name__ == "__main__":
    print_cats("markets_clean_kalshi.csv")
    print_cats("markets_clean_polymarket.csv")
