import pandas as pd
import json
import os
from collections import Counter

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_MARKETS_DIR = os.path.join(ROOT_DIR, "data", "markets")
CONFIG_DIR = os.path.join(ROOT_DIR, "config")

# List of format tags to be excluded from topic tag analysis.
# This list is based on user input and expanded for better filtering.
FORMAT_TAGS = {
    'daily', 'weekly', 'monthly', 'yearly', 'hourly', 'today', 'open', 
    'recurring', 'up or down', 'hit price', 'range', 'parlays', 
    'prediction markets', 'rewards', 'multi strikes', 'parent for derivative', 
    'derivatives', 'hide from new', 'list', 'status', 'base', 'other', 
    'mentions', 'charts', 'balance', 'fees', 'rewards 200, 3.5, 50', 
    'rewards 500, 4.5, 50', 'daily temperature', 'eventuals', 'nea', 
    'pmq', 'ppa', 'tsa', 'fdv', 'new', 'live', 'hot', 'ending soon',
    'politics', 'crypto', 'sports', 'finance', 'science', 'world', 
    'entertainment', 'health', 'technology', 'culture', 'weather',
    'us politics', 'us', 'global'
}

def clean_tag(tag):
    """Cleans and standardizes a single tag."""
    return tag.strip().lower()

def is_format_tag(tag, format_tags):
    """Check if a tag is a format tag."""
    return clean_tag(tag) in format_tags

def analyze_polymarket_subcategories(input_file, output_file):
    """
    Analyzes Polymarket sub-categories to extract and count topic tags.
    
    It reads the market data, processes the 'Sub-Category' column, filters out
    pre-defined format tags, and saves the counts of the remaining topic tags
    to a JSON file.
    """
    print("Starting analysis of Polymarket sub-categories...")
    try:
        df = pd.read_csv(input_file, dtype={'Sub-Category': str})
    except FileNotFoundError:
        print(f"❌ Error: Input file not found at '{input_file}'")
        return
    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")
        return

    # Ensure 'Sub-Category' column exists
    if 'Sub-Category' not in df.columns:
        print("❌ Error: 'Sub-Category' column not found in the input file.")
        return

    # Fill NaN values with empty string to avoid errors
    df['Sub-Category'] = df['Sub-Category'].fillna('')

    topic_tags_counter = Counter()

    # Process each row's sub-categories
    for sub_categories_str in df['Sub-Category']:
        if not sub_categories_str:
            continue
        
        # Split sub-categories string into individual tags
        tags = sub_categories_str.split(',')
        
        # Filter out format tags to keep only topic tags
        topic_tags = [
            clean_tag(tag) for tag in tags if not is_format_tag(tag, FORMAT_TAGS)
        ]
        
        # Update the counter with the topic tags found in the current row
        topic_tags_counter.update(topic_tags)

    # Convert counter to a dictionary for JSON serialization
    # Sorting by frequency for better readability
    sorted_topic_tags = dict(topic_tags_counter.most_common())

    # Save the result to a JSON file
    try:
        with open(output_file, 'w') as f:
            json.dump(sorted_topic_tags, f, indent=4)
        print(f"✅ Successfully analyzed and saved topic tags to '{output_file}'")
    except Exception as e:
        print(f"❌ Error writing to JSON file: {e}")

def main():
    """Main function to run the analysis."""
    input_csv = os.path.join(DATA_MARKETS_DIR, 'markets_clean_polymarket.csv')
    output_json = os.path.join(CONFIG_DIR, 'polymarket_topic_tags.json')
    analyze_polymarket_subcategories(input_csv, output_json)

if __name__ == "__main__":
    main()
