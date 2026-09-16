"""
Entity Extractor - Extracts named entities from event titles using SpaCy,
normalizes them, and enriches the master keywords dictionary.

This script can run in two modes:
- Full mode: Process all events (initial setup)
- Incremental mode: Process only new events (SyncStatus=New)
"""

import pandas as pd
import spacy
import json
import os
import re
import argparse
from collections import defaultdict

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_MARKETS_DIR = os.path.join(ROOT_DIR, "data", "markets")
DATA_KEYWORDS_DIR = os.path.join(ROOT_DIR, "data", "keywords")

# Config
KALSHI_FILE = os.path.join(DATA_MARKETS_DIR, "markets_clean_kalshi.csv")
POLY_FILE = os.path.join(DATA_MARKETS_DIR, "markets_clean_polymarket.csv")
MASTER_KEYWORDS_FILE = os.path.join(DATA_KEYWORDS_DIR, "master_keywords.json")

# SpaCy model
nlp = None


def load_spacy():
    """Load SpaCy model."""
    global nlp
    if nlp is None:
        print("Loading SpaCy model...")
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading SpaCy model...")
            import sys
            os.system(f'"{sys.executable}" -m spacy download en_core_web_sm')
            nlp = spacy.load("en_core_web_sm")
    return nlp


def load_master_keywords():
    """Load the master keywords dictionary."""
    if os.path.exists(MASTER_KEYWORDS_FILE):
        with open(MASTER_KEYWORDS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_master_keywords(master_dict):
    """Save the master keywords dictionary."""
    with open(MASTER_KEYWORDS_FILE, 'w') as f:
        json.dump(master_dict, f, indent=2, ensure_ascii=False)


def normalize_key(text):
    """Normalize text for dictionary keys."""
    if not text:
        return ""
    normalized = text.lower().strip()
    normalized = re.sub(r"['']s$", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized

# Ambiguous single words that should NEVER be added as standalone variants
# These words can mean different things in different contexts (person vs company vs location etc)
AMBIGUOUS_SINGLE_WORDS = {
    'mars', 'apple', 'amazon', 'meta', 'oracle', 'dell', 'ford', 'target',
    'chase', 'wells', 'delta', 'united', 'american', 'national', 'international',
    'general', 'morgan', 'goldman', 'black', 'white', 'brown', 'green', 'blue',
    'red', 'king', 'prince', 'queen', 'jack', 'bush', 'gore', 'stone', 'young',
    'long', 'short', 'best', 'good', 'new', 'old', 'big', 'small', 'first', 'last',
    'high', 'low', 'run', 'bruno', 'jordan', 'taylor', 'swift', 'fox', 'wolf',
    'bear', 'bull', 'eagle', 'lion', 'tiger', 'phoenix', 'sun', 'moon', 'star',
    'gold', 'silver', 'diamond', 'crystal', 'pearl', 'ruby', 'jade', 'rose',
    'grace', 'hope', 'faith', 'joy', 'love', 'will', 'may', 'june', 'august',
    'christian', 'hunter', 'fisher', 'baker', 'cook', 'smith', 'miller', 'walker'
}


def find_existing_entry(word, master_dict):
    """
    Check if a word or any variant exists in master dictionary.
    Returns (canonical_key, entry) if found, (None, None) otherwise.
    """
    word_norm = normalize_key(word)
    
    # Direct match
    if word_norm in master_dict:
        return word_norm, master_dict[word_norm]
    
    # Check variants
    for canonical, entry in master_dict.items():
        variants = [normalize_key(v) for v in entry.get('variants', [])]
        if word_norm in variants:
            return canonical, entry
        # Also check if word_norm contains or is contained in canonical/variants
        if word_norm in canonical or canonical in word_norm:
            return canonical, entry
    
    return None, None


def extract_name_variants(full_name):
    """
    Generate possible variants of a name.
    E.g., "Donald Trump" -> ["donald trump", "trump", "donald", "d trump", "donald t"]
    
    IMPORTANT: Single-word variants that are in AMBIGUOUS_SINGLE_WORDS are excluded
    to prevent incorrect matches (e.g., "mars" for "Bruno Mars" planet confusion)
    """
    parts = full_name.lower().strip().split()
    variants = set()
    
    if len(parts) == 0:
        return variants
    
    # Full name - always add
    variants.add(" ".join(parts))
    
    if len(parts) == 1:
        # Single word name - only add if not ambiguous
        if parts[0] not in AMBIGUOUS_SINGLE_WORDS:
            variants.add(parts[0])
        return variants
    
    # Last name only (most common reference) - but not if ambiguous
    if parts[-1] not in AMBIGUOUS_SINGLE_WORDS:
        variants.add(parts[-1])
    
    # First name only - but not if ambiguous
    if parts[0] not in AMBIGUOUS_SINGLE_WORDS:
        variants.add(parts[0])
    
    # First + Last (if more than 2 parts)
    if len(parts) > 2:
        variants.add(f"{parts[0]} {parts[-1]}")
    
    # Initial + Last name (always safe as it's multi-word)
    variants.add(f"{parts[0][0]} {parts[-1]}")
    
    # First + Initial (for middle names like "Donald J Trump")
    if len(parts) >= 3:
        # Full without middle
        variants.add(f"{parts[0]} {parts[-1]}")
    
    return variants


def merge_entity_into_master(master_dict, entity_text, entity_label):
    """
    Add or merge an entity into the master dictionary.
    Uses the longest/most complete form as canonical.
    """
    entity_norm = normalize_key(entity_text)
    if len(entity_norm) < 2:
        return False
    
    # Generate variants
    if entity_label == "PERSON":
        variants = extract_name_variants(entity_text)
    else:
        variants = {entity_norm}
    
    # Check if any variant already exists
    existing_canonical = None
    for variant in variants:
        found_canonical, _ = find_existing_entry(variant, master_dict)
        if found_canonical:
            existing_canonical = found_canonical
            break
    
    if existing_canonical:
        # Merge: add new variants to existing entry
        existing_entry = master_dict[existing_canonical]
        existing_variants = set(existing_entry.get('variants', []))
        existing_variants.update(variants)
        existing_variants.discard(existing_canonical)
        existing_entry['variants'] = sorted(list(existing_variants))
        
        # Maybe upgrade canonical if new form is longer
        if len(entity_norm) > len(existing_canonical):
            # Create new entry with longer name as canonical
            new_variants = existing_variants | {existing_canonical}
            new_variants.discard(entity_norm)
            master_dict[entity_norm] = {
                'variants': sorted(list(new_variants)),
                'type': entity_label.lower(),
                'sources': existing_entry.get('sources', []) + ['entity_extractor']
            }
            del master_dict[existing_canonical]
        
        return False  # Not a new entry
    else:
        # Create new entry
        # Use longest variant as canonical
        longest = max(variants, key=len) if variants else entity_norm
        other_variants = variants - {longest}
        
        master_dict[longest] = {
            'variants': sorted(list(other_variants)),
            'type': entity_label.lower(),
            'sources': ['entity_extractor']
        }
        return True  # New entry added


def extract_entities_from_events(df, master_dict, nlp_model):
    """
    Extract named entities from event titles and add to master dictionary.
    Returns stats about extraction.
    """
    stats = {
        'events_processed': 0,
        'entities_found': 0,
        'new_entities_added': 0,
        'entities_merged': 0
    }
    
    # Entity types we care about
    TARGET_LABELS = {'PERSON', 'ORG', 'GPE', 'NORP', 'EVENT', 'LOC'}
    
    titles = df['Market Title'].dropna().unique()
    
    print(f"Processing {len(titles)} unique event titles...")
    
    for i, title in enumerate(titles):
        if i % 500 == 0:
            print(f"  Progress: {i}/{len(titles)}")
        
        doc = nlp_model(title)
        stats['events_processed'] += 1
        
        for ent in doc.ents:
            if ent.label_ not in TARGET_LABELS:
                continue
            
            stats['entities_found'] += 1
            
            is_new = merge_entity_into_master(master_dict, ent.text, ent.label_)
            
            if is_new:
                stats['new_entities_added'] += 1
            else:
                stats['entities_merged'] += 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Extract entities from event titles')
    parser.add_argument('--full', action='store_true', 
                       help='Process all events (default: only new events)')
    args = parser.parse_args()
    
    print("🔄 Starting Entity Extraction...")
    
    # Load SpaCy
    nlp_model = load_spacy()
    
    # Load master keywords
    master_dict = load_master_keywords()
    initial_count = len(master_dict)
    print(f"📂 Loaded {initial_count} entries from master keywords.")
    
    # Load event data
    print("📂 Loading event data...")
    df_kalshi = pd.read_csv(KALSHI_FILE, dtype=str)
    df_poly = pd.read_csv(POLY_FILE, dtype=str)
    
    # Filter to new events only if not full mode
    if not args.full:
        if 'SyncStatus' in df_kalshi.columns:
            df_kalshi = df_kalshi[df_kalshi['SyncStatus'] == 'New']
        if 'SyncStatus' in df_poly.columns:
            df_poly = df_poly[df_poly['SyncStatus'] == 'New']
        print(f"🔍 Incremental mode: {len(df_kalshi)} new Kalshi, {len(df_poly)} new Polymarket events")
    else:
        print(f"🔍 Full mode: {len(df_kalshi)} Kalshi, {len(df_poly)} Polymarket events")
    
    # Extract entities from Kalshi events
    print("\n📊 Processing Kalshi events...")
    stats_k = extract_entities_from_events(df_kalshi, master_dict, nlp_model)
    
    # Extract entities from Polymarket events
    print("\n📊 Processing Polymarket events...")
    stats_p = extract_entities_from_events(df_poly, master_dict, nlp_model)
    
    # Save updated master keywords
    save_master_keywords(master_dict)
    
    # Print summary
    final_count = len(master_dict)
    print("\n" + "="*50)
    print("✅ Entity Extraction Complete!")
    print("="*50)
    print(f"\n📊 Kalshi Stats:")
    print(f"   Events processed: {stats_k['events_processed']}")
    print(f"   Entities found: {stats_k['entities_found']}")
    print(f"   New entities added: {stats_k['new_entities_added']}")
    
    print(f"\n📊 Polymarket Stats:")
    print(f"   Events processed: {stats_p['events_processed']}")
    print(f"   Entities found: {stats_p['entities_found']}")
    print(f"   New entities added: {stats_p['new_entities_added']}")
    
    print(f"\n📊 Master Keywords:")
    print(f"   Initial entries: {initial_count}")
    print(f"   Final entries: {final_count}")
    print(f"   Net new entries: {final_count - initial_count}")


if __name__ == "__main__":
    main()
