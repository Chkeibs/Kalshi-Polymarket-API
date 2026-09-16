"""
Keyword Consolidator - Unifies all keyword sources into a master dictionary.

Sources:
1. abbreviations.py (ABBREVIATIONS dict)
2. cleaned_normalized_keywords.json
3. polymarket_topic_tags.json

Output: master_keywords.json
"""

import json
import os
import re

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
COMMON_DIR = os.path.join(ROOT_DIR, "pipeline", "common")
DATA_KEYWORDS_DIR = os.path.join(ROOT_DIR, "data", "keywords")
CONFIG_DIR = os.path.join(ROOT_DIR, "config")

# Input files
ABBREVIATIONS_FILE = os.path.join(COMMON_DIR, "abbreviations.py")
CLEANED_KEYWORDS_FILE = os.path.join(DATA_KEYWORDS_DIR, "cleaned_normalized_keywords.json")
POLYMARKET_TAGS_FILE = os.path.join(CONFIG_DIR, "polymarket_topic_tags.json")

# Output file
MASTER_KEYWORDS_FILE = os.path.join(DATA_KEYWORDS_DIR, "master_keywords.json")


def load_json_file(path):
    """Load a JSON file safely."""
    if not os.path.exists(path):
        print(f"⚠️ File not found: {path}")
        return {}
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading {path}: {e}")
        return {}


def extract_abbreviations_from_py():
    """
    Extract the ABBREVIATIONS dictionary from abbreviations.py.
    Returns a dict of {abbreviation: expansion}.
    """
    abbreviations = {}
    
    try:
        with open(ABBREVIATIONS_FILE, 'r') as f:
            content = f.read()
        
        # Find the ABBREVIATIONS dictionary using regex
        # This is a simple approach - we look for key-value pairs
        pattern = r'"([^"]+)":\s*"([^"]+)"'
        matches = re.findall(pattern, content)
        
        for abbrev, expansion in matches:
            # Skip sport team mappings (they contain colons in keys)
            if ':' in abbrev and any(prefix in abbrev for prefix in ['nfl:', 'nba:', 'mlb:', 'nhl:', 'epl:', 'bundesliga:', 'bun:', 'ligue:', 'laliga:', 'seriea:']):
                continue
            abbreviations[abbrev.lower()] = expansion.lower()
            
    except Exception as e:
        print(f"❌ Error extracting abbreviations: {e}")
    
    return abbreviations


def normalize_key(text):
    """Normalize a text to be used as a dictionary key."""
    if not text:
        return ""
    # Lowercase, strip, remove extra spaces
    normalized = text.lower().strip()
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized


def find_canonical_form(word, master_dict):
    """
    Check if a word or any of its variants already exists in the master dictionary.
    Returns (canonical_key, entry) if found, (None, None) otherwise.
    """
    word_normalized = normalize_key(word)
    
    # Direct match on canonical key
    if word_normalized in master_dict:
        return word_normalized, master_dict[word_normalized]
    
    # Check if word is in any entry's variants
    for canonical, entry in master_dict.items():
        variants = entry.get('variants', [])
        if word_normalized in [normalize_key(v) for v in variants]:
            return canonical, entry
    
    return None, None


def merge_into_master(master_dict, canonical, variants, entry_type, source):
    """
    Add or merge an entry into the master dictionary.
    If the canonical form or any variant already exists, merge variants.
    """
    canonical_normalized = normalize_key(canonical)
    variants_normalized = [normalize_key(v) for v in variants if v]
    
    # Remove duplicates and empty strings
    all_forms = set([canonical_normalized] + variants_normalized)
    all_forms.discard("")
    
    # Check if any of these forms already exist in the dictionary
    existing_canonical = None
    for form in all_forms:
        found_canonical, found_entry = find_canonical_form(form, master_dict)
        if found_canonical:
            existing_canonical = found_canonical
            break
    
    if existing_canonical:
        # Merge with existing entry
        existing_entry = master_dict[existing_canonical]
        existing_variants = set(existing_entry.get('variants', []))
        existing_variants.update(all_forms)
        existing_variants.discard(existing_canonical)  # Don't include canonical in variants
        existing_entry['variants'] = sorted(list(existing_variants))
        
        # Update sources
        existing_sources = existing_entry.get('sources', [])
        if source not in existing_sources:
            existing_sources.append(source)
        existing_entry['sources'] = existing_sources
        
    else:
        # Create new entry
        # Choose the longest form as canonical
        longest_form = max(all_forms, key=len)
        other_forms = all_forms - {longest_form}
        
        master_dict[longest_form] = {
            'variants': sorted(list(other_forms)),
            'type': entry_type,
            'sources': [source]
        }


def consolidate_keywords():
    """Main function to consolidate all keyword sources."""
    print("🔄 Starting keyword consolidation...")
    
    master_dict = {}
    
    # 1. Load existing master_keywords.json if it exists (for incremental updates)
    if os.path.exists(MASTER_KEYWORDS_FILE):
        print(f"📂 Loading existing {MASTER_KEYWORDS_FILE}...")
        master_dict = load_json_file(MASTER_KEYWORDS_FILE)
        print(f"   Loaded {len(master_dict)} existing entries.")
    
    # 2. Process abbreviations.py
    print(f"📂 Processing {ABBREVIATIONS_FILE}...")
    abbreviations = extract_abbreviations_from_py()
    abbrev_count = 0
    for abbrev, expansion in abbreviations.items():
        merge_into_master(
            master_dict,
            canonical=expansion,
            variants=[abbrev],
            entry_type='abbreviation',
            source='abbreviations.py'
        )
        abbrev_count += 1
    print(f"   Processed {abbrev_count} abbreviations.")
    
    # 3. Process cleaned_normalized_keywords.json
    print(f"📂 Processing {CLEANED_KEYWORDS_FILE}...")
    cleaned_keywords = load_json_file(CLEANED_KEYWORDS_FILE)
    cleaned_count = 0
    for canonical, info in cleaned_keywords.items():
        raw_variants = info.get('raw_variants', [])
        merge_into_master(
            master_dict,
            canonical=canonical,
            variants=raw_variants,
            entry_type='keyword',
            source='cleaned_normalized_keywords.json'
        )
        cleaned_count += 1
    print(f"   Processed {cleaned_count} keywords.")
    
    # 4. Process polymarket_topic_tags.json
    print(f"📂 Processing {POLYMARKET_TAGS_FILE}...")
    topic_tags = load_json_file(POLYMARKET_TAGS_FILE)
    tag_count = 0
    for tag, count in topic_tags.items():
        # Topic tags are already clean, add them directly
        merge_into_master(
            master_dict,
            canonical=tag,
            variants=[],
            entry_type='topic_tag',
            source='polymarket_topic_tags.json'
        )
        tag_count += 1
    print(f"   Processed {tag_count} topic tags.")
    
    # 5. Save the consolidated dictionary
    print(f"💾 Saving consolidated dictionary to {MASTER_KEYWORDS_FILE}...")
    with open(MASTER_KEYWORDS_FILE, 'w') as f:
        json.dump(master_dict, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Consolidation complete! Total entries: {len(master_dict)}")
    
    # Print some stats
    types_count = {}
    for entry in master_dict.values():
        t = entry.get('type', 'unknown')
        types_count[t] = types_count.get(t, 0) + 1
    
    print("\n📊 Entries by type:")
    for t, c in sorted(types_count.items()):
        print(f"   {t}: {c}")
    
    return master_dict


def main():
    consolidate_keywords()


if __name__ == "__main__":
    main()
