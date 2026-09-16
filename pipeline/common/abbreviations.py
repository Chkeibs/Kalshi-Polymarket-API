"""
Abbreviation Dictionary for text normalization and grouping.
Used to expand abbreviations to their full forms for better matching.
Adapted to use the new keyword dictionary technique AND preserve token-based expansion.
"""

import json
import os
import re

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_KEYWORDS_DIR = os.path.join(ROOT_DIR, "data", "keywords")

try:
    from pipeline.common.keyword_config import load_aliases
except ImportError:
    from keyword_config import load_aliases

# Path to the discovered keyword dictionary.
KEYWORD_DICT_PATH = os.path.join(DATA_KEYWORDS_DIR, "cleaned_normalized_keywords.json")

def load_keyword_dictionary(path):
    """Load the keyword dictionary to use for expansion."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

# Load the dictionary once at module level
KEYWORD_DICT = load_keyword_dictionary(KEYWORD_DICT_PATH)
# Build a reverse map for faster lookup (Variant -> Canonical)
REVERSE_DICT = {}
for canonical, info in KEYWORD_DICT.items():
    REVERSE_DICT[canonical] = canonical
    if "raw_variants" in info:
        for v in info["raw_variants"]:
            REVERSE_DICT[v] = canonical

# Base Abbreviations (Hardcoded basics that might not be in the discovered dict)
ABBREVIATIONS = {
    # Politicians - US
    "aoc": "alexandria ocasio cortez",
    "djt": "donald trump",
    "jfk": "john f kennedy",
    "rfk": "robert f kennedy",
    "maga": "make america great again trump",
    "potus": "president united states",
    "flotus": "first lady",
    "vpotus": "vice president",
    "vp": "vice president",
    "scotus": "supreme court",
    "gop": "republican party",
    "dnc": "democratic party",
    "rnc": "republican national committee",
    "dem": "democrat",
    "dems": "democrats",
    "rep": "republican",
    "reps": "republicans",
    
    # Economics & Finance
    "fed": "federal reserve",
    "fomc": "federal open market committee",
    "gdp": "gross domestic product",
    "cpi": "consumer price index",
    "ppi": "producer price index",
    "pce": "personal consumption expenditures",
    "bps": "basis points",
    "ipo": "initial public offering",
    "etf": "exchange traded fund",
    "sec": "securities exchange commission",
    "nyse": "nyse",
    "nasdaq": "nasdaq stock market",
    "sp500": "s&p 500",
    "djia": "dow jones industrial average",
    "dow": "dow jones",
    "eth": "ethereum",
    "btc": "bitcoin",
    
    # Sports Mappings (Preserved from original)
    "bun:aug": "fca", 
    "bun:b04": "lev", 
    "bun:bay": "bmu", 
    "bun:bmg": "bmg", 
    "bun:bmu": "bmu", 
    "bun:boc": "boc", 
    "bun:bvb": "bvb", 
    "bun:dor": "bvb", 
    "bun:draw": "tie", 
    "bun:ein": "sge", 
    "bun:fca": "fca", 
    "bun:fch": "fch", 
    "bun:fre": "scf", 
    "bun:hei": "fch", 
    "bun:hof": "tsg", 
    "bun:hol": "hol", 
    "bun:hsv": "hsv", 
    "bun:koe": "koe", 
    "bun:lei": "rbl", 
    "bun:lev": "lev", 
    "bun:m05": "m05", 
    "bun:mai": "m05", 
    "bun:moe": "bmg", 
    "bun:pau": "stp", 
    "bun:rbl": "rbl", 
    "bun:scf": "scf", 
    "bun:sge": "sge", 
    "bun:stp": "stp", 
    "bun:stu": "vfb", 
    "bun:svw": "svw", 
    "bun:tie": "tie", 
    "bun:tsg": "tsg", 
    "bun:uni": "uni", 
    "bun:vfb": "vfb", 
    "bun:wer": "svw", 
    "bun:wob": "wob", 
    "bun:wol": "wob", 
    
    "bundesliga:aug": "fca", 
    "bundesliga:augsburg": "fca", 
    "bundesliga:b04": "lev", 
    "bundesliga:bay": "bmu", 
    "bundesliga:bayern": "bmu", 
    "bundesliga:bayern_munich": "bmu", 
    "bundesliga:bmg": "bmg", 
    "bundesliga:bmu": "bmu", 
    "bundesliga:bremen": "svw", 
    "bundesliga:bvb": "bvb", 
    "bundesliga:cologne": "koe", 
    "bundesliga:dor": "bvb", 
    "bundesliga:dortmund": "bvb", 
    "bundesliga:draw": "tie", 
    "bundesliga:ein": "sge", 
    "bundesliga:eintracht": "sge", 
    "bundesliga:fca": "fca", 
    "bundesliga:fch": "fch", 
    "bundesliga:frankfurt": "sge", 
    "bundesliga:fre": "scf", 
    "bundesliga:freiburg": "scf", 
    "bundesliga:gladbach": "bmg", 
    "bundesliga:hamburg": "hsv", 
    "bundesliga:hei": "fch", 
    "bundesliga:heidenheim": "fch", 
    "bundesliga:hof": "tsg", 
    "bundesliga:hoffenheim": "tsg", 
    "bundesliga:hsv": "hsv", 
    "bundesliga:koe": "koe", 
    "bundesliga:koeln": "koe", 
    "bundesliga:lei": "rbl", 
    "bundesliga:leipzig": "rbl", 
    "bundesliga:lev": "lev", 
    "bundesliga:leverkusen": "lev", 
    "bundesliga:m05": "m05", 
    "bundesliga:mai": "m05", 
    "bundesliga:mainz": "m05", 
    "bundesliga:mgladbach": "bmg", 
    "bundesliga:moe": "bmg", 
    "bundesliga:pau": "stp", 
    "bundesliga:pauli": "stp", 
    "bundesliga:rbl": "rbl", 
    "bundesliga:scf": "scf", 
    "bundesliga:sge": "sge", 
    "bundesliga:st_pauli": "stp", 
    "bundesliga:stp": "stp", 
    "bundesliga:stu": "vfb", 
    "bundesliga:stuttgart": "vfb", 
    "bundesliga:svw": "svw", 
    "bundesliga:tie": "tie", 
    "bundesliga:tsg": "tsg", 
    "bundesliga:uni": "uni", 
    "bundesliga:union": "uni", 
    "bundesliga:union_berlin": "uni", 
    "bundesliga:vfb": "vfb", 
    "bundesliga:wer": "svw", 
    "bundesliga:werder": "svw", 
    "bundesliga:wob": "wob", 
    "bundesliga:wol": "wob", 
    "bundesliga:wolfsburg": "wob", 
    
    "cfb:ala": "ala", 
    "cfb:alabama": "ala", 
    "cfb:army": "army", 
    "cfb:arst": "arst", 
    "cfb:bama": "ala", 
    "cfb:boise": "bsu", 
    "cfb:bsu": "bsu", 
    "cfb:del": "del", 
    "cfb:jmu": "jmu", 
    "cfb:jsu": "jvst", 
    "cfb:jvst": "jvst", 
    "cfb:kenn": "kenn", 
    "cfb:mem": "mem", 
    "cfb:mia": "mia", 
    "cfb:miami": "mia", 
    "cfb:miss": "miss", 
    "cfb:mosu": "mosu", 
    "cfb:navy": "navy", 
    "cfb:ncst": "ncst", 
    "cfb:odu": "odu", 
    "cfb:okla": "okla", 
    "cfb:oklahoma": "okla", 
    "cfb:olemiss": "miss", 
    "cfb:ore": "ore", 
    "cfb:oregon": "ore", 
    "cfb:tamu": "txam", 
    "cfb:troy": "troy", 
    "cfb:tuln": "tuln", 
    "cfb:txam": "txam", 
    "cfb:ull": "ull", 
    "cfb:usf": "usf", 
    "cfb:usu": "usu", 
    "cfb:wash": "wash", 
    "cfb:washington": "wash", 
    "cfb:wmu": "wmu", 
    "cfb:wsu": "wsu", 
    
    "efl:bc": "bc", 
    "efl:bir": "bc", 
    "efl:birmingham": "bc", 
    "efl:bla": "bla", 
    "efl:blackburn": "bla", 
    "efl:boro": "mid", 
    "efl:brc": "brc", 
    "efl:bri": "brc", 
    "efl:bristol_city": "brc", 
    "efl:cha": "cha", 
    "efl:charlton": "cha", 
    "efl:cov": "cov", 
    "efl:coventry": "cov", 
    "efl:der": "der", 
    "efl:derby": "der", 
    "efl:draw": "tie", 
    "efl:hul": "hul", 
    "efl:hull": "hul", 
    "efl:ips": "ips", 
    "efl:ipswich": "ips", 
    "efl:lei": "lei", 
    "efl:leicester": "lei", 
    "efl:mid": "mid", 
    "efl:middlesbrough": "mid", 
    "efl:mil": "mil", 
    "efl:millwall": "mil", 
    "efl:nor": "nor", 
    "efl:norwich": "nor", 
    "efl:oxford": "oxu", 
    "efl:oxford_united": "oxu", 
    "efl:oxu": "oxu", 
    "efl:pne": "pne", 
    "efl:por": "por", 
    "efl:portsmouth": "por", 
    "efl:preston": "pne", 
    "efl:qpr": "qpr", 
    "efl:sheff_utd": "shu", 
    "efl:sheffield_united": "shu", 
    "efl:sheffield_wednesday": "shw", "efl:shu": "shu", "efl:shw": "shw", "efl:sot": "sou", "efl:sou": "sou", "efl:southampton": "sou", 
    "efl:stk": "stk", "efl:stoke": "stk", "efl:swa": "swa", "efl:swansea": "swa", "efl:swe": "shw", "efl:tie": "tie", "efl:wat": "wat", 
    "efl:watford": "wat", "efl:wba": "wba", "efl:wednesday": "shw", "efl:west_brom": "wba", "efl:west_bromwich": "wba", "efl:wre": "wre", 
    "efl:wrexham": "wre", 
    
    "epl:ars": "ars", "epl:arsenal": "ars", "epl:ast": "avl", "epl:aston_villa": "avl", "epl:avl": "avl", "epl:bou": "bou", "epl:bournemouth": "bou", 
    "epl:bre": "bre", "epl:brentford": "bre", "epl:bri": "bri", "epl:brighton": "bri", "epl:bur": "bur", "epl:burnley": "bur", "epl:cfc": "cfc", 
    "epl:che": "cfc", "epl:chelsea": "cfc", "epl:cry": "cry", "epl:crystal_palace": "cry", "epl:draw": "tie", "epl:eve": "eve", "epl:everton": "eve", 
    "epl:forest": "nfo", "epl:ful": "ful", "epl:fulham": "ful", "epl:gunners": "ars", "epl:hammers": "whu", "epl:ips": "ips", "epl:ipswich": "ips", 
    "epl:lee": "lee", "epl:leeds": "lee", "epl:leeds_united": "lee", "epl:lei": "lei", "epl:leicester": "lei", "epl:lfc": "lfc", "epl:liv": "lfc", 
    "epl:liverpool": "lfc", "epl:mac": "mci", "epl:man_city": "mci", "epl:man_utd": "mun", "epl:manchester_city": "mci", "epl:manchester_united": "mun", 
    "epl:mancity": "mci", "epl:manutd": "mun", "epl:mci": "mci", "epl:mnc": "mci", "epl:mun": "mun", "epl:new": "new", "epl:newcastle": "new", 
    "epl:nfo": "nfo", "epl:not": "nfo", "epl:nottingham": "nfo", "epl:palace": "cry", "epl:saints": "sou", "epl:sou": "sou", "epl:southampton": "sou", 
    "epl:spurs": "tot", "epl:sun": "sun", "epl:sunderland": "sun", "epl:tie": "tie", "epl:toffees": "eve", "epl:tot": "tot", "epl:tottenham": "tot", 
    "epl:villa": "avl", "epl:wes": "whu", "epl:west_ham": "whu", "epl:whu": "whu", "epl:wol": "wol", "epl:wolverhampton": "wol", "epl:wolves": "wol", 
    
    "laliga:ala": "ala", "laliga:alaves": "ala", "laliga:ath": "ath", "laliga:athletic": "ath", "laliga:atletico": "atm", "laliga:atletico_madrid": "atm", 
    "laliga:atm": "atm", "laliga:bar": "bar", "laliga:barca": "bar", "laliga:barcelona": "bar", "laliga:bet": "rbb", "laliga:betis": "rbb", 
    "laliga:bil": "ath", "laliga:bilbao": "ath", "laliga:cel": "rcc", "laliga:celta": "rcc", "laliga:celta_vigo": "rcc", "laliga:draw": "tie", 
    "laliga:elc": "elc", "laliga:elche": "elc", "laliga:esp": "esp", "laliga:espanyol": "esp", "laliga:fcb": "bar", "laliga:get": "get", 
    "laliga:getafe": "get", "laliga:gir": "gir", "laliga:girona": "gir", "laliga:lev": "lev", "laliga:levante": "lev", "laliga:mad": "atm", 
    "laliga:mal": "mal", "laliga:mallorca": "mal", "laliga:osa": "osa", "laliga:osasuna": "osa", "laliga:ovi": "ovi", "laliga:oviedo": "ovi", 
    "laliga:ray": "rvc", "laliga:rayo": "rvc", "laliga:rbb": "rbb", "laliga:rcc": "rcc", "laliga:rea": "rma", "laliga:real": "rma", 
    "laliga:real_betis": "rbb", "laliga:real_madrid": "rma", "laliga:real_sociedad": "rso", "laliga:rma": "rma", "laliga:rso": "rso", 
    "laliga:rvc": "rvc", "laliga:sev": "sev", "laliga:sevilla": "sev", "laliga:sociedad": "rso", "laliga:tie": "tie", "laliga:val": "vcf", 
    "laliga:valencia": "vcf", "laliga:vallecano": "rvc", "laliga:vcf": "vcf", "laliga:vil": "vil", "laliga:villarreal": "vil", 
    
    "mlb:ari": "ari", "mlb:atl": "atl", "mlb:bal": "bal", "mlb:bos": "bos", "mlb:chc": "chc", "mlb:cin": "cin", "mlb:cle": "cle", 
    "mlb:col": "col", "mlb:cws": "cws", "mlb:det": "det", "mlb:draw": "tie", "mlb:hou": "hou", "mlb:kc": "kc", "mlb:laa": "laa", 
    "mlb:lad": "lad", "mlb:mia": "mia", "mlb:mil": "mil", "mlb:min": "min", "mlb:nym": "nym", "mlb:nyy": "nyy", "mlb:phi": "phi", 
    "mlb:pit": "pit", "mlb:sd": "sd", "mlb:sea": "sea", "mlb:sf": "sf", "mlb:stl": "stl", "mlb:tb": "tb", "mlb:tex": "tex", 
    "mlb:tie": "tie", "mlb:tor": "tor", "mlb:was": "was", "mlb:wsh": "was", 
    
    "nba:atl": "atl", "nba:bkn": "bkn", "nba:bos": "bos", "nba:cha": "cha", "nba:chi": "chi", "nba:cle": "cle", "nba:dal": "dal", 
    "nba:den": "den", "nba:det": "det", "nba:draw": "tie", "nba:gsw": "gsw", "nba:hou": "hou", "nba:ind": "ind", "nba:lac": "lac", 
    "nba:lal": "lal", "nba:mem": "mem", "nba:mia": "mia", "nba:mil": "mil", "nba:min": "min", "nba:nop": "nop", "nba:nyk": "nyk", 
    "nba:okc": "okc", "nba:orl": "orl", "nba:phi": "phi", "nba:phx": "phx", "nba:por": "por", "nba:sac": "sac", "nba:sas": "sas", 
    "nba:tie": "tie", "nba:tor": "tor", "nba:uta": "uta", "nba:was": "was", 
    
    "nfl:49ers": "sf", 
    "nfl:ari": "ari", 
    "nfl:arizona": "ari", 
    "nfl:atl": "atl", 
    "nfl:atlanta": "atl", 
    "nfl:bal": "bal", 
    "nfl:baltimore": "bal", 
    "nfl:bears": "chi", 
    "nfl:bengals": "cin", 
    "nfl:bills": "buf", 
    "nfl:broncos": "den", 
    "nfl:browns": "cle", 
    "nfl:buccaneers": "tb", 
    "nfl:bucs": "tb", 
    "nfl:buf": "buf", 
    "nfl:buffalo": "buf", 
    "nfl:car": "car", 
    "nfl:cardinals": "ari", 
    "nfl:carolina": "car", "nfl:chargers": "lac", "nfl:chi": "chi", "nfl:chicago": "chi", "nfl:chiefs": "kc", "nfl:cin": "cin", 
    "nfl:cincinnati": "cin", "nfl:cle": "cle", "nfl:cleveland": "cle", "nfl:colts": "ind", "nfl:commanders": "was", "nfl:cowboys": "dal", 
    "nfl:dal": "dal", "nfl:dallas": "dal", "nfl:den": "den", "nfl:denver": "den", "nfl:det": "det", "nfl:detroit": "det", 
    "nfl:dolphins": "mia", "nfl:draw": "tie", "nfl:eagles": "phi", "nfl:falcons": "atl", "nfl:gb": "gb", "nfl:giants": "nyg", 
    "nfl:gnb": "gb", "nfl:green_bay": "gb", "nfl:hou": "hou", "nfl:houston": "hou", "nfl:ind": "ind", "nfl:indianapolis": "ind", 
    "nfl:jac": "jac", "nfl:jacksonville": "jac", "nfl:jaguars": "jac", "nfl:jax": "jac", "nfl:jets": "nyj", "nfl:kansas_city": "kc", 
    "nfl:kc": "kc", "nfl:kcc": "kc", "nfl:la": "la", "nfl:lac": "lac", "nfl:lar": "la", "nfl:las_vegas": "lv", "nfl:lions": "det", 
    "nfl:lv": "lv", "nfl:lvr": "lv", "nfl:mia": "mia", "nfl:miami": "mia", "nfl:min": "min", "nfl:minnesota": "min", "nfl:ne": "ne", 
    "nfl:nep": "ne", "nfl:new_england": "ne", "nfl:new_orleans": "no", "nfl:niners": "sf", "nfl:no": "no", "nfl:nor": "no", 
    "nfl:nyg": "nyg", "nfl:nyj": "nyj", "nfl:packers": "gb", "nfl:panthers": "car", "nfl:patriots": "ne", "nfl:phi": "phi", 
    "nfl:philadelphia": "phi", "nfl:pit": "pit", "nfl:pittsburgh": "pit", "nfl:raiders": "lv", "nfl:rams": "la", "nfl:ravens": "bal", 
    "nfl:saints": "no", "nfl:san_francisco": "sf", "nfl:sea": "sea", "nfl:seahawks": "sea", "nfl:seattle": "sea", "nfl:sf": "sf", 
    "nfl:sfo": "sf", "nfl:steelers": "pit", "nfl:tampa_bay": "tb", "nfl:tb": "tb", "nfl:tbb": "tb", "nfl:ten": "ten", 
    "nfl:tennessee": "ten", "nfl:texans": "hou", "nfl:tie": "tie", "nfl:titans": "ten", "nfl:vikings": "min", "nfl:was": "was", 
    "nfl:washington": "was", "nfl:wsh": "was", 
    
    "nhl:ana": "ana", 
    "nhl:ari": "ari", 
    "nhl:avs": "avs", 
    "nhl:bos": "bos", 
    "nhl:buf": "buf", 
    "nhl:cal": "cgy", 
    "nhl:car": "car", 
    "nhl:cbj": "cbj", 
    "nhl:cgy": "cgy", 
    "nhl:chi": "chi", 
    "nhl:col": "col", 
    "nhl:dal": "dal", 
    "nhl:det": "det", 
    "nhl:draw": "tie", 
    "nhl:edm": "edm", 
    "nhl:fla": "fla", 
    "nhl:la": "la", 
    "nhl:lak": "la", 
    "nhl:las": "vgk", 
    "nhl:min": "min", 
    "nhl:mon": "mtl", 
    "nhl:mtl": "mtl", 
    "nhl:nj": "nj", 
    "nhl:nsh": "nsh", 
    "nhl:nyi": "nyi", 
    "nhl:nyr": "nyr", 
    "nhl:ott": "ott", 
    "nhl:phi": "phi", 
    "nhl:pit": "pit", 
    "nhl:sea": "sea", 
    "nhl:sj": "sj", 
    "nhl:stl": "stl", 
    "nhl:tb": "tb", 
    "nhl:tie": "tie", 
    "nhl:tor": "tor", 
    "nhl:uta": "uta", 
    "nhl:utah": "uta", 
    "nhl:van": "van", 
    "nhl:vgk": "vgk", 
    "nhl:wpg": "wpg", 
    "nhl:wsh": "wsh", 
}

try:
    ABBREVIATIONS.update(load_aliases())
except Exception:
    # Config aliases are an enrichment layer. Keep hardcoded aliases usable if
    # a config JSON is temporarily malformed during local iteration.
    pass

def expand_abbreviations(text):
    """
    Expand abbreviations using the new keyword technique.
    1. Uses the Base Abbreviations map (ABBREVIATIONS).
    2. Uses the discovered dictionary (cleaned_normalized_keywords.json) for canonical mapping.
    3. Handles both full-text matches and word-by-word token matching.
    """
    if not isinstance(text, str):
        return str(text) if text else ""
    
    # Pre-cleaning
    text_lower = text.lower().strip()
    
    # Strategy 1: Whole phrase match (Fastest & Most Accurate for phrases)
    # ---------------------------------------------------------------------
    
    # 1a. Check Hardcoded Abbrevs (Exact match of full string)
    clean_text = ''.join(c for c in text_lower if c.isalnum() or c == ':' or c == '_')
    if clean_text in ABBREVIATIONS:
        return ABBREVIATIONS[clean_text]
        
    # 1b. Check Reverse Dictionary (Whole phrase variant -> Canonical)
    if REVERSE_DICT:
        # Check normalized text (collapse spaces)
        norm_text = re.sub(r'\s+', ' ', text_lower).strip()
        if norm_text in REVERSE_DICT:
             return REVERSE_DICT[norm_text]

    # Strategy 2: Word-by-Word Expansion (Recover missing functionality)
    # ---------------------------------------------------------------------
    # This was missing in the 'abbreviations copy.py' causing it to miss
    # individual words inside a sentence like "Will AOC win?"
    
    words = text_lower.split()
    expanded_words = []
    
    for word in words:
        # Clean word for lookup (remove punctuation like "AOC?" -> "aoc")
        clean_word = ''.join(c for c in word if c.isalnum() or c == ':' or c == '_').strip(':')
        
        # 2a. Check Hardcoded
        if clean_word in ABBREVIATIONS:
            expanded_words.append(ABBREVIATIONS[clean_word])
            continue
            
        # 2b. Check Reverse Dict (Single word lookup)
        if clean_word in REVERSE_DICT:
            expanded_words.append(REVERSE_DICT[clean_word])
            continue
            
        # 2c. Check Sports Prefix Logic (e.g. "nfl:ari")
        found_prefix = False
        for k, v in ABBREVIATIONS.items():
            if k.endswith(":" + clean_word):
                expanded_words.append(v)
                found_prefix = True
                break
        
        if found_prefix:
            continue
            
        # Default: Keep original word
        expanded_words.append(word)

    return ' '.join(expanded_words)
