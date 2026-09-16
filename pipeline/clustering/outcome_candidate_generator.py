#!/usr/bin/env python3
"""
Generate outcome-aware candidate pairs before LLM verification.

This stage is additive: it does not replace event-level candidate_clusters.json.
It catches cases where one side has a broad multi-outcome event and the other
side exposes the specific tradable outcome in the bet/contract title.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from typing import Iterable

import pandas as pd


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
COMMON_DIR = os.path.join(ROOT_DIR, "pipeline", "common")
CONFIG_DIR = os.path.join(ROOT_DIR, "config")
DATA_MARKETS_DIR = os.path.join(ROOT_DIR, "data", "markets")
DATA_CLUSTERS_DIR = os.path.join(ROOT_DIR, "data", "clusters")

if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)

from category_utils import (  # noqa: E402
    categories_overlap,
    kalshi_category_key,
    mapped_polymarket_keys_for_kalshi,
    normalize_category_mapping,
    normalize_label,
    polymarket_category_keys,
)
try:  # noqa: E402
    from pipeline.clustering.group_events_cluster import (
        has_location_conflict as event_location_conflict,
        has_scope_granularity_conflict as event_scope_granularity_conflict,
        has_sports_domain_conflict as event_sports_domain_conflict,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from group_events_cluster import (
        has_location_conflict as event_location_conflict,
        has_scope_granularity_conflict as event_scope_granularity_conflict,
        has_sports_domain_conflict as event_sports_domain_conflict,
    )

KALSHI_CSV = os.path.join(DATA_MARKETS_DIR, "markets_clean_kalshi.csv")
POLYMARKET_CSV = os.path.join(DATA_MARKETS_DIR, "markets_clean_polymarket.csv")
OUTPUT_JSON = os.path.join(DATA_CLUSTERS_DIR, "candidate_outcome_pairs.json")
OUTCOME_BUNDLES_JSON = os.path.join(DATA_CLUSTERS_DIR, "candidate_outcome_bundles.json")
CATEGORY_MAPPING_FILE = os.path.join(CONFIG_DIR, "category_mapping.json")

DEFAULT_MAX_CANDIDATES_PER_KALSHI_BET = 5

KALSHI_USECOLS = (
    "Event ID",
    "Bet ID",
    "Market Title",
    "Bet Title",
    "Category",
    "Sub-Category",
    "Description",
    "Bet_Rules",
    "Event_Description",
)

POLYMARKET_USECOLS = (
    "Event ID",
    "Bet ID",
    "Original_Market_ID",
    "Market Title",
    "Bet Title",
    "Outcome_Name",
    "Category",
    "Sub-Category",
    "Description",
    "Bet_Rules",
    "Event_Description",
)

GENERIC_OUTCOME_SUBJECTS = {
    "",
    "yes",
    "no",
    "other",
    "others",
    "none",
    "n/a",
    "na",
    "unknown",
    "above",
    "below",
    "before",
    "after",
    "over",
    "under",
    "any",
    "someone",
    "anyone",
    "a human",
    "human",
}

STOP_WORDS = {
    "a",
    "an",
    "and",
    "any",
    "are",
    "at",
    "be",
    "by",
    "does",
    "for",
    "from",
    "if",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "will",
    "who",
    "which",
    "what",
    "when",
    "where",
    "with",
}

GENERIC_MARKET_TOKENS = {
    "after", "announce", "announced", "award", "before", "best", "champion",
    "companies", "company", "countries", "country", "during", "election", "event",
    "events", "how", "leader", "many", "market", "most", "new", "next", "nominee",
    "officially", "party", "primary", "race", "regular", "season", "state", "states",
    "title", "top", "trophy", "what", "which", "who", "win", "winner", "winning",
    "wins", "year", "years",
}

SCOPE_WORDS = {
    "champion",
    "championship",
    "cup",
    "division",
    "election",
    "final",
    "governor",
    "group",
    "house",
    "mayor",
    "nominee",
    "playoff",
    "president",
    "presidential",
    "primary",
    "qualifier",
    "qualify",
    "round",
    "semifinal",
    "senate",
    "senator",
    "seat",
    "seats",
    "tournament",
    "winner",
    "world",
}

SUBJECT_ALIASES = {
    "u s": "united states",
    "u s a": "united states",
    "usa": "united states",
    "us": "united states",
    "united states of america": "united states",
    "korea republic": "south korea",
    "republic of korea": "south korea",
    "czech republic": "czechia",
    "cote d ivoire": "ivory coast",
    "cote divoire": "ivory coast",
    "dr congo": "democratic republic of congo",
    "drc": "democratic republic of congo",
    "uae": "united arab emirates",
    "uk": "united kingdom",
    "england": "england",
    "great britain": "england",
}

TOKEN_ALIASES = {
    "ipos": "ipo",
}

MONTH_NAMES = (
    "january|february|march|april|may|june|july|august|"
    "september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)

INTENT_DIMENSION_PATTERNS = {
    "arrest": (r"\barrest(?:ed|s|ing)?\b",),
    "presidential_run": (
        r"\bannounce\w*\s+(?:a\s+)?presidential\s+(?:run|campaign)\b",
        r"\brun\s+for\s+president\b",
        r"\brun\s+for\b.{0,45}\bpresidential\s+nomination\b",
    ),
    "office_exit": (
        r"\bleave\w*\b.{0,40}\b(?:administration|office|cabinet)\b",
        r"\bresign\w*\b",
        r"\bout\s+as\b",
    ),
    "office_entry": (
        r"\bjoin\w*\b.{0,40}\b(?:administration|cabinet)\b",
        r"\bappoint\w*\b.{0,40}\b(?:administration|cabinet)\b",
    ),
    "impeachment": (r"\bimpeach\w*\b",),
    "retirement": (r"\bretir\w*\b",),
    "album_release": (r"\b(?:release|drop)\w*\b.*\balbums?\b", r"\balbums?\b.*\b(?:release|drop)\w*\b"),
    "song_release": (r"\b(?:release|drop)\w*\b.*\b(?:songs?|tracks?)\b", r"\b(?:songs?|tracks?)\b.*\b(?:release|drop)\w*\b"),
    "album_market": (r"\balbums?\b",),
    "song_market": (r"\b(?:songs?|tracks?)\b",),
    "album_feature": (r"\bfeatur\w*\b.{0,50}\balbums?\b",),
    "album_chart": (
        r"(?:#\s*\d+|\btop\b|\bnumber one\b|\bmost\b).{0,35}\balbums?\b",
        r"\balbums?\b.{0,35}(?:#\s*\d+|\btop\b|\bnumber one\b|\bmost\b)",
    ),
    "song_chart": (
        r"(?:#\s*\d+|\btop\b|\bnumber one\b|\bmost\b).{0,35}\b(?:songs?|tracks?)\b",
        r"\b(?:songs?|tracks?)\b.{0,35}(?:#\s*\d+|\btop\b|\bnumber one\b|\bmost\b)",
    ),
    "artist_ranking": (
        r"(?:#\s*\d+|\btop\b|\bnumber one\b|\bmost\b).{0,50}\bartists?\b",
        r"\bartists?\b.{0,50}(?:#\s*\d+|\btop\b|\bnumber one\b|\bmost\b)",
    ),
    "token_launch": (r"\blaunch\w*\b.*\btokens?\b", r"\btokens?\b.*\blaunch\w*\b"),
    "government_stake": (
        r"\b(?:government|us|u s)\b.{0,40}\b(?:take|buy|acquire)\w*\b.{0,30}\bstake\b",
        r"\btake\w*\b.{0,30}\bcontrol\b",
    ),
    "ipo": (r"\bipos?\b", r"\binitial public offerings?\b"),
    "bankruptcy": (r"\bbankrupt\w*\b",),
    "acquisition": (r"\bacquir\w*\b", r"\bbuyout\b",),
    "scripted_series": (r"\bscripted\s+series\b", r"\bmulti[ -]?episode\b",),
    "device_product": (r"\bproduct\s+line\b", r"\b(?:iphone|smartphone|phone|device)\b",),
    "country_visit": (r"\b(?:countries|country)\b.{0,50}\bvisit\w*\b", r"\bvisit\w*\b.{0,50}\b(?:countries|country)\b"),
    "person_visit_location": (r"\bvisit\w*\b",),
    "diplomatic_meeting": (r"\bmeet\w*\b",),
    "conversation_with": (r"\btalk\w*\s+to\b", r"\bspeak\w*\s+with\b"),
    "speech_mention": (r"\b(?:say|said)\b",),
    "legislative_vote": (r"\bvote\w*\b.{0,60}\b(?:act|bill|appropriations?|resolution)\b",),
    "pardon": (r"\bpardon\w*\b",),
    "trade_deal": (r"\btrade\s+deals?\b",),
    "recognition": (r"\brecogniz\w*\b",),
    "relations_normalization": (r"\bnormalize\w*\s+relations\b", r"\babraham accords?\b"),
    "competition_participation": (r"\bparticipat\w*\b",),
    "player_transfer": (r"\b(?:players?|athletes?)\b.{0,45}\btransfer\w*\b", r"\btransfer\w*\b.{0,45}\b(?:clubs?|teams?)\b"),
    "presidential_election": (r"\bpresidential\s+election\b",),
    "presidential_nomination": (r"\b(?:presidential|president)\b.{0,35}\b(?:nominee|nomination)\b", r"\b(?:nominee|nomination)\b.{0,35}\b(?:presidential|president)\b"),
    "medal_award": (r"\bmedal\s+of\s+freedom\b",),
    "tour_announcement": (r"\b(?:announce\w*\s+)?(?:a\s+)?new\s+tour\b",),
    "national_chamber_control": (
        r"\bparty\b.{0,30}\b(?:win|control)\w*\b.{0,30}\b(?:house|senate)\b",
        r"\bparty\b.{0,30}\b(?:house|senate)\b",
    ),
    "individual_legislative_race": (
        r"\b(?:senate|house)\s+(?:race|winner|election)\b",
        r"\b[a-z]{2}[ -]?\d{1,2}\b.{0,30}\b(?:house|election|primary)\b",
    ),
    "unemployment_metric": (r"\bunemployment\b",),
    "inflation_metric": (r"\binflation\b", r"\bcpi\b"),
    "gdp_metric": (r"\bgdp\b", r"\bgross domestic product\b"),
    "bond_role_q": (r"\bplay\s+q\b.{0,30}\bjames\s+bond\b",),
    "bond_role_bond": (r"\b(?:next|play)\b.{0,30}\bjames\s+bond\b",),
    "competition_elimination": (r"\beliminat\w*\b",),
    "competition_finalist": (r"\b(?:make|reach|advance\s+to)\s+(?:the\s+)?finals?\b",),
    "competition_knockout_advance": (r"\badvance\w*\s+to\s+(?:the\s+)?knockout\s+stages?\b",),
    "competition_seed": (r"\b#?\s*\d+\s+seed\b", r"\bseed\s+in\b.{0,30}\btournament\b"),
    "competition_play_in": (r"\bplay[ -]?in\s+tournament\b",),
    "playoffs_qualification": (r"\b(?:make|reach|qualify\w*\s+for)\b.{0,30}\bplayoffs?\b",),
    "round_of_16": (r"\bround\s+of\s+16\b",),
    "quarterfinal": (r"\bquarter[ -]?final\w*\b",),
    "stat_era": (r"\bera\b",),
    "stat_doubles": (r"\bdoubles\b",),
    "stat_war": (r"\bwins?\s+above\s+replacement\b", r"\bwar\b"),
    "stat_rbi": (r"\brbis?\b",),
    "stat_ops": (r"\bops\b",),
    "stat_hits": (r"\bhits?\s+(?:leader|total)\b", r"\bmost\s+hits?\b"),
    "stat_batting_average": (r"\bbatting\s+average\b",),
    "stat_triples": (r"\btriples?\s+leader\b", r"\bmost\s+triples?\b"),
    "stat_home_runs": (r"\bhome\s+runs?\b", r"\bhomers?\b"),
    "stat_stolen_bases": (r"\bstolen\s+bases?\b", r"\bsteals?\s+leader\b"),
    "stat_runs": (r"\bruns?\s+leader\b", r"\bmost\s+runs?\b"),
    "stat_pitching_wins": (r"\bpitching\s+wins?\b", r"\bwins?\s+leader\b"),
    "stat_points": (r"\bpoints?\s+per\s+game\b",),
    "stat_rebounds": (r"\brebounds?\s+per\s+game\b",),
    "stat_assists": (r"\bassists?\s+per\s+game\b",),
    "stat_steals": (r"\bsteals?\s+per\s+game\b",),
    "stat_blocks": (r"\bblocks?\s+per\s+game\b",),
    "stat_three_pointers": (r"\b(?:three|3)[ -]?pointers?\b", r"\b3pm\b"),
    "award_cy_young": (r"\bcy\s+young\b",),
    "award_hank_aaron": (r"\bhank\s+aaron\b",),
    "award_mvp": (r"\bmvp\b", r"\bmost\s+valuable\s+player\b"),
    "award_hart": (r"\bhart\s+memorial\b",),
    "award_art_ross": (r"\bart\s+ross\b",),
    "award_rocket_richard": (r"\brocket.?\s+richard\b",),
    "award_calder": (r"\bcalder\s+memorial\b",),
    "award_vezina": (r"\bvezina\b",),
    "award_jack_adams": (r"\bjack\s+adams\b",),
    "award_clutch": (r"\bclutch\s+player\b",),
    "award_defensive_player": (r"\bdefensive\s+player\s+of\s+the\s+year\b",),
    "award_most_improved": (r"\bmost\s+improved\s+player\b",),
    "award_james_norris": (r"\b(?:james\s+)?norris\s+(?:memorial\s+)?trophy\b",),
    "award_conference_finals_mvp": (r"\bconference\s+finals?\s+mvp\b",),
    "award_finals_mvp": (r"\bfinals?\s+mvp\b",),
    "award_comeback_player": (r"\bcomeback\s+player\s+of\s+the\s+year\b",),
    "award_presidents_trophy": (r"\bpresident(?:s|\s+s)?\s+trophy\b",),
    "award_selke": (r"\b(?:frank\s+j\s+)?selke\s+trophy\b",),
    "award_rookie_of_year": (r"\brookie\s+of\s+the\s+year\b",),
    "award_player_of_year": (r"\bplayers?\s+player\s+of\s+the\s+year\b", r"\bplayer\s+of\s+the\s+year\b"),
    "ufc_pound_for_pound": (r"\bpound[ -]?for[ -]?pound\b",),
    "ufc_weight_title": (r"\bufc\b.{0,40}\b(?:flyweight|bantamweight|featherweight|lightweight|welterweight|middleweight|heavyweight|weight)\b.{0,30}\b(?:title|champion)\b",),
    "ufc_any_title": (r"\bufc\b.{0,40}\b(?:title|champion)\b",),
    "division_title": (r"\bdivision\s+(?:winner|champion)\b",),
    "conference_title": (r"\bconference\s+(?:winner|champion)\b",),
    "stanley_cup_title": (r"\bstanley\s+cup\b",),
    "regular_season_title": (r"\bregular\s+season\s+(?:winner|champion)\b",),
    "conference_tournament_title": (r"\b(?:conference|league)\b.{0,45}\btournament\s+(?:winner|champion)\b",),
    "national_title": (
        r"\b(?:mens|womens)?\s*college\s+basketball\s+(?:winner|champion)\b",
        r"\bncaa\s+tournament\s+(?:winner|champion)\b",
    ),
    "majority_loss": (r"\blose\w*\b.{0,30}\b(?:house|senate)\s+majority\b",),
    "house_seat_count": (r"\bhow\s+many\s+house\s+seats\b",),
    "not_running_count": (r"\bhow\s+many\b.{0,40}\b(?:members|senators)\b.{0,25}\bnot\s+running\b",),
    "criminal_charge": (r"\bcharg(?:e|ed|es|ing)\b",),
    "ai_top_rank": (r"\btop[ -]?ranked\s+ai\s+model\b",),
    "ai_score_threshold": (r"\b(?:hit|reach)\w*\b.{0,20}\b\d{3,4}\b.{0,30}\b(?:arena|benchmark|score)\b",),
    "song_performer": (r"\b(?:sing|perform)\w*\b.{0,35}\bsong\b",),
    "person_of_year": (r"\bperson\s+of\s+the\s+year\b",),
    "richest_person": (r"\brichest\s+(?:person|man|woman)\b",),
    "google_search_ranking": (r"\bsearch(?:ed|es)?\b.{0,35}\bgoogle\b", r"\bgoogle\b.{0,35}\bsearch(?:ed|es)?\b"),
    "sexiest_person": (r"\bsexiest\s+(?:man|woman|person)\b",),
    "candidate_nomination": (r"\b(?:nominee|nomination)\b", r"\bprimary\s+winner\b"),
    "election_winner": (r"\b(?:general\s+|special\s+)?election\s+winner\b", r"\bwin\w*\b.{0,80}\belection\b"),
    "governor_race": (r"\b(?:governor|governorship|gubernatorial)\b",),
    "attorney_general_race": (r"\battorney\s+general\b",),
    "district_race": (r"\b[a-z]{2}[ -]\d{1,2}\b",),
    "political_endorsement": (r"\bendorse\w*\b",),
    "party_leader_election": (r"\b(?:senate|house)\b.{0,30}\bleader\s+election\b",),
    "runoff_advancement": (r"\badvance\w*\s+to\b.{0,30}\brunoff\b",),
    "primary_advancement": (r"\badvance\w*\s+from\b.{0,40}\bprimary\b", r"\bprimary\s+advancers?\b"),
    "primary_ranked_finish": (
        r"\b(?:\d+(?:st|nd|rd|th)|first|second|third)\s+place\b.{0,60}\bprimary\b",
        r"\bprimary\b.{0,60}\b(?:\d+(?:st|nd|rd|th)|first|second|third)\s+place\b",
        r"\bplace\s+(?:first|second|third)\b.{0,60}\bprimary\b",
        r"\bprimary\b.{0,60}\bplace\s+(?:first|second|third)\b",
    ),
    "first_round_winner": (r"\b(?:1st|first)\s+round\s+winner\b",),
    "cover_athlete": (r"\bcover\s+of\b.{0,35}\b(?:2k|madden|fc|fifa|nhl|mlb)\w*\b",),
    "ipo_announcement": (r"\b(?:announce|file|files|filed|filing)\w*\b.{0,30}\bipo\b", r"\bipo\b.{0,30}\b(?:announce|file|files|filed|filing)\w*\b"),
    "ipo_first": (r"\bipo\b.{0,30}\bfirst\b", r"\bfirst\b.{0,30}\bipo\b"),
    "largest_ipo": (r"\blargest\s+ipo\b", r"\bipo\b.{0,35}\b(?:largest|market\s+cap)\b"),
    "fed_governor_role": (r"\b(?:federal\s+reserve|fed)\s+governor\b",),
    "fed_chair_role": (r"\b(?:federal\s+reserve|fed)\s+(?:chair|chairman|chairwoman)\b",),
    "award_total_count": (r"\b(?:most|how\s+many)\s+(?:oscars?|academy\s+awards?)\b",),
    "award_category_win": (
        r"\b(?:academy\s+awards?|oscars?)\b.{0,80}\bbest\s+[a-z]",
        r"\bbest\s+[a-z].{0,80}\b(?:academy\s+awards?|oscars?)\b",
    ),
}

# Specific competition names are stronger evidence than the shared sport or team.
# Multiple keys may be extracted (for example NCAAB + WCC); any shared key keeps
# the pair compatible, while two explicit disjoint namespaces are a hard reject.
SPORT_COMPETITION_PATTERNS = (
    ("competition:womens_ucl", (r"\bwomens?\s+(?:uefa\s+)?champions\s+league\b",)),
    ("competition:ucl", (r"\b(?:uefa\s+)?champions\s+league\b",)),
    ("competition:europa_league", (r"\b(?:uefa\s+)?europa\s+league\b",)),
    ("competition:europa_conference_league", (r"\b(?:uefa\s+)?(?:europa\s+)?conference\s+league\b",)),
    ("competition:fifa_world_cup", (r"\bfifa\s+world\s+cup\b",)),
    ("competition:mls", (r"\bmls\b", r"\bmajor\s+league\s+soccer\b")),
    ("competition:fa_cup", (r"\bfa\s+cup\b",)),
    ("competition:carabao_cup", (r"\bcarabao\s+cup\b",)),
    ("competition:epl", (r"\benglish\s+premier\s+league\b", r"\bepl\b")),
    ("competition:serie_a", (r"\bserie\s+a\b",)),
    ("competition:ligue_1", (r"\bligue\s+1\b",)),
    ("competition:greece_super_league", (r"\bgreece\s+super\s+league\b",)),
    ("competition:masters_golf", (r"\b(?:the\s+)?masters(?:\s+tournament)?\b",)),
    ("competition:french_open", (r"\bfrench\s+open\b",)),
    ("competition:wimbledon", (r"\bwimbledon\b",)),
    ("competition:us_open_tennis", (r"\bus\s+open\b.{0,20}\btennis\b", r"\b(?:mens|womens)\s+us\s+open\s+winner\b")),
    ("competition:indian_wells", (r"\bindian\s+wells\b",)),
    ("competition:calendar_grand_slam", (r"\bcalendar\s+grand\s+slam\b",)),
    ("competition:belgian_pro_league", (r"\bbelgian\s+pro\s+league\b",)),
    ("competition:wbc", (r"\bworld\s+baseball\s+classic\b", r"\bwbc\b")),
    ("competition:mlb", (r"\bmajor\s+league\s+baseball\b", r"\bmlb\b", r"\bpro\s+baseball\b")),
    ("competition:nba", (r"\bnba\b", r"\bpro\s+basketball\b")),
    ("competition:ncaa_basketball", (r"\bncaab\b", r"\bncaa\b.{0,25}\bbasketball\b", r"\bcollege\s+basketball\b")),
    ("competition:ncaa_wcc", (r"\bwcc\b",)),
    ("competition:ncaa_big_12", (r"\bbig\s+12\b",)),
    ("competition:ncaa_acc", (r"\bacc\b.{0,35}\b(?:basketball|conference|champion|winner)\b",)),
    ("competition:nfl", (r"\bnfl\b", r"\bpro\s+football\b")),
    ("competition:ncaa_football", (r"\bncaaf\b", r"\bncaa\b.{0,25}\bfootball\b", r"\bcollege\s+football\b")),
    ("competition:nhl", (r"\bnhl\b", r"\bpro\s+hockey\b")),
    ("competition:six_nations", (r"\bsix\s+nations\b",)),
)

AWARD_CATEGORY_PATTERNS = (
    ("supporting_actor", (r"\bsupporting\s+actor\b", r"\bmale\s+actor\b.{0,35}\bsupporting\s+role\b")),
    ("supporting_actress", (r"\bsupporting\s+actress\b", r"\bfemale\s+actor\b.{0,35}\bsupporting\s+role\b")),
    ("original_screenplay", (r"\boriginal\s+screenplay\b",)),
    ("adapted_screenplay", (r"\badapted\s+screenplay\b",)),
    ("original_score", (r"\boriginal\s+score\b", r"\bbest\s+music\b.{0,25}\bscore\b")),
    ("original_song", (r"\boriginal\s+song\b",)),
    ("documentary_feature", (r"\bdocumentary\s+feature\b",)),
    ("international_feature", (r"\binternational\s+feature\b",)),
    ("animated_short", (r"\banimated\s+short\b",)),
    ("production_design", (r"\bproduction\s+design\b",)),
    ("costume_design", (r"\bcostume\s+design\b",)),
    ("makeup_hairstyling", (r"\bmakeup\b.{0,20}\bhairstyling\b",)),
    ("film_editing", (r"\bfilm\s+editing\b",)),
    ("visual_effects", (r"\bvisual\s+effects\b",)),
    ("cinematography", (r"\bcinematography\b",)),
    ("best_casting", (r"\bbest\s+casting\b",)),
    ("best_director", (r"\bbest\s+director\b",)),
    ("best_picture", (r"\bbest\s+picture\b",)),
    ("best_actor", (r"\bbest\s+actor\b",)),
    ("best_actress", (r"\bbest\s+actress\b",)),
    ("best_sound", (r"\bbest\s+sound\b",)),
)

SPORT_METRIC_DIMENSIONS = {
    "stat_era", "stat_doubles", "stat_war", "stat_rbi", "stat_ops",
    "stat_hits", "stat_batting_average", "stat_triples", "stat_home_runs",
    "stat_stolen_bases", "stat_runs", "stat_pitching_wins", "stat_points",
    "stat_rebounds", "stat_assists", "stat_steals", "stat_blocks",
    "stat_three_pointers",
}

SPORT_AWARD_DIMENSIONS = {
    "award_cy_young", "award_hank_aaron", "award_mvp", "award_hart",
    "award_art_ross", "award_rocket_richard", "award_calder", "award_vezina",
    "award_jack_adams", "award_clutch", "award_defensive_player",
    "award_most_improved", "award_james_norris", "award_conference_finals_mvp",
    "award_finals_mvp",
    "award_comeback_player", "award_presidents_trophy", "award_selke",
    "award_rookie_of_year", "award_player_of_year",
}

SPORT_TITLE_DIMENSIONS = {
    "division_title", "conference_title", "stanley_cup_title",
    "regular_season_title", "conference_tournament_title", "national_title",
    "cover_athlete", "playoffs_qualification", "player_transfer",
}

INCOMPATIBLE_INTENT_DIMENSIONS = (
    {
        "arrest", "presidential_run", "presidential_election", "presidential_nomination",
        "office_exit", "office_entry", "impeachment", "retirement", "medal_award",
        "criminal_charge", "pardon", "runoff_advancement", "cover_athlete",
    },
    {"album_release", "song_release"},
    {"album_market", "song_market"},
    {
        "album_release", "song_release", "album_feature", "album_chart", "song_chart",
        "artist_ranking", "tour_announcement", "song_performer",
    },
    {"token_launch", "government_stake", "ipo", "bankruptcy", "acquisition"},
    {"country_visit", "diplomatic_meeting", "pardon", "trade_deal", "recognition", "relations_normalization"},
    {"scripted_series", "device_product"},
    {"national_chamber_control", "individual_legislative_race", "majority_loss"},
    {"unemployment_metric", "inflation_metric", "gdp_metric"},
    {"bond_role_q", "bond_role_bond"},
    {"winner", "competition_elimination"},
    {"winner", "qualifier", "round_of_16", "quarterfinal"},
    SPORT_METRIC_DIMENSIONS | SPORT_AWARD_DIMENSIONS | SPORT_TITLE_DIMENSIONS,
    {"ufc_pound_for_pound", "ufc_weight_title", "ufc_any_title"},
    {"house_seat_count", "not_running_count"},
    {"ai_top_rank", "ai_score_threshold"},
    {
        "person_of_year", "richest_person", "google_search_ranking", "sexiest_person",
        "artist_ranking", "album_chart", "song_chart",
    },
    {"candidate_nomination", "election_winner"},
    {"candidate_nomination", "political_endorsement", "party_leader_election"},
    {
        "candidate_nomination", "political_endorsement", "primary_advancement",
        "primary_ranked_finish", "first_round_winner",
    },
    {"governor_race", "attorney_general_race", "district_race", "party_leader_election"},
    {"election_winner", "runoff_advancement"},
    {"overall_election_winner", "first_round_winner", "runoff_advancement"},
    {
        "conversation_with", "diplomatic_meeting", "person_visit_location",
        "legislative_vote", "office_entry", "speech_mention",
    },
    {"ipo_announcement", "ipo_first", "largest_ipo"},
    {"fed_governor_role", "fed_chair_role"},
    {"award_total_count", "award_category_win"},
)


@dataclass(frozen=True)
class OutcomeProfile:
    platform: str
    event_id: str
    bet_id: str
    original_market_id: str
    market_title: str
    bet_title: str
    category: str
    sub_category: str
    outcome_name: str
    outcome_subject: str
    outcome_key: str
    market_intent: str
    market_keywords: frozenset[str]
    scope_terms: frozenset[str]
    deadline_date: str
    event_years: frozenset[str]
    actor_key: str
    subject_source: str
    event_subject_count: int

    @property
    def is_multi_outcome_event(self) -> bool:
        return self.event_subject_count > 1


def safe_text(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def ascii_fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("ascii", "ignore").decode("ascii")


def normalize_phrase(text: str) -> str:
    text = ascii_fold(safe_text(text)).lower()
    text = text.replace("&", " and ")
    text = text.replace("’", "'").replace("`", "'")
    text = re.sub(r"\bmen's\b", "mens", text)
    text = re.sub(r"\bwomen's\b", "womens", text)
    text = re.sub(r"\bu\.s\.a\.\b", "usa", text)
    text = re.sub(r"\bu\.s\.\b", "us", text)
    text = re.sub(r"[^a-z0-9.#%$]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return SUBJECT_ALIASES.get(text, text)


MONTH_NUMBERS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


@lru_cache(maxsize=None)
def extract_deadline_date(*values: str) -> str:
    pattern = rf"\b({MONTH_NAMES})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,)?\s+(20\d{{2}})\b"
    candidates = []
    for value_index, value in enumerate(values):
        text = safe_text(value).lower()
        for match in re.finditer(pattern, text):
            month_name, raw_day, raw_year = match.groups()
            try:
                parsed = date(int(raw_year), MONTH_NUMBERS[month_name], int(raw_day)).isoformat()
            except (KeyError, ValueError):
                continue

            sentence_start = max(text.rfind(".", 0, match.start()), text.rfind("\n", 0, match.start())) + 1
            next_period = text.find(".", match.end())
            sentence_end = next_period if next_period >= 0 else len(text)
            sentence = text[sentence_start:sentence_end]
            sentence_prefix = text[sentence_start:match.start()]
            nearby_prefix = text[max(0, match.start() - 70):match.start()]
            if re.search(r"\bif\b", sentence) and re.search(r"\bresolve\w*\s+to\s+[\"']?other\b", sentence):
                continue
            if re.search(
                r"\bif\b.{0,80}\b(?:no winner|no result|nothing|none)\b.{0,35}\bby\s*$",
                sentence_prefix,
            ):
                continue

            if len(text) <= 60:
                score = 6
            elif re.search(r"\b(?:ceremony|event|game|match)\b.{0,30}\b(?:held|take place|scheduled)?\s*on\s*$", nearby_prefix):
                score = 5
            elif re.search(r"\b(?:by|before|until|through)\s*$", nearby_prefix):
                score = 4
            elif re.search(r"\b(?:held|take place|scheduled)\s+(?:for|on)\s*$", nearby_prefix):
                score = 3
            elif re.search(r"\bon\s*$", nearby_prefix):
                score = 2
            else:
                continue
            candidates.append((score, -value_index, match.start(), parsed))

    return max(candidates)[-1] if candidates else ""


def deadlines_compatible(kalshi_deadline: str, polymarket_deadline: str) -> bool:
    if not kalshi_deadline or not polymarket_deadline:
        return True
    try:
        k_date = date.fromisoformat(kalshi_deadline)
        p_date = date.fromisoformat(polymarket_deadline)
    except ValueError:
        return True
    return abs((k_date - p_date).days) <= 1


def annual_window_conflicts_with_deadline(scope_terms: set[str], deadline: str) -> bool:
    if "period:annual" not in scope_terms or not deadline:
        return False
    try:
        deadline_date = date.fromisoformat(deadline)
    except ValueError:
        return False
    return (deadline_date.month, deadline_date.day) not in {(12, 31), (1, 1)}


def event_years_conflict_with_deadline(event_years: frozenset[str], deadline: str) -> bool:
    if not event_years or not deadline:
        return False
    try:
        deadline_date = date.fromisoformat(deadline)
    except ValueError:
        return False
    if str(deadline_date.year) in event_years:
        return False
    return not (
        deadline_date.month == 1
        and deadline_date.day == 1
        and str(deadline_date.year - 1) in event_years
    )


@lru_cache(maxsize=None)
def extract_actor_key(market_title: str) -> str:
    text = normalize_phrase(market_title)
    patterns = (
        r"^(?:what|which) countries will (.+?) visit\b",
        r"^what will (.+?) say\b",
        r"^what will be said on (?:the\s+(?:first|second|third)\s+)?(.+?) episode\b",
        r"^where will (.+?) meet\b",
        r"^which countries will (.+?) make (?:new )?trade deals?\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        actor = normalize_phrase(match.group(1))
        if actor not in {"", "who", "someone", "anyone"}:
            return actor
    return ""


def actor_keys_compatible(kalshi_actor: str, polymarket_actor: str) -> bool:
    if not kalshi_actor or not polymarket_actor:
        return True
    k_tokens = tokenize(kalshi_actor)
    p_tokens = tokenize(polymarket_actor)
    return bool(k_tokens.intersection(p_tokens))


@lru_cache(maxsize=None)
def extract_event_years(*values: str) -> frozenset[str]:
    years = set()
    for value in values:
        text = safe_text(value).lower()
        for match in re.finditer(r"\b20[2-3]\d\b", text):
            prefix = text[max(0, match.start() - 35):match.start()]
            if re.search(r"\b(?:before|by|until|through|deadline)\b.{0,25}$", prefix):
                continue
            years.add(match.group(0))
    return frozenset(years)


def event_years_compatible(kalshi_years: frozenset[str], polymarket_years: frozenset[str]) -> bool:
    if not kalshi_years or not polymarket_years:
        return True
    return bool(kalshi_years.intersection(polymarket_years))


def tokenize(text: str) -> set[str]:
    normalized = normalize_phrase(text)
    tokens = {
        TOKEN_ALIASES.get(token, token)
        for token in normalized.split()
        if token and token not in STOP_WORDS and len(token) > 1
    }
    tokens.update(extract_phrase_tokens(normalized))
    return tokens


def extract_phrase_tokens(normalized_text: str) -> set[str]:
    tokens = set()
    if "world cup" in normalized_text:
        tokens.add("world cup")
    if "world baseball classic" in normalized_text or re.search(r"\bwbc\b", normalized_text):
        tokens.add("world baseball classic")
    if "golden boot" in normalized_text:
        tokens.add("golden boot")
    if "super bowl" in normalized_text:
        tokens.add("super bowl")
    if "stanley cup" in normalized_text:
        tokens.add("stanley cup")
    if "champions league" in normalized_text:
        tokens.add("champions league")
    for match in re.finditer(r"\bgroup\s+([a-z])\b", normalized_text):
        tokens.add(f"group {match.group(1)}")
    return tokens


def meaningful_market_keywords(keywords: Iterable[str]) -> set[str]:
    return {
        token
        for token in keywords
        if token not in GENERIC_MARKET_TOKENS
        and not re.fullmatch(r"20\d{2}|\d+", token)
    }


def strip_polymarket_outcome_suffix(title: str) -> tuple[str, str]:
    text = safe_text(title)
    match = re.search(r"\s*[\[\(]\s*(yes|no)\s*[\]\)]\s*$", text, flags=re.IGNORECASE)
    if not match:
        return text.strip(), ""
    return text[: match.start()].strip(), match.group(1).title()


def is_generic_subject(subject: str) -> bool:
    normalized = normalize_phrase(subject)
    if normalized in GENERIC_OUTCOME_SUBJECTS:
        return True
    if re.fullmatch(rf"(?:{MONTH_NAMES})\s+\d{{1,2}}(?:\s+20\d{{2}})?", normalized):
        return True
    if re.fullmatch(r"(?:q[1-4]\s+)?20\d{2}", normalized):
        return True
    tokens = tokenize(normalized)
    if not tokens:
        return True
    return all(token in SCOPE_WORDS for token in tokens)


def infer_subject_from_title(title: str) -> str:
    normalized = safe_text(title)
    patterns = (
        r"^\s*will\s+(.+?)\s+win\s+(?:the\s+)?",
        r"^\s*will\s+(.+?)\s+qualify\s+",
        r"^\s*will\s+(.+?)\s+make\s+(?:the\s+)?",
        r"^\s*will\s+(.+?)\s+play\s+in\s+",
        r"^\s*will\s+(.+?)\s+(?:launch|release|announce|leave|retire|become|get|have|hit|reach)\b",
        r"^\s*will\s+(.+?)\s+be\s+",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if not match:
            continue
        subject = re.sub(r"\b(any|a|an|the)\b", " ", match.group(1), flags=re.IGNORECASE)
        subject = re.sub(r"\s+", " ", subject).strip()
        if subject and not is_generic_subject(subject):
            return subject
    return ""


def extract_outcome_subject(row: dict[str, str], platform: str) -> tuple[str, str]:
    bet_title = safe_text(row.get("Bet Title"))
    market_title = safe_text(row.get("Market Title"))

    if platform == "polymarket":
        candidate, _ = strip_polymarket_outcome_suffix(bet_title)
        source = "bet_title"
    else:
        candidate = bet_title
        source = "bet_title"

    if normalize_phrase(candidate) == normalize_phrase(market_title):
        candidate = ""

    if is_generic_subject(candidate):
        inferred = infer_subject_from_title(market_title)
        if inferred:
            return inferred, "market_title"
        return "", ""

    return candidate, source


def build_market_intent(market_title: str, outcome_subject: str) -> str:
    intent = normalize_phrase(market_title)
    subject = normalize_phrase(outcome_subject)
    if subject:
        intent = re.sub(rf"\b{re.escape(subject)}\b", " ", intent)
    intent = re.sub(r"\b(will|who|which|what|when|where|does|is|are)\b", " ", intent)
    intent = re.sub(r"\s+", " ", intent).strip()
    return intent


@lru_cache(maxsize=None)
def extract_scope_terms(market_intent: str) -> frozenset[str]:
    text = normalize_phrase(market_intent)
    tokens = tokenize(text)
    scopes = {token for token in tokens if token in SCOPE_WORDS}
    scopes.update({token for token in tokens if token.startswith("group ")})

    if {"president", "presidential"}.intersection(tokens):
        scopes.add("president")
    if {"senate", "senator", "senators"}.intersection(tokens):
        scopes.add("senate")
    if {"house", "congress", "congressional"}.intersection(tokens):
        scopes.add("house")
    if {"governor", "gubernatorial", "governorship"}.intersection(tokens):
        scopes.add("governor")
    if {"mayor", "mayoral"}.intersection(tokens):
        scopes.add("mayor")
    if "vp" in tokens or re.search(r"\bvice\s+president(?:ial)?\b", text):
        scopes.add("vice_president")
    if {"seat", "seats"}.intersection(tokens):
        scopes.add("seat_count")
    if {"city", "cities"}.intersection(tokens):
        scopes.add("city_count")

    if "win" in tokens or "winner" in tokens or "champion" in tokens:
        scopes.add("winner")
    if "qualify" in tokens or "qualifier" in tokens or "qualifiers" in tokens:
        scopes.add("qualifier")
    if "golden boot" in tokens:
        scopes.add("golden boot")
    if re.search(r"\btop\s+scorer\b", text):
        scopes.add("top_scorer")
    if re.search(r"\bsemi[ -]?final\b", text):
        scopes.add("semifinal")
    if re.search(r"\brelegat\w*\b", text):
        scopes.add("relegation")
    if re.search(r"\btop\s+(?:\d+|four|five|ten)\b", text):
        scopes.add("top_finish")
    if re.search(r"\bundefeated\b", text):
        scopes.add("undefeated")
    if re.search(r"\b(?:ranked|rankings?|poll)\b", text):
        scopes.add("competition_ranking")

    for platform_name in ("spotify", "billboard", "youtube", "apple music", "ifpi", "netflix"):
        if platform_name in text:
            scopes.add(f"platform:{platform_name}")

    rank_match = re.search(r"#\s*(\d+)\b", text)
    if rank_match:
        scopes.add(f"rank:{int(rank_match.group(1))}")
    elif re.search(r"\blast\s+place\b", text):
        scopes.add("rank:last")
    elif (ordinal_rank := re.search(r"\b(\d+)(?:st|nd|rd|th)\s+(?:overall\s+pick|place)\b", text)):
        scopes.add(f"rank:{int(ordinal_rank.group(1))}")
    elif (word_rank := re.search(r"\b(first|second|third)\s+(?:overall\s+pick|place)\b", text)):
        rank_value = {"first": 1, "second": 2, "third": 3}[word_rank.group(1)]
        scopes.add(f"rank:{rank_value}")
    elif re.search(r"\b(?:runner[ -]?up|second place)\b", text):
        scopes.add("rank:2")
    elif re.search(r"\b(?:top|most)\b", text) and re.search(r"\b(?:artist|album|song|listeners|scorer)\w*\b", text):
        scopes.add("rank:1")

    if re.search(r"\b(?:this year|annual|yearly|in 20\d{2})\b", text):
        scopes.add("period:annual")
    if re.search(rf"\b(?:{MONTH_NAMES}|monthly)\b", text):
        scopes.add("period:monthly")
    if re.search(r"\b(?:week|weekly)\b", text):
        scopes.add("period:weekly")
    if re.search(r"\b(?:day|daily)\b", text):
        scopes.add("period:daily")

    for match in re.finditer(r"\b(before|by|until|through)\s+(?:the\s+end\s+of\s+)?(20[2-3]\d)\b", text):
        relation, raw_year = match.groups()
        deadline_year = int(raw_year) - 1 if relation == "before" else int(raw_year)
        scopes.add(f"deadline_year:{deadline_year}")

    for dimension, patterns in INTENT_DIMENSION_PATTERNS.items():
        if any(re.search(pattern, text) for pattern in patterns):
            scopes.add(dimension)

    for party_name, aliases in {
        "democratic": ("democratic", "democrat"),
        "republican": ("republican", "gop"),
    }.items():
        if any(re.search(rf"\b{alias}\b", text) for alias in aliases):
            scopes.add(f"party:{party_name}")

    weight_match = re.search(
        r"\b(light\s+heavyweight|flyweight|bantamweight|featherweight|lightweight|welterweight|middleweight|heavyweight)\b",
        text,
    )
    if weight_match and "ufc" in tokens:
        scopes.add(f"weight_class:{weight_match.group(1).replace(' ', '_')}")

    if re.search(r"\bstate\s+(?:house|senate)\b", text):
        scopes.add("legislative_level:state")
    elif "district_race" in scopes and {"house", "congress", "congressional"}.intersection(tokens):
        scopes.add("legislative_level:federal")

    for competition_key, patterns in SPORT_COMPETITION_PATTERNS:
        if any(re.search(pattern, text) for pattern in patterns):
            scopes.add(competition_key)

    if "competition:wbc" in scopes:
        if re.search(r"\bpool\s+[a-z0-9]+\b", text):
            scopes.add("competition_stage:pool")
        elif "winner" in scopes:
            scopes.add("competition_stage:overall")

    if re.search(r"\b(?:nba|nfl|nhl|mlb|pro\s+(?:basketball|football|hockey|baseball))\b", text):
        scopes.add("sport_level:professional")
    if re.search(r"\b(?:ncaa[bf]?|ncaab|college\s+(?:basketball|football)|wcc|big\s+12)\b", text):
        scopes.add("sport_level:college")
    if re.search(r"\bwomens?\b", text):
        scopes.add("gender:women")
    elif re.search(r"\bmens?\b", text):
        scopes.add("gender:men")
    elif "competition:ucl" in scopes and "competition:womens_ucl" not in scopes:
        scopes.add("gender:men")

    if re.search(r"\b(?:oscars?|academy\s+awards?)\b", text):
        scopes.add("award_show:oscars")
    if re.search(r"\b(?:actor\s+awards?|screen\s+actors?\s+guild|sag\s+awards?)\b", text):
        scopes.add("award_show:actor_awards")
    for category_name, patterns in AWARD_CATEGORY_PATTERNS:
        if any(re.search(pattern, text) for pattern in patterns):
            scopes.add(f"award_category:{category_name}")

    if re.search(r"\b(?:say|said)\b", text):
        if re.search(r"\bnext\s+live\s+announcement\b", text):
            scopes.add("speech_context:live_announcement")
        elif "period:weekly" in scopes:
            scopes.add("speech_context:weekly")
        elif "period:monthly" in scopes:
            scopes.add("speech_context:monthly")

    manager_target = ""
    if match := re.search(r"\bnext\s+manager\s+of\s+(.+)$", text):
        manager_target = match.group(1)
    elif match := re.search(r"\bnext\s+(.+?)\s+manager\b", text):
        manager_target = match.group(1)
    manager_target = re.sub(r"\b20[2-3]\d\b", " ", manager_target)
    manager_target = re.sub(r"\s+", " ", manager_target).strip()
    if manager_target:
        scopes.add(f"manager_target:{manager_target}")

    if "bond_role_q" in scopes:
        scopes.discard("bond_role_bond")
    if "presidential_run" in scopes:
        scopes.discard("presidential_nomination")
    if "individual_legislative_race" in scopes:
        scopes.discard("national_chamber_control")
    if {"award_conference_finals_mvp", "award_finals_mvp"}.intersection(scopes):
        scopes.discard("award_mvp")
    if "ufc_weight_title" in scopes:
        scopes.discard("ufc_any_title")
    if "winner" in scopes:
        scopes.add("rank:1")
    if "election_winner" in scopes and not {"first_round_winner", "runoff_advancement"}.intersection(scopes):
        scopes.add("overall_election_winner")
    return frozenset(scopes)


@lru_cache(maxsize=None)
def extract_context_scope_terms(event_id: str, category: str, sub_category: str) -> frozenset[str]:
    context = normalize_phrase(f"{category} {sub_category}")
    event_key = re.sub(r"[^A-Z0-9]", "", safe_text(event_id).upper())
    scopes = set()

    for competition_key, patterns in SPORT_COMPETITION_PATTERNS:
        if any(re.search(pattern, context) for pattern in patterns):
            scopes.add(competition_key)

    event_competitions = (
        ("NBA", "competition:nba", "sport_level:professional"),
        ("NFL", "competition:nfl", "sport_level:professional"),
        ("NHL", "competition:nhl", "sport_level:professional"),
        ("MLB", "competition:mlb", "sport_level:professional"),
        ("NCAAB", "competition:ncaa_basketball", "sport_level:college"),
        ("NCAAF", "competition:ncaa_football", "sport_level:college"),
        ("WBC", "competition:wbc", "sport_level:professional"),
    )
    for marker, competition_key, level_key in event_competitions:
        if marker in event_key:
            scopes.update({competition_key, level_key})

    if re.search(r"\b(?:nba|nfl|nhl|mlb|professional)\b", context):
        scopes.add("sport_level:professional")
    if re.search(r"\b(?:ncaa[bf]?|ncaab|college)\b", context):
        scopes.add("sport_level:college")
    if re.search(r"\bwomens?\b", context):
        scopes.add("gender:women")
    elif re.search(r"\bmens?\b", context):
        scopes.add("gender:men")
    return frozenset(scopes)


def subject_key(subject: str) -> str:
    normalized = normalize_phrase(subject)
    return SUBJECT_ALIASES.get(normalized, normalized)


def load_category_mapping() -> dict[str, set[str]]:
    if not os.path.exists(CATEGORY_MAPPING_FILE):
        return {}
    with open(CATEGORY_MAPPING_FILE, "r") as handle:
        return normalize_category_mapping(json.load(handle))


def rows_from_dataframe(df: pd.DataFrame) -> Iterable[dict[str, str]]:
    columns = [str(column) for column in df.columns]
    for values in df.fillna("").itertuples(index=False, name=None):
        yield {column: safe_text(value) for column, value in zip(columns, values)}


def build_outcome_profiles(df: pd.DataFrame, platform: str) -> list[OutcomeProfile]:
    raw_profiles = []
    for row in rows_from_dataframe(df):
        if platform == "polymarket":
            outcome_name = safe_text(row.get("Outcome_Name"))
            _, suffix_side = strip_polymarket_outcome_suffix(row.get("Bet Title"))
            if outcome_name and outcome_name.lower() == "no":
                continue
            if suffix_side and suffix_side.lower() == "no":
                continue
        else:
            outcome_name = "Yes"

        subject, subject_source = extract_outcome_subject(row, platform)
        key = subject_key(subject)
        if is_generic_subject(key):
            continue

        market_title = safe_text(row.get("Market Title"))
        market_intent = build_market_intent(market_title, subject)
        market_keywords = tokenize(market_intent)
        category = normalize_label(row.get("Category"))
        sub_category = normalize_label(row.get("Sub-Category"))
        scope_terms = set(extract_scope_terms(market_intent))
        scope_terms.update(
            extract_context_scope_terms(
                row.get("Event ID"),
                category,
                sub_category,
            )
        )
        actor_key = extract_actor_key(market_title)
        deadline_date = extract_deadline_date(
            row.get("Bet_Rules"),
            row.get("Event_Description"),
            row.get("Description"),
            row.get("Bet Title"),
            market_title,
        )
        # Resolution-rule years often describe a deadline boundary rather than
        # the event season. Deadlines are compared separately and more precisely.
        event_years = extract_event_years(market_title)

        raw_profiles.append(
            {
                "platform": platform,
                "event_id": safe_text(row.get("Event ID")),
                "bet_id": safe_text(row.get("Bet ID")),
                "original_market_id": safe_text(row.get("Original_Market_ID")),
                "market_title": market_title,
                "bet_title": safe_text(row.get("Bet Title")),
                "category": category,
                "sub_category": sub_category,
                "outcome_name": outcome_name or "Yes",
                "outcome_subject": subject,
                "outcome_key": key,
                "market_intent": market_intent,
                "market_keywords": frozenset(market_keywords),
                "scope_terms": frozenset(scope_terms),
                "deadline_date": deadline_date,
                "event_years": event_years,
                "actor_key": actor_key,
                "subject_source": subject_source,
            }
        )

    event_subjects = defaultdict(set)
    for profile in raw_profiles:
        event_subjects[profile["event_id"]].add(profile["outcome_key"])

    return [
        OutcomeProfile(
            **profile,
            event_subject_count=len(event_subjects[profile["event_id"]]),
        )
        for profile in raw_profiles
    ]


def has_scope_conflict(kalshi: OutcomeProfile, polymarket: OutcomeProfile) -> bool:
    k_scope = set(kalshi.scope_terms)
    p_scope = set(polymarket.scope_terms)

    if not deadlines_compatible(kalshi.deadline_date, polymarket.deadline_date):
        return True
    if annual_window_conflicts_with_deadline(k_scope, polymarket.deadline_date):
        return True
    if annual_window_conflicts_with_deadline(p_scope, kalshi.deadline_date):
        return True
    if event_years_conflict_with_deadline(kalshi.event_years, polymarket.deadline_date):
        return True
    if event_years_conflict_with_deadline(polymarket.event_years, kalshi.deadline_date):
        return True
    if not event_years_compatible(kalshi.event_years, polymarket.event_years):
        return True
    if not actor_keys_compatible(kalshi.actor_key, polymarket.actor_key):
        return True

    k_groups = {term for term in k_scope if term.startswith("group ")}
    p_groups = {term for term in p_scope if term.startswith("group ")}
    if k_groups or p_groups:
        if k_groups != p_groups:
            return True

    if ("winner" in k_scope and "qualifier" in p_scope) or ("qualifier" in k_scope and "winner" in p_scope):
        return True

    if ("golden boot" in k_scope) != ("golden boot" in p_scope):
        return True

    for dimensions in INCOMPATIBLE_INTENT_DIMENSIONS:
        k_dimensions = k_scope.intersection(dimensions)
        p_dimensions = p_scope.intersection(dimensions)
        if k_dimensions and p_dimensions and k_dimensions.isdisjoint(p_dimensions):
            return True

    competition_dimensions = {
        "winner", "qualifier", "semifinal", "top_scorer", "relegation",
        "top_finish", "undefeated", "competition_participation",
        "competition_ranking", "competition_finalist", "competition_knockout_advance",
        "competition_seed", "competition_play_in", "playoffs_qualification",
        "primary_advancement", "runoff_advancement",
    }
    k_competition = k_scope.intersection(competition_dimensions)
    p_competition = p_scope.intersection(competition_dimensions)
    if k_competition and p_competition and k_competition.isdisjoint(p_competition):
        return True

    k_platforms = {term for term in k_scope if term.startswith("platform:")}
    p_platforms = {term for term in p_scope if term.startswith("platform:")}
    if k_platforms and p_platforms and k_platforms.isdisjoint(p_platforms):
        return True

    for dimension_prefix in ("award_show:", "award_category:", "speech_context:", "manager_target:"):
        k_dimensions = {term for term in k_scope if term.startswith(dimension_prefix)}
        p_dimensions = {term for term in p_scope if term.startswith(dimension_prefix)}
        if k_dimensions and p_dimensions and k_dimensions.isdisjoint(p_dimensions):
            return True

    k_competitions = {term for term in k_scope if term.startswith("competition:")}
    p_competitions = {term for term in p_scope if term.startswith("competition:")}
    if k_competitions and p_competitions and k_competitions.isdisjoint(p_competitions):
        return True

    k_levels = {term for term in k_scope if term.startswith("sport_level:")}
    p_levels = {term for term in p_scope if term.startswith("sport_level:")}
    if k_levels and p_levels and k_levels.isdisjoint(p_levels):
        return True

    k_genders = {term for term in k_scope if term.startswith("gender:")}
    p_genders = {term for term in p_scope if term.startswith("gender:")}
    if k_genders and p_genders and k_genders.isdisjoint(p_genders):
        return True

    k_stages = {term for term in k_scope if term.startswith("competition_stage:")}
    p_stages = {term for term in p_scope if term.startswith("competition_stage:")}
    if k_stages and p_stages and k_stages.isdisjoint(p_stages):
        return True

    knockout_stages = {"round_of_16", "quarterfinal", "semifinal"}
    k_knockout_stages = k_scope.intersection(knockout_stages)
    p_knockout_stages = p_scope.intersection(knockout_stages)
    if k_knockout_stages and p_knockout_stages and k_knockout_stages.isdisjoint(p_knockout_stages):
        return True

    k_deadline_years = {term for term in k_scope if term.startswith("deadline_year:")}
    p_deadline_years = {term for term in p_scope if term.startswith("deadline_year:")}
    if k_deadline_years and p_deadline_years and k_deadline_years.isdisjoint(p_deadline_years):
        return True
    k_event_year_terms = {f"deadline_year:{year}" for year in kalshi.event_years}
    p_event_year_terms = {f"deadline_year:{year}" for year in polymarket.event_years}
    if k_event_year_terms and p_deadline_years and k_event_year_terms.isdisjoint(p_deadline_years):
        return True
    if p_event_year_terms and k_deadline_years and p_event_year_terms.isdisjoint(k_deadline_years):
        return True

    k_ranks = {term for term in k_scope if term.startswith("rank:")}
    p_ranks = {term for term in p_scope if term.startswith("rank:")}
    if k_ranks and p_ranks and k_ranks.isdisjoint(p_ranks):
        return True

    ranking_dimensions = {"artist_ranking", "album_chart", "song_chart", "top_scorer"}
    if k_scope.intersection(ranking_dimensions) and p_scope.intersection(ranking_dimensions):
        k_periods = {term for term in k_scope if term.startswith("period:")}
        p_periods = {term for term in p_scope if term.startswith("period:")}
        if k_periods and p_periods and k_periods.isdisjoint(p_periods):
            return True

    political_offices = {
        "president", "vice_president", "senate", "house", "governor", "mayor",
        "attorney_general_race", "district_race",
    }
    k_offices = k_scope.intersection(political_offices)
    p_offices = p_scope.intersection(political_offices)
    if k_offices and p_offices and k_offices.isdisjoint(p_offices):
        return True

    k_parties = {term for term in k_scope if term.startswith("party:")}
    p_parties = {term for term in p_scope if term.startswith("party:")}
    if k_parties and p_parties and k_parties.isdisjoint(p_parties):
        return True

    k_weight_classes = {term for term in k_scope if term.startswith("weight_class:")}
    p_weight_classes = {term for term in p_scope if term.startswith("weight_class:")}
    if k_weight_classes and p_weight_classes and k_weight_classes.isdisjoint(p_weight_classes):
        return True

    k_legislative_levels = {term for term in k_scope if term.startswith("legislative_level:")}
    p_legislative_levels = {term for term in p_scope if term.startswith("legislative_level:")}
    if k_legislative_levels and p_legislative_levels and k_legislative_levels.isdisjoint(p_legislative_levels):
        return True

    count_scopes = {"seat_count", "city_count"}
    k_counts = k_scope.intersection(count_scopes)
    p_counts = p_scope.intersection(count_scopes)
    if k_counts and p_counts and k_counts.isdisjoint(p_counts):
        return True

    return False


def categories_compatible(
    kalshi: OutcomeProfile,
    polymarket: OutcomeProfile,
    category_mapping: dict[str, set[str]] | None = None,
) -> bool:
    if categories_overlap(kalshi.category, kalshi.sub_category, polymarket.category, polymarket.sub_category):
        return True

    category_mapping = category_mapping or {}
    if not category_mapping:
        return False

    k_key = kalshi_category_key(kalshi.category, kalshi.sub_category)
    mapped_poly_keys = mapped_polymarket_keys_for_kalshi(k_key, category_mapping)
    poly_keys = polymarket_category_keys(polymarket.category, polymarket.sub_category)
    return bool(mapped_poly_keys.intersection(poly_keys))


def market_intent_compatible(
    kalshi: OutcomeProfile,
    polymarket: OutcomeProfile,
    category_mapping: dict[str, set[str]] | None = None,
) -> bool:
    k_event_data = {
        "title": kalshi.market_title,
        "category": kalshi.category,
        "sub_category": kalshi.sub_category,
    }
    p_event_data = {
        "title": polymarket.market_title,
        "category": polymarket.category,
        "sub_category": polymarket.sub_category,
    }
    if event_scope_granularity_conflict(
        kalshi.event_id,
        polymarket.event_id,
        k_event_data,
        p_event_data,
    ):
        return False
    if event_location_conflict(kalshi.market_title, polymarket.market_title):
        return False
    if event_sports_domain_conflict(
        k_event_data,
        p_event_data,
    ):
        return False
    if not categories_compatible(kalshi, polymarket, category_mapping):
        return False

    if has_scope_conflict(kalshi, polymarket):
        return False

    shared = set(kalshi.market_keywords).intersection(polymarket.market_keywords)
    meaningful_shared = meaningful_market_keywords(shared)
    if not meaningful_shared:
        return False
    if len(shared) >= 2:
        return True

    shared_scope = set(kalshi.scope_terms).intersection(polymarket.scope_terms)
    return bool(shared_scope and meaningful_shared)


def score_pair(kalshi: OutcomeProfile, polymarket: OutcomeProfile) -> tuple[int, list[str], list[str]]:
    shared_market = sorted(set(kalshi.market_keywords).intersection(polymarket.market_keywords))
    shared_outcome = sorted(set(tokenize(kalshi.outcome_key)).intersection(tokenize(polymarket.outcome_key)))
    score = len(shared_market) + 3 * len(shared_outcome)
    if kalshi.outcome_key == polymarket.outcome_key:
        score += 12
    if set(kalshi.scope_terms).intersection(polymarket.scope_terms):
        score += 3
    if kalshi.is_multi_outcome_event or polymarket.is_multi_outcome_event:
        score += 2
    return score, shared_market, shared_outcome


def candidate_type(kalshi: OutcomeProfile, polymarket: OutcomeProfile) -> str:
    if kalshi.is_multi_outcome_event and polymarket.is_multi_outcome_event:
        return "outcome_outcome"
    return "event_outcome"


def profile_to_output(profile: OutcomeProfile, platform: str) -> dict[str, str]:
    base = {
        "event_id": profile.event_id,
        "bet_id": profile.bet_id,
        "market_title": profile.market_title,
        "bet_title": profile.bet_title,
        "outcome_subject": profile.outcome_subject,
        "outcome_key": profile.outcome_key,
        "market_intent": profile.market_intent,
        "deadline_date": profile.deadline_date,
        "event_years": sorted(profile.event_years),
        "actor_key": profile.actor_key,
        "category": profile.category,
        "sub_category": profile.sub_category,
        "subject_source": profile.subject_source,
        "event_subject_count": profile.event_subject_count,
    }
    if platform == "polymarket":
        base["original_market_id"] = profile.original_market_id
        base["outcome_name"] = profile.outcome_name
    return base


def build_candidate(
    candidate_id: int,
    kalshi: OutcomeProfile,
    polymarket: OutcomeProfile,
    match_score: int,
    shared_market: list[str],
    shared_outcome: list[str],
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "candidate_type": candidate_type(kalshi, polymarket),
        "kalshi": profile_to_output(kalshi, "kalshi"),
        "polymarket": profile_to_output(polymarket, "polymarket"),
        "shared_market_keywords": shared_market,
        "shared_outcome_keywords": shared_outcome,
        "match_score": match_score,
        "llm_verified": None,
    }


def generate_outcome_candidates(
    kalshi_df: pd.DataFrame,
    polymarket_df: pd.DataFrame,
    max_candidates_per_kalshi_bet: int = DEFAULT_MAX_CANDIDATES_PER_KALSHI_BET,
    category_mapping: dict[str, set[str]] | None = None,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    kalshi_profiles = build_outcome_profiles(kalshi_df, "kalshi")
    polymarket_profiles = build_outcome_profiles(polymarket_df, "polymarket")
    category_mapping = load_category_mapping() if category_mapping is None else category_mapping

    polymarket_by_outcome = defaultdict(list)
    for profile in polymarket_profiles:
        polymarket_by_outcome[profile.outcome_key].append(profile)

    candidates = []
    for kalshi in kalshi_profiles:
        scored_pairs = []
        for polymarket in polymarket_by_outcome.get(kalshi.outcome_key, []):
            if not (kalshi.is_multi_outcome_event or polymarket.is_multi_outcome_event):
                continue
            if not market_intent_compatible(kalshi, polymarket, category_mapping):
                continue
            score, shared_market, shared_outcome = score_pair(kalshi, polymarket)
            scored_pairs.append((score, shared_market, shared_outcome, polymarket))

        scored_pairs.sort(key=lambda item: item[0], reverse=True)
        for score, shared_market, shared_outcome, polymarket in scored_pairs[:max_candidates_per_kalshi_bet]:
            candidates.append(
                build_candidate(
                    len(candidates) + 1,
                    kalshi,
                    polymarket,
                    score,
                    shared_market,
                    shared_outcome,
                )
            )

    stats = {
        "kalshi_profiles": len(kalshi_profiles),
        "polymarket_profiles": len(polymarket_profiles),
        "kalshi_multi_outcome_events": count_multi_events(kalshi_profiles),
        "polymarket_multi_outcome_events": count_multi_events(polymarket_profiles),
        "candidate_pairs": len(candidates),
    }
    stats.update(candidate_type_counts(candidates))
    return candidates, stats


def count_multi_events(profiles: list[OutcomeProfile]) -> int:
    event_counts = {profile.event_id: profile.event_subject_count for profile in profiles}
    return sum(1 for count in event_counts.values() if count > 1)


def candidate_type_counts(candidates: list[dict[str, object]]) -> dict[str, int]:
    counts = Counter(str(candidate["candidate_type"]) for candidate in candidates)
    return {f"{candidate_type}_pairs": count for candidate_type, count in counts.items()}


def build_outcome_bundles(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped = defaultdict(list)
    for candidate in candidates:
        key = (
            str(candidate["candidate_type"]),
            str(candidate["kalshi"]["event_id"]),
            str(candidate["polymarket"]["event_id"]),
        )
        grouped[key].append(candidate)

    bundles = []
    for candidate_type_name, kalshi_event_id, polymarket_event_id in sorted(grouped):
        members = sorted(
            grouped[(candidate_type_name, kalshi_event_id, polymarket_event_id)],
            key=lambda item: (
                str(item["kalshi"].get("outcome_key", "")),
                str(item["polymarket"].get("outcome_key", "")),
                str(item["kalshi"].get("bet_id", "")),
                str(item["polymarket"].get("bet_id", "")),
            ),
        )
        first = members[0]
        raw_key = f"{candidate_type_name}|{kalshi_event_id}|{polymarket_event_id}"
        bundle_id = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:20]
        mappings = [
            {
                "candidate_id": member["candidate_id"],
                "kalshi_bet_id": member["kalshi"].get("bet_id", ""),
                "polymarket_bet_id": member["polymarket"].get("bet_id", ""),
                "kalshi_outcome_key": member["kalshi"].get("outcome_key", ""),
                "polymarket_outcome_key": member["polymarket"].get("outcome_key", ""),
                "match_score": member.get("match_score", 0),
            }
            for member in members
        ]
        bundles.append({
            "bundle_id": bundle_id,
            "candidate_type": candidate_type_name,
            "kalshi_event_id": kalshi_event_id,
            "polymarket_event_id": polymarket_event_id,
            "kalshi_market_title": first["kalshi"].get("market_title", ""),
            "polymarket_market_title": first["polymarket"].get("market_title", ""),
            "mapping_count": len(mappings),
            "outcome_mappings": mappings,
        })
    return bundles


def write_candidates(candidates: list[dict[str, object]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(candidates, handle, indent=2, ensure_ascii=False)


def write_outcome_bundles(bundles: list[dict[str, object]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(bundles, handle, indent=2, ensure_ascii=False)


def load_market_csv(path: str, usecols: tuple[str, ...]) -> pd.DataFrame:
    available_columns = set(pd.read_csv(path, nrows=0).columns)
    selected_columns = [column for column in usecols if column in available_columns]
    return pd.read_csv(path, dtype=str, keep_default_na=False, usecols=selected_columns)


def run(
    kalshi_csv: str = KALSHI_CSV,
    polymarket_csv: str = POLYMARKET_CSV,
    output_json: str = OUTPUT_JSON,
    bundles_json: str = OUTCOME_BUNDLES_JSON,
    max_candidates_per_kalshi_bet: int = DEFAULT_MAX_CANDIDATES_PER_KALSHI_BET,
) -> dict[str, int]:
    started = time.time()
    kalshi_df = load_market_csv(kalshi_csv, KALSHI_USECOLS)
    polymarket_df = load_market_csv(polymarket_csv, POLYMARKET_USECOLS)
    candidates, stats = generate_outcome_candidates(
        kalshi_df,
        polymarket_df,
        max_candidates_per_kalshi_bet=max_candidates_per_kalshi_bet,
    )
    write_candidates(candidates, output_json)
    bundles = build_outcome_bundles(candidates)
    write_outcome_bundles(bundles, bundles_json)
    stats["parent_event_bundles"] = len(bundles)
    stats["elapsed_ms"] = int((time.time() - started) * 1000)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate outcome-aware candidate pairs.")
    parser.add_argument("--kalshi-csv", default=KALSHI_CSV)
    parser.add_argument("--polymarket-csv", default=POLYMARKET_CSV)
    parser.add_argument("--output", default=OUTPUT_JSON)
    parser.add_argument("--bundles-output", default=OUTCOME_BUNDLES_JSON)
    parser.add_argument("--max-candidates-per-kalshi-bet", type=int, default=DEFAULT_MAX_CANDIDATES_PER_KALSHI_BET)
    args = parser.parse_args()

    stats = run(
        kalshi_csv=args.kalshi_csv,
        polymarket_csv=args.polymarket_csv,
        output_json=args.output,
        bundles_json=args.bundles_output,
        max_candidates_per_kalshi_bet=args.max_candidates_per_kalshi_bet,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
