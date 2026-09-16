"""Structured, deterministic event profiles used by candidate clustering."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import re
import sys
import unicodedata
from typing import Iterable

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
COMMON_DIR = os.path.join(ROOT_DIR, "pipeline", "common")
if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)

try:
    from pipeline.common.abbreviations import expand_abbreviations
    from pipeline.common.keyword_config import canonicalize_keyword, find_domain_entity_tokens
except ImportError:  # Direct script execution from the clustering directory.
    from abbreviations import expand_abbreviations
    from keyword_config import canonicalize_keyword, find_domain_entity_tokens


PROFILE_SCHEMA_VERSION = "event-profile-v2"

MARKET_LAYER_PATTERNS = {
    "exact_score": r"\bexact score\b",
    "first_half": r"\b(?:half[ -]?time|first half|1st half)\b",
    "map": r"\bmap\s*\d*\b",
    "more_markets": r"\bmore markets\b",
    "spread": r"\b(?:spread|wins? by|margin of victory)\b",
    "total": r"\b(?:total|over/under|o/u)\b",
}

SCOPE_PATTERNS = {
    "aggregate": r"\b(?:which party|control of|majority in|majority of|balance of power)\b",
    "annual_end": r"\b(?:at the end of|by the end of|year[ -]?end)\b",
    "annual_window": r"\b(?:in|during)\s+20[2-3][0-9]\b",
    "award": r"\b(?:mvp|most valuable player|golden boot|top scorer|rookie of the year|player of the year)\b",
    "district": r"\b(?:district|[a-z]{2}-(?:\d{1,2}|al))\b",
    "general_election": r"\bgeneral election\b",
    "match": r"\b(?:match|fixture|game)\b|\b(?:vs\.?|v\.?)\b",
    "meeting": r"\bmeeting\b|\b(?:fed|fomc) decision in\b",
    "metric": r"\b(?:home ?runs?|strikeouts?|no hitter|perfect game|steals?|rbi|ops|doubles?|hits? leader|goals? leader|points? leader)\b",
    "primary": r"\b(?:primary|nominee|nomination)\b",
    "runoff": r"\b(?:runoff|second round)\b",
    "seat": r"\b(?:house|senate|mayoral|governor) race for\b",
    "season": r"\bseason\b",
    "tournament": r"\b(?:tournament|championship|champion|cup|open|world series|classic)\b",
    "women": r"\b(?:women|woman|women's|womens|female)\b|\(w\)",
}

ACTION_PATTERNS = {
    "approval": r"\b(?:approval|rating)\b",
    "qualify": r"\bqualif(?:y|ier|ies|ied)\b",
    "release": r"\b(?:release|released)\b",
    "retire": r"\b(?:retire|retirement)\b",
    "win": r"\b(?:win|wins|winner|champion)\b",
}

STAGE_PATTERNS = {
    "final": r"\bfinals?\b",
    "group_stage": r"\bgroup stage\b",
    "quarterfinal": r"\bquarter[ -]?finals?\b",
    "round_of_16": r"\bround of 16\b",
    "round_of_32": r"\bround of 32\b",
    "semifinal": r"\bsemi[ -]?finals?\b",
}

DIRECTION_PATTERNS = {
    "down": r"\b(?:at most|below|decrease|drop|fall|fewer than|less than|lower|lowest|under)\b",
    "exact": r"\b(?:exactly|equal to)\b",
    "up": r"\b(?:above|at least|exceed|greater than|higher|highest|more than|over)\b",
}

MONTHS = {
    "jan": "01", "january": "01", "feb": "02", "february": "02",
    "mar": "03", "march": "03", "apr": "04", "april": "04",
    "may": "05", "jun": "06", "june": "06", "jul": "07", "july": "07",
    "aug": "08", "august": "08", "sep": "09", "sept": "09",
    "september": "09", "oct": "10", "october": "10", "nov": "11",
    "november": "11", "dec": "12", "december": "12",
}

GENERIC_ENTITY_TERMS = {
    "baseball", "basketball", "champion", "cup", "democratic", "election",
    "fifa", "football", "governor", "group", "league", "match", "nominee",
    "party", "president", "pro", "republican", "season", "tournament", "world",
}


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[^a-z0-9%+.-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def canonical_phrase(value: object) -> str:
    expanded = expand_abbreviations(str(value or ""))
    text = normalize_text(expanded)
    text = text.replace("-", " ")
    text = re.sub(r"\b(?:the|fc|cf|sc|afc|club)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip(" .-")


def _extract_pattern_names(text: str, patterns: dict[str, str]) -> frozenset[str]:
    return frozenset(name for name, pattern in patterns.items() if re.search(pattern, text, re.I))


def extract_group_ids(title: str) -> frozenset[str]:
    values = re.findall(r"\bgroup\s+([a-z0-9]+)\b", title, re.I)
    return frozenset(value.lower() for value in values if len(value) == 1 or value.isdigit())


def extract_districts(title: str) -> frozenset[str]:
    return frozenset(
        f"{state.upper()}-{district.upper()}"
        for state, district in re.findall(r"\b([a-z]{2})-(\d{1,2}|al)\b", title, re.I)
    )


def extract_years(title: str) -> frozenset[str]:
    return frozenset(re.findall(r"\b(20[2-3][0-9])\b", title))


def extract_seasons(title: str) -> frozenset[str]:
    seasons = set()
    for first, second in re.findall(r"\b(20\d{2})\s*[-/]\s*(\d{2,4})\b", title):
        if len(second) == 2:
            second = first[:2] + second
        seasons.add(f"{first}-{second}")
    return frozenset(seasons)


def extract_dates(title: str) -> frozenset[str]:
    dates = set()
    month_pattern = "|".join(sorted(MONTHS, key=len, reverse=True))
    pattern = rf"\b({month_pattern})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s*(20[2-3][0-9]))?\b"
    for month, day, year in re.findall(pattern, title, re.I):
        dates.add(f"{year or '????'}-{MONTHS[month.lower()]}-{int(day):02d}")
    return frozenset(dates)


def extract_periods(title: str) -> frozenset[str]:
    periods = set()
    month_pattern = "|".join(sorted(MONTHS, key=len, reverse=True))
    for month, year in re.findall(rf"\b({month_pattern})\.?\s*(20[2-3][0-9])?\b", title, re.I):
        periods.add(f"{year or '????'}-{MONTHS[month.lower()]}")
    for quarter, year in re.findall(r"\b(Q[1-4])\s*(20[2-3][0-9])?\b", title, re.I):
        periods.add(f"{year or '????'}-{quarter.upper()}")
    return frozenset(periods)


def extract_locations(title: str) -> frozenset[str]:
    values = set()
    patterns = (
        r"\btemperature in ([A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){0,3})\b",
        r"\bmayor of ([A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){0,3})\b",
        r"\b(?:election|race) in ([A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){0,3})\b",
        r"\bgovernor of ([A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){0,3})\b",
    )
    for pattern in patterns:
        for match in re.findall(pattern, title):
            value = canonical_phrase(match)
            value = re.split(r"\b(?:on|by|before|after|during)\b", value, maxsplit=1)[0].strip()
            if value and value not in {"the", "united states"}:
                values.add(value)
    return frozenset(values)


def extract_thresholds(title: str) -> frozenset[str]:
    normalized = normalize_text(title)
    if not _extract_pattern_names(normalized, DIRECTION_PATTERNS) and not any(
        marker in normalized for marker in ("%", "price", "rate", "score", "temperature")
    ):
        return frozenset()
    values = set()
    years = extract_years(normalized)
    for number, unit in re.findall(r"\b(\d+(?:\.\d+)?)\s*(%|percent|bps|points?|degrees?|usd|dollars?)?\b", normalized):
        if number in years:
            continue
        canonical_number = str(int(float(number))) if float(number).is_integer() else number.rstrip("0").rstrip(".")
        values.add(f"{canonical_number}:{unit or 'number'}")
    return frozenset(values)


def _strip_fixture_context(title: str) -> str:
    value = str(title or "").strip()
    if ":" in value and re.search(r"\b(?:vs\.?|v\.?|at|@)\b", value.split(":", 1)[1], re.I):
        value = value.split(":", 1)[1]
    value = re.split(
        r"\s+-\s+(?:more markets|half[ -]?time|exact score|first half|1st half|team props?|spread|total).*?$",
        value,
        maxsplit=1,
        flags=re.I,
    )[0]
    return value.strip()


def extract_participants(title: str) -> frozenset[str]:
    value = _strip_fixture_context(title)
    match = re.search(r"^(.*?)\s+(?:vs\.?|v\.?|at|@)\s+(.*?)$", value, re.I)
    if not match:
        return frozenset()
    participants = {canonical_phrase(match.group(1)), canonical_phrase(match.group(2))}
    participants.discard("")
    return frozenset(participants) if len(participants) == 2 else frozenset()


PARTICIPANT_NOISE_TOKENS = {
    "ac", "afc", "as", "ca", "cf", "club", "de", "fc", "hc", "ogc", "rb", "rc", "rcd", "sc", "w",
}


def participant_names_compatible(left: str, right: str) -> bool:
    if left == right:
        return True
    aliases = {"st": "state"}
    left_tokens = {aliases.get(token, token) for token in left.split() if token not in PARTICIPANT_NOISE_TOKENS}
    right_tokens = {aliases.get(token, token) for token in right.split() if token not in PARTICIPANT_NOISE_TOKENS}
    if not left_tokens or not right_tokens:
        return False
    common = left_tokens.intersection(right_tokens)
    if not common:
        return any(
            len(left_token) >= 3
            and len(right_token) >= 3
            and (left_token.startswith(right_token) or right_token.startswith(left_token))
            for left_token in left_tokens
            for right_token in right_tokens
        )
    if left_tokens.issubset(right_tokens) or right_tokens.issubset(left_tokens):
        return True
    return len(common) >= 2 and len(common) / min(len(left_tokens), len(right_tokens)) >= 0.66


def participant_sets_compatible(left: frozenset[str], right: frozenset[str]) -> bool:
    if not left or not right or len(left) != len(right):
        return False
    right_values = tuple(sorted(right))
    if len(left) == 2:
        first, second = tuple(sorted(left))
        return (
            participant_names_compatible(first, right_values[0])
            and participant_names_compatible(second, right_values[1])
        ) or (
            participant_names_compatible(first, right_values[1])
            and participant_names_compatible(second, right_values[0])
        )
    return all(any(participant_names_compatible(value, other) for other in right) for value in left)


def _entity_anchors(title: str, keywords: Iterable[str]) -> frozenset[str]:
    entities = {canonical_phrase(value) for value in find_domain_entity_tokens(title)}
    for keyword in keywords:
        value = canonical_phrase(keyword)
        value_tokens = set(value.split())
        if (
            value
            and not value.startswith("year ")
            and value not in GENERIC_ENTITY_TERMS
            and not (value_tokens and value_tokens.issubset(GENERIC_ENTITY_TERMS))
            and (" " in value or any(char.isdigit() for char in value))
        ):
            entities.add(value)
    entities.discard("")
    return frozenset(entities)


def _stable_key(parts: Iterable[str]) -> str:
    payload = "|".join(sorted(part for part in parts if part))
    if not payload:
        return ""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


@dataclass(frozen=True)
class EventProfile:
    event_id: str
    platform: str
    title: str
    context_title: str
    child_title_count: int
    domain: str
    years: frozenset[str]
    seasons: frozenset[str]
    dates: frozenset[str]
    periods: frozenset[str]
    groups: frozenset[str]
    stages: frozenset[str]
    districts: frozenset[str]
    locations: frozenset[str]
    participants: frozenset[str]
    entities: frozenset[str]
    scopes: frozenset[str]
    market_layers: frozenset[str]
    actions: frozenset[str]
    directions: frozenset[str]
    thresholds: frozenset[str]
    occurrence_key: str
    proposition_key: str
    schema_version: str = PROFILE_SCHEMA_VERSION

    @property
    def explicit_fields(self) -> frozenset[str]:
        fields = (
            "years", "seasons", "dates", "periods", "groups", "stages", "districts", "locations",
            "participants", "entities", "scopes", "market_layers", "actions",
            "directions", "thresholds",
        )
        return frozenset(name for name in fields if getattr(self, name))

    def to_dict(self) -> dict[str, object]:
        result = {
            name: sorted(value) if isinstance(value, frozenset) else value
            for name, value in self.__dict__.items()
        }
        result["explicit_fields"] = sorted(self.explicit_fields)
        return result


def build_event_profile(event_id: str, data: dict[str, object]) -> EventProfile:
    title = str(data.get("title", "") or "")
    bet_titles = sorted({str(value).strip() for value in data.get("bet_titles", []) or [] if str(value).strip()})
    context_title = bet_titles[0] if len(bet_titles) == 1 else ""
    semantic_text = f"{title} {context_title}".strip()
    normalized = normalize_text(semantic_text)
    category = canonicalize_keyword(data.get("category", ""))
    sub_category = canonicalize_keyword(data.get("sub_category", ""))
    domain = canonical_phrase(f"{category} {sub_category}")
    keywords = data.get("keywords", []) or []

    years = extract_years(semantic_text)
    seasons = extract_seasons(semantic_text)
    dates = extract_dates(semantic_text)
    periods = extract_periods(semantic_text)
    groups = extract_group_ids(semantic_text)
    stages = _extract_pattern_names(normalized, STAGE_PATTERNS)
    districts = extract_districts(semantic_text)
    locations = extract_locations(semantic_text)
    sports_context = category == "sports" or any(
        term in sub_category
        for term in ("baseball", "basketball", "football", "hockey", "soccer", "tennis")
    )
    participants = extract_participants(title) if sports_context else frozenset()
    if sports_context and not participants:
        participants = extract_participants(context_title)
    entities = _entity_anchors(semantic_text, keywords)
    scopes = _extract_pattern_names(normalized, SCOPE_PATTERNS)
    market_layers = _extract_pattern_names(normalized, MARKET_LAYER_PATTERNS)
    actions = _extract_pattern_names(normalized, ACTION_PATTERNS)
    scopes = set(scopes)
    if participants:
        scopes.add("match")
    if districts or "seat" in scopes:
        scopes.discard("aggregate")
    if periods:
        scopes.discard("annual_window")
    if groups:
        scopes.add("group")
    if (
        "win" in actions
        and "tournament" in scopes
        and not groups
        and not participants
        and "award" not in scopes
    ):
        scopes.add("overall")
    scopes = frozenset(scopes)
    directions = _extract_pattern_names(normalized, DIRECTION_PATTERNS)
    thresholds = extract_thresholds(semantic_text)

    occurrence_parts = {
        f"domain:{domain}",
        *(f"participant:{value}" for value in participants),
        *(f"entity:{value}" for value in entities),
        *(f"district:{value}" for value in districts),
        *(f"year:{value}" for value in years),
        *(f"season:{value}" for value in seasons),
        *(f"date:{value}" for value in dates),
        *(f"period:{value}" for value in periods),
        *(f"group:{value}" for value in groups),
        *(f"stage:{value}" for value in stages),
        *(f"location:{value}" for value in locations),
    }
    occurrence_key = _stable_key(occurrence_parts)
    proposition_key = _stable_key({
        f"occurrence:{occurrence_key}",
        *(f"scope:{value}" for value in scopes),
        *(f"layer:{value}" for value in market_layers),
        *(f"action:{value}" for value in actions),
        *(f"direction:{value}" for value in directions),
        *(f"threshold:{value}" for value in thresholds),
    })

    return EventProfile(
        event_id=str(event_id),
        platform=str(data.get("platform", "") or ""),
        title=title,
        context_title=context_title,
        child_title_count=len(bet_titles),
        domain=domain,
        years=years,
        seasons=seasons,
        dates=dates,
        periods=periods,
        groups=groups,
        stages=stages,
        districts=districts,
        locations=locations,
        participants=participants,
        entities=entities,
        scopes=scopes,
        market_layers=market_layers,
        actions=actions,
        directions=directions,
        thresholds=thresholds,
        occurrence_key=occurrence_key,
        proposition_key=proposition_key,
    )


def build_event_profiles(index: dict[str, dict[str, object]]) -> dict[str, EventProfile]:
    return {event_id: build_event_profile(event_id, data) for event_id, data in index.items()}


def explicit_conflicts(left: EventProfile, right: EventProfile) -> tuple[str, ...]:
    conflicts = []
    strict_fields = (
        "groups", "districts", "locations", "seasons", "directions", "thresholds",
    )
    for field in strict_fields:
        left_values = getattr(left, field)
        right_values = getattr(right, field)
        if left_values and right_values and left_values.isdisjoint(right_values):
            conflicts.append(field)

    if left.participants and right.participants and not participant_sets_compatible(left.participants, right.participants):
        conflicts.append("participants")

    def temporal_values_compatible(left_values: frozenset[str], right_values: frozenset[str]) -> bool:
        for left_value in left_values:
            for right_value in right_values:
                left_parts = left_value.split("-")
                right_parts = right_value.split("-")
                if len(left_parts) != len(right_parts):
                    continue
                if all(a == b or a == "????" or b == "????" for a, b in zip(left_parts, right_parts)):
                    return True
        return False

    for field in ("dates", "periods"):
        left_values = getattr(left, field)
        right_values = getattr(right, field)
        if left_values and right_values and not temporal_values_compatible(left_values, right_values):
            conflicts.append(field)

    mutually_exclusive_scopes = (
        {"primary", "general_election"},
        {"match", "season"},
        {"match", "tournament"},
        {"district", "aggregate"},
        {"seat", "aggregate"},
        {"group", "overall"},
        {"award", "overall"},
        {"award", "metric"},
        {"award", "match"},
        {"award", "season"},
        {"meeting", "annual_end"},
        {"meeting", "annual_window"},
    )
    for dimensions in mutually_exclusive_scopes:
        left_values = left.scopes.intersection(dimensions)
        right_values = right.scopes.intersection(dimensions)
        if left_values and right_values and left_values.isdisjoint(right_values):
            if dimensions == {"meeting", "annual_end"}:
                meeting_profile = left if "meeting" in left_values else right
                annual_profile = right if meeting_profile is left else left
                december_meeting = any(period.endswith("-12") for period in meeting_profile.periods)
                compatible_year = (
                    not meeting_profile.years
                    or not annual_profile.years
                    or bool(meeting_profile.years.intersection(annual_profile.years))
                )
                if december_meeting and compatible_year:
                    continue
            conflicts.append("scopes")
            break
    return tuple(sorted(set(conflicts)))


def one_sided_fields(left: EventProfile, right: EventProfile) -> tuple[str, ...]:
    fields = (
        "years", "seasons", "dates", "periods", "groups", "stages", "districts", "locations", "participants",
        "scopes", "market_layers", "actions", "directions", "thresholds",
    )
    return tuple(sorted(
        field for field in fields if bool(getattr(left, field)) != bool(getattr(right, field))
    ))


def profiles_fingerprint(profiles: dict[str, EventProfile]) -> str:
    payload = json.dumps(
        {event_id: profile.to_dict() for event_id, profile in sorted(profiles.items())},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
