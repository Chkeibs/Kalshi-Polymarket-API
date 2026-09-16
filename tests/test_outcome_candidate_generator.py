import os
import sys

import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pipeline.clustering.outcome_candidate_generator import (  # noqa: E402
    build_outcome_bundles,
    build_outcome_profiles,
    extract_deadline_date,
    extract_event_years,
    generate_outcome_candidates,
    market_intent_compatible,
    strip_polymarket_outcome_suffix,
    subject_key,
)


def kalshi_rows():
    return pd.DataFrame(
        [
            {
                "Platform": "Kalshi",
                "Event ID": "KXMENWORLDCUP-26",
                "Bet ID": "KXMENWORLDCUP-26-ES",
                "Market Title": "2026 Men's World Cup winner?",
                "Bet Title": "Spain",
                "Category": "Sports",
                "Sub-Category": "Soccer",
                "Description": "If Spain wins the 2026 Men's World Cup, then this resolves Yes.",
                "Bet_Rules": "If Spain wins the 2026 Men's World Cup, then this resolves Yes.",
            },
            {
                "Platform": "Kalshi",
                "Event ID": "KXMENWORLDCUP-26",
                "Bet ID": "KXMENWORLDCUP-26-FR",
                "Market Title": "2026 Men's World Cup winner?",
                "Bet Title": "France",
                "Category": "Sports",
                "Sub-Category": "Soccer",
                "Description": "If France wins the 2026 Men's World Cup, then this resolves Yes.",
                "Bet_Rules": "If France wins the 2026 Men's World Cup, then this resolves Yes.",
            },
            {
                "Platform": "Kalshi",
                "Event ID": "KXWCGROUPWIN-26A",
                "Bet ID": "KXWCGROUPWIN-26A-MEX",
                "Market Title": "World Cup Group A Winner",
                "Bet Title": "Mexico",
                "Category": "Sports",
                "Sub-Category": "Soccer",
                "Description": "If Mexico finish first in Group A, this resolves Yes.",
                "Bet_Rules": "If Mexico finish first in Group A, this resolves Yes.",
            },
            {
                "Platform": "Kalshi",
                "Event ID": "KXSPAINWINWC-26",
                "Bet ID": "KXSPAINWINWC-26",
                "Market Title": "Will Spain win the 2026 FIFA World Cup?",
                "Bet Title": "Yes",
                "Category": "Sports",
                "Sub-Category": "Soccer",
                "Description": "If Spain wins the 2026 FIFA World Cup, this resolves Yes.",
                "Bet_Rules": "If Spain wins the 2026 FIFA World Cup, this resolves Yes.",
            },
            {
                "Platform": "Kalshi",
                "Event ID": "KXDEMOPRES-28",
                "Bet ID": "KXDEMOPRES-28-DEM",
                "Market Title": "Which party will win the 2028 presidential election?",
                "Bet Title": "Democratic",
                "Category": "Politics",
                "Sub-Category": "Elections",
                "Description": "If the Democratic Party wins the 2028 presidential election, this resolves Yes.",
                "Bet_Rules": "If the Democratic Party wins the 2028 presidential election, this resolves Yes.",
            },
            {
                "Platform": "Kalshi",
                "Event ID": "KXHOUSESEATS-26",
                "Bet ID": "KXHOUSESEATS-26-10",
                "Market Title": "How many House seats will Republicans win in 2026?",
                "Bet Title": "10",
                "Category": "Politics",
                "Sub-Category": "Elections",
                "Description": "If Republicans win exactly 10 House seats, this resolves Yes.",
                "Bet_Rules": "If Republicans win exactly 10 House seats, this resolves Yes.",
            },
        ]
    )


def polymarket_rows():
    return pd.DataFrame(
        [
            {
                "Platform": "Polymarket",
                "Event ID": "30615",
                "Bet ID": "PM-WC-SPAIN-YES",
                "Original_Market_ID": "558934",
                "Market Title": "2026 FIFA World Cup Winner",
                "Bet Title": "Spain [Yes]",
                "Outcome_Name": "Yes",
                "Category": "Sports",
                "Sub-Category": "Soccer, FIFA World Cup",
                "Description": "This market resolves Yes if Spain wins the 2026 FIFA World Cup.",
                "Bet_Rules": "Spain wins the 2026 FIFA World Cup.",
            },
            {
                "Platform": "Polymarket",
                "Event ID": "30615",
                "Bet ID": "PM-WC-FRANCE-YES",
                "Original_Market_ID": "558936",
                "Market Title": "2026 FIFA World Cup Winner",
                "Bet Title": "France [Yes]",
                "Outcome_Name": "Yes",
                "Category": "Sports",
                "Sub-Category": "Soccer, FIFA World Cup",
                "Description": "This market resolves Yes if France wins the 2026 FIFA World Cup.",
                "Bet_Rules": "France wins the 2026 FIFA World Cup.",
            },
            {
                "Platform": "Polymarket",
                "Event ID": "26313",
                "Bet ID": "PM-WC-QUALIFY-SPAIN-YES",
                "Original_Market_ID": "550999",
                "Market Title": "2026 FIFA World Cup: Which countries qualify?",
                "Bet Title": "Spain [Yes]",
                "Outcome_Name": "Yes",
                "Category": "Sports",
                "Sub-Category": "Soccer, FIFA World Cup",
                "Description": "This market resolves Yes if Spain qualifies.",
                "Bet_Rules": "Spain qualifies for the 2026 FIFA World Cup.",
            },
            {
                "Platform": "Polymarket",
                "Event ID": "98252",
                "Bet ID": "PM-GROUP-A-MEXICO-YES",
                "Original_Market_ID": "839357",
                "Market Title": "FIFA World Cup Group A Winner",
                "Bet Title": "Mexico [Yes]",
                "Outcome_Name": "Yes",
                "Category": "Sports",
                "Sub-Category": "Soccer, FIFA World Cup",
                "Description": "Mexico wins Group A.",
                "Bet_Rules": "Mexico wins Group A.",
            },
            {
                "Platform": "Polymarket",
                "Event ID": "98252",
                "Bet ID": "PM-GROUP-A-SOUTH-KOREA-YES",
                "Original_Market_ID": "839359",
                "Market Title": "FIFA World Cup Group A Winner",
                "Bet Title": "South Korea [Yes]",
                "Outcome_Name": "Yes",
                "Category": "Sports",
                "Sub-Category": "Soccer, FIFA World Cup",
                "Description": "South Korea wins Group A.",
                "Bet_Rules": "South Korea wins Group A.",
            },
            {
                "Platform": "Polymarket",
                "Event ID": "30615",
                "Bet ID": "PM-WC-SPAIN-NO",
                "Original_Market_ID": "558934",
                "Market Title": "2026 FIFA World Cup Winner",
                "Bet Title": "Spain [No]",
                "Outcome_Name": "No",
                "Category": "Sports",
                "Sub-Category": "Soccer, FIFA World Cup",
                "Description": "This market resolves No if Spain does not win.",
                "Bet_Rules": "Spain does not win the 2026 FIFA World Cup.",
            },
            {
                "Platform": "Polymarket",
                "Event ID": "PMCA45",
                "Bet ID": "PM-CA45-DEM-YES",
                "Original_Market_ID": "770001",
                "Market Title": "California 45th District House Election Winner",
                "Bet Title": "Democratic [Yes]",
                "Outcome_Name": "Yes",
                "Category": "Politics",
                "Sub-Category": "Elections, US Elections",
                "Description": "This market resolves Yes if the Democratic candidate wins CA-45.",
                "Bet_Rules": "Democratic candidate wins CA-45.",
            },
            {
                "Platform": "Polymarket",
                "Event ID": "PMWAYMO",
                "Bet ID": "PM-WAYMO-10-YES",
                "Original_Market_ID": "770002",
                "Market Title": "How many cities will Waymo operate in by end of 2026?",
                "Bet Title": "10 [Yes]",
                "Outcome_Name": "Yes",
                "Category": "Science",
                "Sub-Category": "AI, Waymo",
                "Description": "This market resolves Yes if Waymo operates in exactly 10 cities.",
                "Bet_Rules": "Waymo operates in exactly 10 cities.",
            },
        ]
    )


def pair_ids(candidates):
    return {
        (
            c["kalshi"]["bet_id"],
            c["polymarket"]["event_id"],
            c["polymarket"]["bet_id"],
        )
        for c in candidates
    }


def make_profile(platform, event_id, market_title, bet_title, category, sub_category):
    row = {
        "Event ID": event_id,
        "Bet ID": f"{event_id}-BET",
        "Original_Market_ID": f"{event_id}-MARKET",
        "Market Title": market_title,
        "Bet Title": bet_title,
        "Outcome_Name": "Yes",
        "Category": category,
        "Sub-Category": sub_category,
    }
    return build_outcome_profiles(pd.DataFrame([row]), platform)[0]


def test_strip_polymarket_outcome_suffix():
    assert strip_polymarket_outcome_suffix("Spain [Yes]") == ("Spain", "Yes")
    assert strip_polymarket_outcome_suffix("France [No]") == ("France", "No")
    assert strip_polymarket_outcome_suffix("Mexico") == ("Mexico", "")


def test_subject_alias_normalization():
    assert subject_key("USA") == "united states"
    assert subject_key("Korea Republic") == "south korea"


def test_deadline_extraction_ignores_fallback_resolution_date():
    rules = (
        "The ceremony will be held on March 15, 2026. "
        "If, for any reason, no winner is declared by June 30, 2026, the market resolves to Other."
    )

    assert extract_deadline_date(rules) == "2026-03-15"
    assert extract_deadline_date("MetaMask launches before Jan 1, 2027.") == "2027-01-01"
    assert extract_deadline_date("December 31, 2026 [Yes]") == "2026-12-31"
    assert extract_deadline_date(
        'If the pick is not definitively known by July 30, 2026, this market will resolve to "Other".'
    ) == ""
    assert extract_event_years("Election winner in 2028") == frozenset({"2028"})
    assert extract_event_years("before 2027") == frozenset()


def test_generates_world_cup_winner_country_candidates():
    candidates, stats = generate_outcome_candidates(kalshi_rows(), polymarket_rows())
    ids = pair_ids(candidates)

    assert ("KXMENWORLDCUP-26-ES", "30615", "PM-WC-SPAIN-YES") in ids
    assert ("KXMENWORLDCUP-26-FR", "30615", "PM-WC-FRANCE-YES") in ids
    assert ("KXSPAINWINWC-26", "30615", "PM-WC-SPAIN-YES") in ids
    assert stats["candidate_pairs"] >= 3


def test_generates_group_winner_candidate():
    candidates, _ = generate_outcome_candidates(kalshi_rows(), polymarket_rows())
    ids = pair_ids(candidates)

    assert ("KXWCGROUPWIN-26A-MEX", "98252", "PM-GROUP-A-MEXICO-YES") in ids


def test_rejects_different_scope_and_no_tokens():
    candidates, _ = generate_outcome_candidates(kalshi_rows(), polymarket_rows())
    ids = pair_ids(candidates)

    assert ("KXMENWORLDCUP-26-ES", "26313", "PM-WC-QUALIFY-SPAIN-YES") not in ids
    assert ("KXMENWORLDCUP-26-ES", "30615", "PM-WC-FRANCE-YES") not in ids
    assert ("KXMENWORLDCUP-26-ES", "30615", "PM-WC-SPAIN-NO") not in ids


def test_rejects_same_outcome_with_incompatible_political_scope():
    candidates, _ = generate_outcome_candidates(kalshi_rows(), polymarket_rows())
    ids = pair_ids(candidates)

    assert ("KXDEMOPRES-28-DEM", "PMCA45", "PM-CA45-DEM-YES") not in ids


def test_rejects_same_numeric_outcome_with_incompatible_category():
    candidates, _ = generate_outcome_candidates(kalshi_rows(), polymarket_rows())
    ids = pair_ids(candidates)

    assert ("KXHOUSESEATS-26-10", "PMWAYMO", "PM-WAYMO-10-YES") not in ids


def test_binary_polymarket_title_supplies_subject_when_bet_is_a_date():
    kalshi = pd.DataFrame(
        [
            {
                "Event ID": "KXTOKEN",
                "Bet ID": "KXTOKEN-META",
                "Market Title": "Who will launch a token this year?",
                "Bet Title": "MetaMask",
                "Category": "Crypto",
                "Sub-Category": "Pre-Market",
                "Bet_Rules": "MetaMask launches a token before Jan 1, 2027.",
            },
            {
                "Event ID": "KSTAKE",
                "Bet ID": "KSTAKE-META",
                "Market Title": "Which companies will the US government take a stake in before 2027?",
                "Bet Title": "MetaMask",
                "Category": "Economics",
                "Sub-Category": "Business",
            },
            {
                "Event ID": "KSTAKE",
                "Bet ID": "KSTAKE-OPENAI",
                "Market Title": "Which companies will the US government take a stake in before 2027?",
                "Bet Title": "OpenAI",
                "Category": "Economics",
                "Sub-Category": "Business",
            },
            {
                "Event ID": "KXTOKEN",
                "Bet ID": "KXTOKEN-BASE",
                "Market Title": "Who will launch a token this year?",
                "Bet Title": "Base",
                "Category": "Crypto",
                "Sub-Category": "Pre-Market",
                "Bet_Rules": "Base launches a token before Jan 1, 2027.",
            },
        ]
    )
    polymarket = pd.DataFrame(
        [
            {
                "Event ID": "PMMETA",
                "Bet ID": "PMMETA-DEC",
                "Original_Market_ID": "META-DEC",
                "Market Title": "Will MetaMask launch a token by ___?",
                "Bet Title": "December 31, 2026 [Yes]",
                "Outcome_Name": "Yes",
                "Category": "Crypto",
                "Sub-Category": "Pre-Market, token launch",
                "Bet_Rules": "Will MetaMask launch a token by December 31, 2026?",
            },
            {
                "Event ID": "PMMETA",
                "Bet ID": "PMMETA-DEC-NO",
                "Original_Market_ID": "META-DEC",
                "Market Title": "Will MetaMask launch a token by ___?",
                "Bet Title": "December 31, 2026 [No]",
                "Outcome_Name": "No",
                "Category": "Crypto",
                "Sub-Category": "Pre-Market, token launch",
                "Bet_Rules": "Will MetaMask launch a token by December 31, 2026?",
            },
            {
                "Event ID": "PMMETA",
                "Bet ID": "PMMETA-2027",
                "Original_Market_ID": "META-2027",
                "Market Title": "Will MetaMask launch a token by ___?",
                "Bet Title": "December 31, 2027 [Yes]",
                "Outcome_Name": "Yes",
                "Category": "Crypto",
                "Sub-Category": "Pre-Market, token launch",
                "Bet_Rules": "Will MetaMask launch a token by December 31, 2027?",
            },
        ]
    )

    candidates, _ = generate_outcome_candidates(kalshi, polymarket)
    ids = pair_ids(candidates)

    assert ("KXTOKEN-META", "PMMETA", "PMMETA-DEC") in ids
    assert ("KXTOKEN-META", "PMMETA", "PMMETA-2027") not in ids
    assert ("KSTAKE-META", "PMMETA", "PMMETA-DEC") not in ids
    candidate = next(c for c in candidates if c["kalshi"]["bet_id"] == "KXTOKEN-META")
    assert candidate["candidate_type"] == "event_outcome"
    assert candidate["polymarket"]["outcome_subject"] == "MetaMask"


def test_rejects_same_subject_when_event_intents_conflict():
    kalshi = pd.DataFrame(
        [
            {"Event ID": "KARREST", "Bet ID": "KARREST-GAVIN", "Market Title": "Who will be arrested before 2027?", "Bet Title": "Gavin Newsom", "Category": "Politics", "Sub-Category": "World"},
            {"Event ID": "KARREST", "Bet ID": "KARREST-BARACK", "Market Title": "Who will be arrested before 2027?", "Bet Title": "Barack Obama", "Category": "Politics", "Sub-Category": "World"},
        ]
    )
    polymarket = pd.DataFrame(
        [
            {"Event ID": "PRUN", "Bet ID": "PRUN-GAVIN", "Original_Market_ID": "RUN-GAVIN", "Market Title": "Who will announce Presidential run before 2027?", "Bet Title": "Gavin Newsom [Yes]", "Outcome_Name": "Yes", "Category": "Politics", "Sub-Category": "World"},
            {"Event ID": "PRUN", "Bet ID": "PRUN-BARACK", "Original_Market_ID": "RUN-BARACK", "Market Title": "Who will announce Presidential run before 2027?", "Bet Title": "Barack Obama [Yes]", "Outcome_Name": "Yes", "Category": "Politics", "Sub-Category": "World"},
            {"Event ID": "PLEAVE", "Bet ID": "PLEAVE-GAVIN", "Original_Market_ID": "LEAVE-GAVIN", "Market Title": "Who will leave the administration before 2027?", "Bet Title": "Gavin Newsom [Yes]", "Outcome_Name": "Yes", "Category": "Politics", "Sub-Category": "World"},
            {"Event ID": "PLEAVE", "Bet ID": "PLEAVE-BARACK", "Original_Market_ID": "LEAVE-BARACK", "Market Title": "Who will leave the administration before 2027?", "Bet Title": "Barack Obama [Yes]", "Outcome_Name": "Yes", "Category": "Politics", "Sub-Category": "World"},
        ]
    )

    candidates, _ = generate_outcome_candidates(kalshi, polymarket)

    assert not candidates


def test_rejects_song_release_but_keeps_album_release_for_same_artist():
    kalshi = pd.DataFrame(
        [
            {"Event ID": "KALBUM", "Bet ID": "KALBUM-DRAKE", "Market Title": "Who will release a new album in 2026?", "Bet Title": "Drake", "Category": "Culture", "Sub-Category": "Music"},
            {"Event ID": "KALBUM", "Bet ID": "KALBUM-RIHANNA", "Market Title": "Who will release a new album in 2026?", "Bet Title": "Rihanna", "Category": "Culture", "Sub-Category": "Music"},
        ]
    )
    polymarket = pd.DataFrame(
        [
            {"Event ID": "PALBUM", "Bet ID": "PALBUM-DRAKE", "Original_Market_ID": "ALBUM-DRAKE", "Market Title": "Which artists will release new albums in 2026?", "Bet Title": "Drake [Yes]", "Outcome_Name": "Yes", "Category": "Culture", "Sub-Category": "Music"},
            {"Event ID": "PALBUM", "Bet ID": "PALBUM-RIHANNA", "Original_Market_ID": "ALBUM-RIHANNA", "Market Title": "Which artists will release new albums in 2026?", "Bet Title": "Rihanna [Yes]", "Outcome_Name": "Yes", "Category": "Culture", "Sub-Category": "Music"},
            {"Event ID": "PSONG", "Bet ID": "PSONG-DRAKE", "Original_Market_ID": "SONG-DRAKE", "Market Title": "Which artists will release a new song in 2026?", "Bet Title": "Drake [Yes]", "Outcome_Name": "Yes", "Category": "Culture", "Sub-Category": "Music"},
            {"Event ID": "PSONG", "Bet ID": "PSONG-RIHANNA", "Original_Market_ID": "SONG-RIHANNA", "Market Title": "Which artists will release a new song in 2026?", "Bet Title": "Rihanna [Yes]", "Outcome_Name": "Yes", "Category": "Culture", "Sub-Category": "Music"},
            {"Event ID": "PCHART", "Bet ID": "PCHART-DRAKE", "Original_Market_ID": "CHART-DRAKE", "Market Title": "Which artists will have a Billboard #1 song this year?", "Bet Title": "Drake [Yes]", "Outcome_Name": "Yes", "Category": "Culture", "Sub-Category": "Music"},
            {"Event ID": "PCHART", "Bet ID": "PCHART-RIHANNA", "Original_Market_ID": "CHART-RIHANNA", "Market Title": "Which artists will have a Billboard #1 song this year?", "Bet Title": "Rihanna [Yes]", "Outcome_Name": "Yes", "Category": "Culture", "Sub-Category": "Music"},
        ]
    )

    candidates, _ = generate_outcome_candidates(kalshi, polymarket)
    ids = pair_ids(candidates)

    assert ("KALBUM-DRAKE", "PALBUM", "PALBUM-DRAKE") in ids
    assert ("KALBUM-DRAKE", "PSONG", "PSONG-DRAKE") not in ids
    assert ("KALBUM-DRAKE", "PCHART", "PCHART-DRAKE") not in ids


def test_scope_guards_reject_sport_result_rank_and_period_mismatches():
    football = make_profile("kalshi", "KFOOT", "College Football Undefeated Season", "Alabama", "Sports", "Football")
    basketball = make_profile("polymarket", "PBASK", "SEC Men's College Basketball Regular Season Champion", "Alabama [Yes]", "Sports", "Basketball")
    champion = make_profile("kalshi", "KEUROPA", "Europa League Champion", "Lyon", "Sports", "Soccer")
    scorer = make_profile("polymarket", "PSCORER", "UEFA Europa League: Top Scorer (Club)", "Lyon [Yes]", "Sports", "Soccer")
    winner = make_profile("kalshi", "KEUROVISION", "Eurovision Winner 2026?", "Belgium", "Culture", "Music")
    semifinal = make_profile("polymarket", "PSEMI", "Eurovision 2026: First Semi-Final", "Belgium [Yes]", "Culture", "Music")
    annual_rank = make_profile("kalshi", "KSPOTIFY", "#2 Artist on Spotify in 2026?", "Drake", "Culture", "Music")
    monthly_rank = make_profile("polymarket", "PSPOTIFY", "#2 Spotify artist in March?", "Drake [Yes]", "Culture", "Music")
    matching_monthly_rank = make_profile("kalshi", "KSPOTIFYMAR", "#2 Spotify artist in March?", "Drake", "Culture", "Music")
    league_winner = make_profile("kalshi", "KLEAGUE", "English Premier League Winner?", "Liverpool", "Sports", "Soccer")
    second_place = make_profile("polymarket", "PSECOND", "English Premier League - 2nd Place", "Liverpool [Yes]", "Sports", "Soccer")
    first_place = make_profile("polymarket", "PFIRST", "English Premier League - 1st Place", "Liverpool [Yes]", "Sports", "Soccer")
    poll_ranking = make_profile("kalshi", "KPOLL", "#1 ranked team on Men's College Basketball AP Poll", "Duke", "Sports", "Basketball")
    season_champion = make_profile("polymarket", "PSEASON", "ACC Men's College Basketball Regular Season Champion", "Duke [Yes]", "Sports", "Basketball")
    wbc_long = make_profile("kalshi", "KWBC", "World Baseball Classic Winner?", "Brazil", "Sports", "Baseball")
    wbc_short = make_profile("polymarket", "PWBC", "WBC Winner 2026", "Brazil [Yes]", "Sports", "Baseball")
    nominee = make_profile("kalshi", "KNOM", "2028 Republican nominee for President?", "Nikki Haley", "Politics", "Elections")
    election_winner = make_profile("polymarket", "PWIN", "Presidential Election Winner 2028", "Nikki Haley [Yes]", "Politics", "Elections")

    assert not market_intent_compatible(football, basketball)
    assert not market_intent_compatible(champion, scorer)
    assert not market_intent_compatible(winner, semifinal)
    assert not market_intent_compatible(annual_rank, monthly_rank)
    assert market_intent_compatible(matching_monthly_rank, monthly_rank)
    assert not market_intent_compatible(league_winner, second_place)
    assert market_intent_compatible(league_winner, first_place)
    assert not market_intent_compatible(poll_ranking, season_champion)
    assert market_intent_compatible(wbc_long, wbc_short)
    assert not market_intent_compatible(nominee, election_winner)


def test_scope_guards_compare_explicit_actors_and_company_actions():
    pope_visit = make_profile("kalshi", "KPOPE", "What countries will Pope Leo visit before 2027?", "Mexico", "Politics", "World")
    trump_visit = make_profile("polymarket", "PTRUMP", "Which countries will Donald Trump visit before 2027?", "Mexico [Yes]", "Politics", "World")
    matching_trump_visit = make_profile("kalshi", "KTRUMP", "What countries will Trump visit before 2027?", "Mexico", "Politics", "World")
    ipo = make_profile("kalshi", "KIPO", "Which companies will officially announce an IPO this year?", "OpenAI", "Economics", "Business")
    bankruptcy = make_profile("polymarket", "PBANK", "Which companies announce bankruptcy before 2027?", "OpenAI [Yes]", "Economics", "Business")
    ipos = make_profile("polymarket", "PIPO", "IPOs before 2027?", "OpenAI [Yes]", "Economics", "Business")
    government_stake = make_profile("kalshi", "KSTAKE", "Which companies will the US take a stake in before 2027?", "OpenAI", "Economics", "Business")
    acquisition = make_profile("polymarket", "PACQUIRE", "Which companies will be acquired before 2027?", "OpenAI [Yes]", "Economics", "Business")
    scripted_series = make_profile("kalshi", "KSERIES", "Which companies will release a fully AI-generated multi-episode scripted series before 2027?", "Apple", "Culture", "Big Tech")
    product_line = make_profile("polymarket", "PPRODUCT", "Will Apple release a new product line before 2027?", "Apple [Yes]", "Culture", "Big Tech")

    assert not market_intent_compatible(pope_visit, trump_visit)
    assert market_intent_compatible(matching_trump_visit, trump_visit)
    assert not market_intent_compatible(ipo, bankruptcy)
    assert not market_intent_compatible(government_stake, ipos)
    assert not market_intent_compatible(government_stake, acquisition)
    assert not market_intent_compatible(scripted_series, product_line)


def test_scope_guards_compare_years_locations_metrics_roles_and_awards():
    state_senate = make_profile("kalshi", "KSENATE", "Indiana Senate winner? (2028)", "Democratic Party", "Politics", "Elections")
    senate_control = make_profile("polymarket", "PCONTROL", "Which party will win the Senate in 2026?", "Democratic Party [Yes]", "Politics", "Elections")
    same_year_state_senate = make_profile("kalshi", "KSENATE26", "Virginia Senate winner? (2026)", "Democratic Party", "Politics", "Elections")
    italian_election = make_profile("kalshi", "KITALY", "Who will win the next Italian election?", "Democratic Party", "Politics", "Elections")
    california_house = make_profile("polymarket", "PCA", "CA-22 House Election Winner", "Democratic Party [Yes]", "Politics", "Elections")
    unemployment = make_profile("kalshi", "KUNEMP", "How high will unemployment get before 2030?", "Above 5%", "Economics", "Macro")
    inflation = make_profile("polymarket", "PINFL", "How high will inflation get in 2026?", "Above 5% [Yes]", "Economics", "Macro")
    bond_q = make_profile("kalshi", "KBONDQ", "Who will play Q in the next James Bond?", "Tom Holland", "Culture", "Movies")
    next_bond = make_profile("polymarket", "PBOND", "Next James Bond actor?", "Tom Holland [Yes]", "Culture", "Movies")
    rebounds = make_profile("kalshi", "KREB", "Pro Basketball Rebounds Per Game Leader", "Victor Wembanyama", "Sports", "Basketball")
    points = make_profile("polymarket", "PPTS", "NBA Points Per Game Leader", "Victor Wembanyama [Yes]", "Sports", "Basketball")
    art_ross = make_profile("kalshi", "KROSS", "NHL Art Ross Trophy", "Cale Makar", "Sports", "Hockey")
    hart = make_profile("polymarket", "PHART", "NHL Hart Memorial Trophy Winner", "Cale Makar [Yes]", "Sports", "Hockey")
    house_seats = make_profile("kalshi", "KSEATS", "How many House seats will Democrats win in California?", "44", "Politics", "Elections")
    members_retiring = make_profile("polymarket", "PRETIRE", "How many Republican House members not running in 2026?", "44 [Yes]", "Politics", "Elections")
    governor_exit = make_profile("kalshi", "KWALZOUT", "Tim Walz out as Governor of Minnesota?", "Before 2027", "Politics", "Governor")
    governor_charged = make_profile("polymarket", "KWALZCHARGE", "Tim Walz charged by...?", "Before 2027 [Yes]", "Politics", "Governor")
    ai_rank = make_profile("kalshi", "KAIRANK", "Which companies will have a top-ranked AI model this year?", "OpenAI", "Science", "AI")
    ai_threshold = make_profile("polymarket", "PAISCORE", "Which company's AI will first hit 1550 on Chatbot Arena in 2026?", "OpenAI [Yes]", "Science", "AI")
    world_cup_song = make_profile("kalshi", "KWCSONG", "Who will sing the next World Cup song?", "Rihanna", "Culture", "Music")
    billboard_song = make_profile("polymarket", "PBILLBOARD", "Which artists will have a Billboard #1 song this year?", "Rihanna [Yes]", "Culture", "Music")

    assert not market_intent_compatible(state_senate, senate_control)
    assert not market_intent_compatible(same_year_state_senate, senate_control)
    assert not market_intent_compatible(italian_election, california_house)
    assert not market_intent_compatible(unemployment, inflation)
    assert not market_intent_compatible(bond_q, next_bond)
    assert not market_intent_compatible(rebounds, points)
    assert not market_intent_compatible(art_ross, hart)
    assert not market_intent_compatible(house_seats, members_retiring)
    assert not market_intent_compatible(governor_exit, governor_charged)
    assert not market_intent_compatible(ai_rank, ai_threshold)
    assert not market_intent_compatible(world_cup_song, billboard_song)


def test_scope_guards_compare_exact_sports_metrics_and_named_awards():
    baseball_hits = make_profile("kalshi", "KHITS", "Pro Baseball Hits Leader", "Aaron Judge", "Sports", "Baseball")
    baseball_homers = make_profile("polymarket", "PHR", "MLB Home Runs Leader", "Aaron Judge [Yes]", "Sports", "Baseball")
    matching_hits = make_profile("polymarket", "PHITS", "MLB Hits Leader", "Aaron Judge [Yes]", "Sports", "Baseball")
    basketball_steals = make_profile("kalshi", "KSTL", "NBA Steals Per Game Leader", "Victor Wembanyama", "Sports", "Basketball")
    basketball_blocks = make_profile("polymarket", "PBLK", "NBA Blocks Per Game Leader", "Victor Wembanyama [Yes]", "Sports", "Basketball")
    james_norris = make_profile("kalshi", "KNORRIS", "NHL James Norris Trophy Winner", "Cale Makar", "Sports", "Hockey")
    hart = make_profile("polymarket", "PHART2", "NHL Hart Memorial Trophy Winner", "Cale Makar [Yes]", "Sports", "Hockey")
    conference_mvp = make_profile("kalshi", "KCFMVP", "NBA Conference Finals MVP", "Jayson Tatum", "Sports", "Basketball")
    season_mvp = make_profile("polymarket", "PMVP", "NBA MVP Winner", "Jayson Tatum [Yes]", "Sports", "Basketball")

    assert not market_intent_compatible(baseball_hits, baseball_homers)
    assert market_intent_compatible(baseball_hits, matching_hits)
    assert not market_intent_compatible(basketball_steals, basketball_blocks)
    assert not market_intent_compatible(james_norris, hart)
    assert not market_intent_compatible(conference_mvp, season_mvp)


def test_scope_guards_compare_competition_namespace_stage_and_rank():
    pro_hits = make_profile("kalshi", "KMLBHITS", "Pro Baseball Hits Leader", "Juan Soto", "Sports", "Baseball")
    wbc_hits = make_profile("polymarket", "PWBCHITS", "World Baseball Classic Hits Leader", "Juan Soto [Yes]", "Sports", "Baseball")
    wbc_winner = make_profile("kalshi", "KWBCWIN", "World Baseball Classic Winner", "Japan", "Sports", "Baseball")
    wbc_pool = make_profile("polymarket", "PWBCPOOL", "World Baseball Classic Pool A Winner", "Japan [Yes]", "Sports", "Baseball")
    draft_third = make_profile("kalshi", "KDRAFT3", "Pro Football Draft #3 Pick", "Arch Manning", "Sports", "Football")
    draft_second = make_profile("polymarket", "PDRAFT2", "NFL Draft 2nd Overall Pick", "Arch Manning [Yes]", "Sports", "Football")
    draft_third_text = make_profile("polymarket", "PDRAFT3", "NFL Draft third overall pick", "Arch Manning [Yes]", "Sports", "Football")
    round_of_16 = make_profile("kalshi", "KR16", "FIFA World Cup Round of 16 Qualifier", "Japan", "Sports", "Soccer")
    quarterfinal = make_profile("polymarket", "PQF", "FIFA World Cup Quarterfinal Qualifier", "Japan [Yes]", "Sports", "Soccer")

    assert not market_intent_compatible(pro_hits, wbc_hits)
    assert not market_intent_compatible(wbc_winner, wbc_pool)
    assert not market_intent_compatible(draft_third, draft_second)
    assert market_intent_compatible(draft_third, draft_third_text)
    assert not market_intent_compatible(round_of_16, quarterfinal)


def test_scope_guards_compare_leagues_levels_and_tournament_stages():
    nba_conference = make_profile("kalshi", "KNBAEAST", "Pro Basketball Eastern Conference Champion", "Charlotte", "Sports", "Basketball")
    nba_from_event_id = make_profile("kalshi", "KXNBAEAST-26", "Eastern Conference Champion?", "Charlotte", "Sports", "Basketball")
    ncaa_conference = make_profile("polymarket", "PNCAAAM", "NCAAB Conference Winner: American", "Charlotte [Yes]", "Sports", "Basketball")
    europa = make_profile("kalshi", "KEUROPA2", "Europa League Champion", "Roma", "Sports", "Soccer")
    serie_a = make_profile("polymarket", "PSERIEA", "Serie A League Winner", "Roma [Yes]", "Sports", "Soccer")
    womens_ucl = make_profile("kalshi", "KWUCL", "Women's Champions League Champion", "Arsenal", "Sports", "Soccer")
    epl = make_profile("polymarket", "PEPL", "English Premier League Winner", "Arsenal [Yes]", "Sports", "Soccer")
    wbc_winner = make_profile("kalshi", "KWBCFINAL", "World Baseball Classic Winner", "Japan", "Sports", "Baseball")
    wbc_finalist = make_profile("polymarket", "PWBCFINAL", "WBC: Team to make final", "Japan [Yes]", "Sports", "Baseball")
    wbc_knockout = make_profile("polymarket", "PWBCKO", "WBC: Team to advance to Knockout Stages", "Japan [Yes]", "Sports", "Baseball")
    ap_poll = make_profile("kalshi", "KAPPOLL", "#1 ranked team on Men's College Basketball AP Poll", "Duke", "Sports", "Basketball")
    ncaa_seed = make_profile("polymarket", "PNCAASEED", "Which teams will be a #1 seed in NCAA Tournament?", "Duke [Yes]", "Sports", "Basketball")
    nba_play_in = make_profile("kalshi", "KNBAPLAYIN", "Which teams will make the western conference play-in tournament", "Houston", "Sports", "Basketball")
    ncaa_winner = make_profile("polymarket", "PNCAAWIN", "2026 NCAA Tournament Winner", "Houston [Yes]", "Sports", "Basketball")
    wcc_regular = make_profile("kalshi", "KWCC", "WCC Regular Season Champion", "Gonzaga", "Sports", "Basketball")
    wcc_conference = make_profile("polymarket", "PWCC", "NCAAB Conference Winner: WCC", "Gonzaga [Yes]", "Sports", "Basketball")
    wcc_matching_regular = make_profile("polymarket", "PWCCREG", "WCC Regular Season Champion", "Gonzaga [Yes]", "Sports", "Basketball")
    masters_long = make_profile("kalshi", "KMASTERS", "Masters Tournament Champion", "Rory McIlroy", "Sports", "Golf")
    masters_short = make_profile("polymarket", "PMASTERS", "The Masters - Winner", "Rory McIlroy [Yes]", "Sports", "Golf")

    assert not market_intent_compatible(nba_conference, ncaa_conference)
    assert not market_intent_compatible(nba_from_event_id, ncaa_conference)
    assert not market_intent_compatible(europa, serie_a)
    assert not market_intent_compatible(womens_ucl, epl)
    assert not market_intent_compatible(wbc_winner, wbc_finalist)
    assert not market_intent_compatible(wbc_winner, wbc_knockout)
    assert not market_intent_compatible(ap_poll, ncaa_seed)
    assert not market_intent_compatible(nba_play_in, ncaa_winner)
    assert not market_intent_compatible(wcc_regular, wcc_conference)
    assert market_intent_compatible(wcc_regular, wcc_matching_regular)
    assert market_intent_compatible(masters_long, masters_short)


def test_scope_guards_compare_sports_objectives_and_tournaments():
    presidents_trophy = make_profile("kalshi", "KPRES", "NHL President's Trophy", "Boston Bruins", "Sports", "Hockey")
    division = make_profile("polymarket", "PDIV", "NHL Atlantic Division Winner", "Boston Bruins [Yes]", "Sports", "Hockey")
    division_winner = make_profile("kalshi", "KPDIV", "NHL Pacific Division Winner", "Edmonton Oilers", "Sports", "Hockey")
    playoffs = make_profile("polymarket", "PPLAYOFF", "Which teams will make the NHL Playoffs?", "Edmonton Oilers [Yes]", "Sports", "Hockey")
    national_champion = make_profile("kalshi", "KNCAATITLE", "Men's College Basketball Champion", "UConn", "Sports", "Basketball")
    conference_regular = make_profile("polymarket", "PBIGEAST", "Big East Men's College Basketball Regular Season Champion", "UConn [Yes]", "Sports", "Basketball")
    ufc_weight = make_profile("kalshi", "KUFCFLY", "UFC Flyweight Title Holder on Dec 31, 2026", "Alexandre Pantoja", "Sports", "MMA")
    ufc_any = make_profile("polymarket", "PUFCANY", "Who will become a UFC champion in 2026?", "Alexandre Pantoja [Yes]", "Sports", "MMA")
    baseball_steals = make_profile("kalshi", "KSTEALS", "Pro Baseball Steals Leader", "Bobby Witt Jr.", "Sports", "Baseball")
    baseball_mvp = make_profile("polymarket", "PMVP2", "Pro Baseball: 2026 AL MVP", "Bobby Witt Jr. [Yes]", "Sports", "Baseball")
    comeback = make_profile("kalshi", "KCOMEBACK", "AL Comeback Player of the Year Winner", "Gerrit Cole", "Sports", "Baseball")
    cy_young = make_profile("polymarket", "PCYY", "Pro Baseball: 2026 AL Cy Young Winner", "Gerrit Cole [Yes]", "Sports", "Baseball")
    french_open = make_profile("kalshi", "KFRENCH", "Women's French Open Winner", "Iga Swiatek", "Sports", "Tennis")
    wimbledon = make_profile("polymarket", "PWIMB", "2026 Women's Wimbledon Winner", "Iga Swiatek [Yes]", "Sports", "Tennis")
    welterweight = make_profile("kalshi", "KUFCWELTER", "UFC Welterweight Title Holder on Dec 31, 2026", "Ilia Topuria", "Sports", "MMA")
    lightweight = make_profile("polymarket", "PUFCLIGHT", "Who will be UFC Lightweight champion at the end of 2026?", "Ilia Topuria [Yes]", "Sports", "MMA")
    art_ross = make_profile("kalshi", "KARTROSS", "NHL Art Ross Trophy", "Leon Draisaitl", "Sports", "Hockey")
    selke = make_profile("polymarket", "PSELKE", "NHL Frank J. Selke Trophy Winner", "Leon Draisaitl [Yes]", "Sports", "Hockey")
    rookie = make_profile("kalshi", "KROOKIE", "NL Rookie of the Year Winner", "Nolan McLean", "Sports", "Baseball")
    cy_young_rookie = make_profile("polymarket", "PCYROOKIE", "Pro Baseball: 2026 NL Cy Young Winner", "Nolan McLean [Yes]", "Sports", "Baseball")
    indian_wells = make_profile("kalshi", "KINDIAN", "Indian Wells Women's Singles Winner", "Iga Swiatek", "Sports", "Tennis")
    french_open_same_player = make_profile("polymarket", "PFRENCH2", "2026 Women's French Open Winner", "Iga Swiatek [Yes]", "Sports", "Tennis")
    nba_cover = make_profile("kalshi", "KNBACOVER", "Who will be on the cover of NBA 2K27?", "Stephen Curry", "Sports", "Basketball")
    nba_award = make_profile("polymarket", "PNBAAWARD", "NBA Clutch Player of the Year Winner", "Stephen Curry [Yes]", "Sports", "Basketball")
    finals_mvp = make_profile("kalshi", "KFINALSMVP", "Finals MVP Winner?", "Nikola Jokic", "Sports", "Basketball")
    season_mvp = make_profile("polymarket", "PSEASONMVP", "NBA MVP", "Nikola Jokic [Yes]", "Sports", "Basketball")
    cover_retirement = make_profile("kalshi", "KCOVERRET", "Who will be on the cover of NBA 2K27?", "LeBron James", "Sports", "Basketball")
    retirement = make_profile("polymarket", "PRETIRELBJ", "Will LeBron James retire before next NBA season?", "LeBron James [Yes]", "Sports", "Basketball")
    fa_cup = make_profile("kalshi", "KFACUP", "FA Cup Winner", "Manchester City", "Sports", "Soccer")
    carabao = make_profile("polymarket", "PCARABAO", "Carabao Cup Winner", "Manchester City [Yes]", "Sports", "Soccer")
    world_cup_boot = make_profile("kalshi", "KWCGOLD", "FIFA World Cup Golden Boot Winner", "Lionel Messi", "Sports", "Soccer")
    mls_boot = make_profile("polymarket", "PMLSGOLD", "MLS: 2026 Golden Boot Winner", "Lionel Messi [Yes]", "Sports", "Soccer")
    womens_champions = make_profile("kalshi", "KWCL2", "Women's Champions League Champion", "Arsenal", "Sports", "Soccer")
    mens_champions = make_profile("polymarket", "PUCL2", "UEFA Champions League Winner", "Arsenal [Yes]", "Sports", "Soccer")

    assert not market_intent_compatible(presidents_trophy, division)
    assert not market_intent_compatible(division_winner, playoffs)
    assert not market_intent_compatible(national_champion, conference_regular)
    assert not market_intent_compatible(ufc_weight, ufc_any)
    assert not market_intent_compatible(baseball_steals, baseball_mvp)
    assert not market_intent_compatible(comeback, cy_young)
    assert not market_intent_compatible(french_open, wimbledon)
    assert not market_intent_compatible(welterweight, lightweight)
    assert not market_intent_compatible(art_ross, selke)
    assert not market_intent_compatible(rookie, cy_young_rookie)
    assert not market_intent_compatible(indian_wells, french_open_same_player)
    assert not market_intent_compatible(nba_cover, nba_award)
    assert not market_intent_compatible(finals_mvp, season_mvp)
    assert not market_intent_compatible(cover_retirement, retirement)
    assert not market_intent_compatible(presidents_trophy, playoffs)
    assert not market_intent_compatible(fa_cup, carabao)
    assert not market_intent_compatible(world_cup_boot, mls_boot)
    assert not market_intent_compatible(womens_champions, mens_champions)


def test_scope_guards_compare_general_non_sports_intentions():
    meeting = make_profile("kalshi", "KMEET", "Who will Donald Trump meet before 2027?", "Nicolas Maduro", "Politics", "World")
    pardon = make_profile("polymarket", "PPARDON", "Who will Trump pardon before 2027?", "Nicolas Maduro [Yes]", "Politics", "World")
    time_person = make_profile("kalshi", "KTIME", "TIME Person of the Year", "Elon Musk", "Culture", "People")
    richest = make_profile("polymarket", "PRICH", "Richest person at the end of 2026", "Elon Musk [Yes]", "Culture", "People")
    ipo_announce = make_profile("kalshi", "KIPOANN", "Which company will announce an IPO before 2027?", "OpenAI", "Economics", "Business")
    largest_ipo = make_profile("polymarket", "PLIPO", "Largest IPO by market cap in 2026", "OpenAI [Yes]", "Economics", "Business")
    generic_ipo = make_profile("polymarket", "PIPO2", "IPOs before 2027?", "OpenAI [Yes]", "Economics", "Business")
    fed_governor = make_profile("kalshi", "KFEDGOV", "Next Federal Reserve Governor nominee", "Kevin Warsh", "Economics", "Fed")
    fed_chair = make_profile("polymarket", "KFEDCHAIR", "Next Fed Chair", "Kevin Warsh [Yes]", "Economics", "Fed")
    ufc_weight = make_profile("kalshi", "KUFCW", "Next UFC Heavyweight Champion", "Tom Aspinall", "Sports", "MMA")
    ufc_any = make_profile("polymarket", "PUFCA", "Who will become a UFC champion?", "Tom Aspinall [Yes]", "Sports", "MMA")
    most_oscars = make_profile("kalshi", "KOSCARS", "Which film will win the most Oscars?", "Hamnet", "Culture", "Movies")
    best_picture = make_profile("polymarket", "PBESTPIC", "Academy Awards Best Picture Winner", "Hamnet [Yes]", "Culture", "Movies")
    cinematography = make_profile("polymarket", "PCINEMA", "Oscars 2026: Best Cinematography Winner", "Hamnet [Yes]", "Culture", "Movies")
    google_search = make_profile("kalshi", "KGOOGLE", "Top 5 Most Searched People on Google in 2026", "Timothee Chalamet", "Culture", "People")
    sexiest = make_profile("polymarket", "PSEXIEST", "People's Sexiest Man Alive 2026", "Timothee Chalamet [Yes]", "Culture", "People")
    district_nominee = make_profile("kalshi", "KGA14", "GA-14 Republican nominee?", "Star Black", "Politics", "Primaries")
    district_winner = make_profile("polymarket", "PGA14", "GA-14 special election winner?", "Star Black [Yes]", "Politics", "Elections")
    trump_says = make_profile("kalshi", "KTRUMPSAY", "What will Trump say this week?", "Epstein", "Mentions", "World")
    rogan_says = make_profile("polymarket", "PROGANSAY", "What will be said on the first Joe Rogan Experience episode of the week?", "Epstein [Yes]", "Mentions", "YouTube")
    district_house = make_profile("kalshi", "KOH14", "Which party will win the House race for OH-14?", "Democratic party", "Politics", "Elections")
    national_house = make_profile("polymarket", "PHOUSE", "Which party will win the House in 2026?", "Democratic Party [Yes]", "Politics", "Elections")
    governor_nominee = make_profile("kalshi", "KMEGOV", "Maine Democratic Governor nominee?", "Jared Golden", "Politics", "Primaries")
    district_primary = make_profile("polymarket", "PME02", "ME-02 Democratic Primary Winner", "Jared Golden [Yes]", "Politics", "Primaries")
    senate_nominee = make_profile("kalshi", "KTXSEN", "Texas Republican Senate nominee?", "John Cornyn", "Politics", "Primaries")
    endorsement = make_profile("polymarket", "PENDORSE", "Trump Endorsement in Texas GOP Senate Primary", "John Cornyn [Yes]", "Politics", "Primaries")
    overall_winner = make_profile("kalshi", "KBRAZIL", "Brazil Presidential election winner?", "Lula", "Politics", "Elections")
    runoff_advance = make_profile("polymarket", "PBRAZILRUN", "Which candidates will advance to Brazil's presidential runoff?", "Lula [Yes]", "Politics", "Elections")
    first_round = make_profile("polymarket", "PBRAZILR1", "Brazil Presidential Election 1st round winner?", "Lula [Yes]", "Politics", "Elections")
    presidential_nominee = make_profile("kalshi", "KPRESNOM", "2028 Republican nominee for President?", "Thomas Massie", "Politics", "Elections")
    district_nominee = make_profile("polymarket", "PKY04", "KY-04 Republican Primary Winner", "Thomas Massie [Yes]", "Politics", "Primaries")
    republican_senate = make_profile("kalshi", "KNCSEN", "North Carolina Republican Senate nominee?", "Roy Cooper", "Politics", "Primaries")
    democratic_senate = make_profile("polymarket", "PNCSEN", "North Carolina Democratic Senate Primary Winner", "Roy Cooper [Yes]", "Politics", "Primaries")
    vp_nominee = make_profile("kalshi", "KVPNOM", "Who will be the Democratic VP nominee in 2028?", "Gavin Newsom", "Politics", "Elections")
    president_nominee = make_profile("polymarket", "PPRESNOM", "Democratic Presidential Nominee 2028", "Gavin Newsom [Yes]", "Politics", "Elections")
    state_house = make_profile("kalshi", "KPASTATE", "Pennsylvania State House winner?", "Democratic party", "Politics", "Elections")
    federal_house = make_profile("polymarket", "PPA10", "PA-10 House Election Winner", "Democratic Party [Yes]", "Politics", "Elections")
    governor_winner = make_profile("kalshi", "KAKGOV", "Alaska Governor winner? (Person)", "Tom Begich", "Politics", "Elections")
    primary_advance = make_profile("polymarket", "PAKADV", "Who will advance from the Alaska Governor primary?", "Tom Begich [Yes]", "Politics", "Primaries")
    run_for_president = make_profile("kalshi", "KRUNBR", "Who will run for President of Brazil?", "Lula", "Politics", "Elections")
    slovenia = make_profile("kalshi", "KSLOVENIA", "Who will win the 2026 Slovenian parliamentary election?", "Social Democrats", "Politics", "Elections")
    denmark = make_profile("polymarket", "PDENMARK", "Denmark Parliamentary Election Winner", "Social Democrats [Yes]", "Politics", "Elections")
    attorney_general = make_profile("kalshi", "KTXAG", "Texas Republican Attorney General nominee?", "Chip Roy", "Politics", "Primaries")
    house_primary = make_profile("polymarket", "PTX21", "TX-21 Republican Primary Winner", "Chip Roy [Yes]", "Politics", "Primaries")
    california_advancer = make_profile("kalshi", "KCAGOVADV", "California Governor primary advancers? (Person)", "Eleni Kounalakis", "Politics", "Primaries")
    california_winner = make_profile("polymarket", "KCAGOVWIN", "California Governor Election Winner", "Eleni Kounalakis [Yes]", "Politics", "Elections")
    trump_talk = make_profile("kalshi", "KTRUMPTALK", "Who will Donald Trump talk to in March?", "Chuck Schumer", "Politics", "World")
    dhs_vote = make_profile("polymarket", "PDHSVOTE", 'Who will vote "Yea" on the DHS Appropriations Act, 2026 by March 31?', "Chuck Schumer [Yes]", "Politics", "Legislation")
    spotify_album = make_profile("kalshi", "KSPOTALBUM", "#2 Album on Spotify in 2026?", "KPop Demon Hunters", "Culture", "Music")
    netflix_movie = make_profile("polymarket", "PNETFLIX", "What will be the #2 global Netflix movie this week?", "KPop Demon Hunters [Yes]", "Culture", "Movies")
    trump_meet = make_profile("polymarket", "PTRUMPMEET", "Who will Trump meet with in March?", "Keir Starmer [Yes]", "Politics", "World")
    primary_second = make_profile("kalshi", "KTXSECOND", "2nd place in first round of Texas Senate Republican primary?", "Ken Paxton", "Politics", "Primaries")
    trump_endorsement = make_profile("polymarket", "PTXENDORSE", "Trump Endorsement in Texas GOP Senate Primary", "Ken Paxton [Yes]", "Politics", "Primaries")
    senate_nominee_final = make_profile("kalshi", "KTXNOM", "Texas Republican Senate nominee?", "Ken Paxton", "Politics", "Primaries")
    primary_third = make_profile("polymarket", "KTXTHIRD", "3rd place in first round of Texas Republican Senate Primary?", "Ken Paxton [Yes]", "Politics", "Primaries")
    run_for_nomination = make_profile("kalshi", "KRUNNOM", "Who will run for the 2028 Republican presidential nomination?", "Ron DeSantis", "Politics", "Elections")
    become_nominee = make_profile("polymarket", "PBECOMENOM", "Republican Presidential Nominee 2028", "Ron DeSantis [Yes]", "Politics", "Elections")
    advance_primary = make_profile("kalshi", "KADVCA", "Who will advance from the CA-11 primary?", "Nancy Pelosi", "Politics", "Primaries")
    first_primary = make_profile("polymarket", "PFIRSTCA", "Who will place first in the primary for Nancy Pelosi's congressional seat (CA-11)?", "Nancy Pelosi [Yes]", "Politics", "Primaries")
    first_round_primary = make_profile("polymarket", "PTXR1", "Texas Republican Senate Primary: First Round Winner", "Ken Paxton [Yes]", "Politics", "Primaries")
    run_2028 = make_profile("kalshi", "KRUN2028", "Who will announce a presidential run for 2028?", "Gavin Newsom", "Politics", "Elections")
    run_before_2027 = make_profile("polymarket", "PRUN2027", "Who will announce Presidential run before 2027?", "Gavin Newsom [Yes]", "Politics", "Elections")
    run_binary_2026 = make_profile("polymarket", "PRUNBIN26", "Will Gavin Newsom announce Presidential run by...?", "December 31, 2026 [Yes]", "Politics", "Elections")
    recognition_one = make_profile("kalshi", "KRECOG", "Who will recognize Palestine before 2027?", "Italy", "Politics", "World")
    recognition_two = make_profile("polymarket", "PRECOG", "Which countries will recognize Palestine before 2027?", "Italy [Yes]", "Politics", "World")
    join_admin = make_profile("kalshi", "KJOIN", "Who will join the Trump administration before July?", "Marco Rubio", "Politics", "World")
    trump_meeting = make_profile("polymarket", "PMEET2", "Who will Trump meet with in March?", "Marco Rubio [Yes]", "Politics", "World")
    trump_says_again = make_profile("polymarket", "PSAYJOIN", "What will Trump say in March?", "Marco Rubio [Yes]", "Mentions", "Politics")
    ipo_first = make_profile("kalshi", "KIPOFIRST", "Will OpenAI or Anthropic IPO first?", "OpenAI", "Economics", "Business")
    ipo_largest = make_profile("polymarket", "PIPOLARGE", "Largest IPO by market cap in 2026?", "OpenAI [Yes]", "Economics", "Business")
    live_speech = make_profile("kalshi", "KLIVESAY", "What will Trump say during his next live announcement?", "Tariffs", "Mentions", "Politics")
    weekly_speech = make_profile("polymarket", "PWEEKSAY", "What will Trump say this week (March 8)?", "Tariffs [Yes]", "Mentions", "Politics")
    oscar_score = make_profile("kalshi", "KOSCSCORE", "2026 Oscar for Best Music (Original Score)?", "Sinners", "Culture", "Awards")
    oscar_screenplay = make_profile("polymarket", "POSCSCREEN", "Oscars 2026: Best Original Screenplay Winner", "Sinners [Yes]", "Culture", "Awards")
    actor_award = make_profile("kalshi", "KACTORAWARD", "Actor Award for Outstanding Performance by a Male Actor in a Supporting Role?", "Kieran Culkin", "Culture", "Awards")
    oscar_actor = make_profile("polymarket", "POSCACTOR", "Oscars 2026: Best Supporting Actor Winner", "Kieran Culkin [Yes]", "Culture", "Awards")
    matching_oscar_score = make_profile("polymarket", "POSCSCORE", "Oscars 2026: Best Original Score Winner", "Sinners [Yes]", "Culture", "Awards")
    player_transfer = make_profile("kalshi", "KTRANSFER", "Which soccer players will transfer clubs before Sep 15, 2026?", "Bruno Fernandes", "Sports", "Soccer")
    player_award = make_profile("polymarket", "PPFA", "2025-2026 PFA Players' Player of the Year Winner", "Bruno Fernandes [Yes]", "Sports", "Soccer")
    united_manager = make_profile("kalshi", "KMUFCMGR", "Who will be the next manager of Manchester United?", "Xabi Alonso", "Sports", "Soccer")
    madrid_manager = make_profile("polymarket", "PRMMGR", "Next Real Madrid manager?", "Xabi Alonso [Yes]", "Sports", "Soccer")
    matching_united_manager = make_profile("polymarket", "PMUFCMGR", "Next Manchester United manager?", "Xabi Alonso [Yes]", "Sports", "Soccer")

    assert not market_intent_compatible(meeting, pardon)
    assert not market_intent_compatible(time_person, richest)
    assert not market_intent_compatible(ipo_announce, largest_ipo)
    assert market_intent_compatible(ipo_announce, generic_ipo)
    assert not market_intent_compatible(fed_governor, fed_chair)
    assert not market_intent_compatible(ufc_weight, ufc_any)
    assert not market_intent_compatible(most_oscars, best_picture)
    assert not market_intent_compatible(most_oscars, cinematography)
    assert not market_intent_compatible(google_search, sexiest)
    assert not market_intent_compatible(district_nominee, district_winner)
    assert not market_intent_compatible(trump_says, rogan_says)
    assert not market_intent_compatible(district_house, national_house)
    assert not market_intent_compatible(governor_nominee, district_primary)
    assert not market_intent_compatible(senate_nominee, endorsement)
    assert not market_intent_compatible(overall_winner, runoff_advance)
    assert not market_intent_compatible(overall_winner, first_round)
    assert not market_intent_compatible(presidential_nominee, district_nominee)
    assert not market_intent_compatible(republican_senate, democratic_senate)
    assert not market_intent_compatible(vp_nominee, president_nominee)
    assert not market_intent_compatible(state_house, federal_house)
    assert not market_intent_compatible(governor_winner, primary_advance)
    assert not market_intent_compatible(run_for_president, runoff_advance)
    assert not market_intent_compatible(slovenia, denmark)
    assert not market_intent_compatible(attorney_general, house_primary)
    assert not market_intent_compatible(california_advancer, california_winner)
    assert not market_intent_compatible(trump_talk, dhs_vote)
    assert not market_intent_compatible(spotify_album, netflix_movie)
    assert not market_intent_compatible(trump_talk, trump_meet)
    assert not market_intent_compatible(primary_second, trump_endorsement)
    assert not market_intent_compatible(senate_nominee_final, primary_third)
    assert not market_intent_compatible(run_for_nomination, become_nominee)
    assert not market_intent_compatible(advance_primary, first_primary)
    assert not market_intent_compatible(senate_nominee_final, first_round_primary)
    assert not market_intent_compatible(run_2028, run_before_2027)
    assert not market_intent_compatible(run_2028, run_binary_2026)
    assert market_intent_compatible(recognition_one, recognition_two)
    assert not market_intent_compatible(join_admin, trump_meeting)
    assert not market_intent_compatible(join_admin, trump_says_again)
    assert not market_intent_compatible(ipo_first, ipo_largest)
    assert not market_intent_compatible(live_speech, weekly_speech)
    assert not market_intent_compatible(oscar_score, oscar_screenplay)
    assert not market_intent_compatible(actor_award, oscar_actor)
    assert market_intent_compatible(oscar_score, matching_oscar_score)
    assert not market_intent_compatible(player_transfer, player_award)
    assert not market_intent_compatible(united_manager, madrid_manager)
    assert market_intent_compatible(united_manager, matching_united_manager)


def test_scope_guards_compare_annual_and_explicit_deadline_windows():
    annual_album = make_profile("kalshi", "KANNUALALBUM", "Who will release a new album in 2026?", "Lana Del Rey", "Culture", "Music")
    june_album = make_profile("polymarket", "PJUNEALBUM", "Will Lana Del Rey release a new album by June 30, 2026?", "Lana Del Rey [Yes]", "Culture", "Music")
    year_end_album = make_profile("polymarket", "PDECALBUM", "Will Lana Del Rey release a new album by December 31, 2026?", "Lana Del Rey [Yes]", "Culture", "Music")

    assert not market_intent_compatible(annual_album, june_album)
    assert market_intent_compatible(annual_album, year_end_album)


def test_outcome_bundles_preserve_every_mapping():
    candidates, _ = generate_outcome_candidates(
        kalshi_rows(),
        polymarket_rows(),
        category_mapping={"sports": {"sports", "soccer"}},
    )
    bundles = build_outcome_bundles(candidates)

    assert bundles
    assert sum(bundle["mapping_count"] for bundle in bundles) == len(candidates)
    assert len({bundle["bundle_id"] for bundle in bundles}) == len(bundles)
    assert all(bundle["outcome_mappings"] for bundle in bundles)


def run() -> None:
    test_strip_polymarket_outcome_suffix()
    test_subject_alias_normalization()
    test_deadline_extraction_ignores_fallback_resolution_date()
    test_generates_world_cup_winner_country_candidates()
    test_generates_group_winner_candidate()
    test_rejects_different_scope_and_no_tokens()
    test_rejects_same_outcome_with_incompatible_political_scope()
    test_rejects_same_numeric_outcome_with_incompatible_category()
    test_binary_polymarket_title_supplies_subject_when_bet_is_a_date()
    test_rejects_same_subject_when_event_intents_conflict()
    test_rejects_song_release_but_keeps_album_release_for_same_artist()
    test_scope_guards_reject_sport_result_rank_and_period_mismatches()
    test_scope_guards_compare_explicit_actors_and_company_actions()
    test_scope_guards_compare_years_locations_metrics_roles_and_awards()
    test_scope_guards_compare_exact_sports_metrics_and_named_awards()
    test_scope_guards_compare_competition_namespace_stage_and_rank()
    test_scope_guards_compare_leagues_levels_and_tournament_stages()
    test_scope_guards_compare_sports_objectives_and_tournaments()
    test_scope_guards_compare_general_non_sports_intentions()
    test_scope_guards_compare_annual_and_explicit_deadline_windows()
    test_outcome_bundles_preserve_every_mapping()


if __name__ == "__main__":
    run()
    print("ok")
