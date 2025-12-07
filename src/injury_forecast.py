"""
Injury/Suspension Forecast for Upcoming Matches

Fetches current player availability data from Transfermarkt for teams
with upcoming matches, then adds injury features to the prediction pipeline.

Usage:
    from src.injury_forecast import fetch_injuries_for_matches
    odds_df = fetch_injuries_for_matches(odds_df)
"""
from __future__ import annotations

import time
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import pandas as pd

# Cache for injury data (team -> injury count) to avoid re-scraping
_INJURY_CACHE: Dict[str, Dict[str, int]] = {}
_CACHE_TIMESTAMP: Optional[datetime] = None
_CACHE_TTL_HOURS = 6  # Cache valid for 6 hours


# Team name mappings: Norsk Tipping / common names -> Transfermarkt format
# Format: "NT Name": ("slug", team_id)
TEAM_TRANSFERMARKT_MAP = {
    # England - Premier League (using exact Norsk Tipping names)
    "Manchester United": ("manchester-united", 985),
    "Man United": ("manchester-united", 985),
    "Manchester City": ("manchester-city", 281),
    "Man City": ("manchester-city", 281),
    "Liverpool": ("fc-liverpool", 31),
    "Arsenal": ("fc-arsenal", 11),
    "Chelsea": ("fc-chelsea", 631),
    "Tottenham Hotspur": ("tottenham-hotspur", 148),
    "Tottenham": ("tottenham-hotspur", 148),
    "Newcastle United": ("newcastle-united", 762),
    "Newcastle": ("newcastle-united", 762),
    "Aston Villa": ("aston-villa", 405),
    "Brighton and Hove Albion": ("brighton-amp-hove-albion", 1237),
    "Brighton": ("brighton-amp-hove-albion", 1237),
    "West Ham United": ("west-ham-united", 379),
    "West Ham United FC": ("west-ham-united", 379),
    "West Ham": ("west-ham-united", 379),
    "Bournemouth": ("afc-bournemouth", 989),
    "AFC Bournemouth": ("afc-bournemouth", 989),
    "Fulham": ("fc-fulham", 931),
    "Wolverhampton Wanderers": ("wolverhampton-wanderers", 543),
    "Wolves": ("wolverhampton-wanderers", 543),
    "Crystal Palace": ("crystal-palace", 873),
    "Everton": ("fc-everton", 29),
    "Brentford": ("fc-brentford", 1148),
    "Nottingham Forest": ("nottingham-forest", 703),
    "Nott'm Forest": ("nottingham-forest", 703),
    "Ipswich Town": ("ipswich-town", 677),
    "Ipswich": ("ipswich-town", 677),
    "Leicester City": ("leicester-city", 1003),
    "Leicester": ("leicester-city", 1003),
    "Southampton": ("fc-southampton", 180),
    
    # England - Championship
    "Leeds United": ("leeds-united", 399),
    "Leeds": ("leeds-united", 399),
    "Burnley": ("fc-burnley", 1132),
    "Sheffield United": ("sheffield-united", 350),
    "Luton Town": ("luton-town", 1031),
    "Luton": ("luton-town", 1031),
    "Middlesbrough": ("fc-middlesbrough", 432),
    "Sunderland": ("afc-sunderland", 289),
    "West Bromwich Albion": ("west-bromwich-albion", 984),
    "West Brom": ("west-bromwich-albion", 984),
    "Norwich City": ("norwich-city", 1123),
    "Norwich": ("norwich-city", 1123),
    "Coventry City": ("coventry-city", 435),
    "Coventry": ("coventry-city", 435),
    "Bristol City": ("bristol-city", 1062),
    "Stoke City": ("stoke-city", 512),
    "Stoke": ("stoke-city", 512),
    "Watford": ("fc-watford", 1010),
    "Swansea City": ("swansea-city", 2288),
    "Swansea": ("swansea-city", 2288),
    "Millwall": ("fc-millwall", 1577),
    "Hull City": ("hull-city", 1065),
    "Hull": ("hull-city", 1065),
    "Blackburn Rovers": ("blackburn-rovers", 164),
    "Blackburn": ("blackburn-rovers", 164),
    "Cardiff City": ("cardiff-city", 1047),
    "Cardiff": ("cardiff-city", 1047),
    "Plymouth Argyle": ("plymouth-argyle", 1070),
    "Plymouth": ("plymouth-argyle", 1070),
    "Preston North End": ("preston-north-end", 1042),
    "Preston": ("preston-north-end", 1042),
    "Queens Park Rangers": ("queens-park-rangers", 1039),
    "QPR": ("queens-park-rangers", 1039),
    "Derby County": ("derby-county", 22),
    "Derby": ("derby-county", 22),
    "Portsmouth": ("fc-portsmouth", 1000),
    "Oxford United": ("oxford-united", 8611),
    "Barnsley FC": ("fc-barnsley", 349),
    "Barnsley": ("fc-barnsley", 349),
    "Blackpool FC": ("blackpool-fc", 1062),
    "Blackpool": ("blackpool-fc", 1062),
    "Bolton Wanderers": ("bolton-wanderers", 855),
    "Bolton": ("bolton-wanderers", 855),
    "Huddersfield Town": ("huddersfield-town", 1110),
    "Huddersfield": ("huddersfield-town", 1110),
    "Reading": ("fc-reading", 1080),
    "Stevenage": ("stevenage-fc", 2462),
    "Wycombe Wanderers": ("wycombe-wanderers", 2440),
    "Wycombe": ("wycombe-wanderers", 2440),
    "Bradford City FC": ("bradford-city", 974),
    "Bradford City": ("bradford-city", 974),
    "Doncaster Rovers": ("doncaster-rovers", 1096),
    "Doncaster": ("doncaster-rovers", 1096),
    "Exeter City": ("exeter-city", 1071),
    "Exeter": ("exeter-city", 1071),
    "Lincoln City": ("lincoln-city", 3012),
    "Lincoln": ("lincoln-city", 3012),
    "Mansfield Town": ("mansfield-town", 1054),
    "Mansfield": ("mansfield-town", 1054),
    "Northampton Town": ("northampton-town", 1057),
    "Northampton": ("northampton-town", 1057),
    "Leyton Orient": ("leyton-orient", 1066),
    "Port Vale": ("port-vale", 1037),
    "Rotherham United": ("rotherham-united", 972),
    "Rotherham": ("rotherham-united", 972),
    "Stockport County FC": ("stockport-county", 2415),
    "Stockport County": ("stockport-county", 2415),
    "Peterborough United": ("peterborough-united", 1036),
    "Peterborough": ("peterborough-united", 1036),
    "AFC Wimbledon": ("afc-wimbledon", 3557),
    
    # Germany - Bundesliga (using exact Norsk Tipping names)
    "Bayern München": ("fc-bayern-munchen", 27),
    "Bayern Munich": ("fc-bayern-munchen", 27),
    "Borussia Dortmund": ("borussia-dortmund", 16),
    "Dortmund": ("borussia-dortmund", 16),
    "RB Leipzig": ("rasenballsport-leipzig", 23826),
    "Leipzig": ("rasenballsport-leipzig", 23826),
    "Bayer Leverkusen": ("bayer-04-leverkusen", 15),
    "Leverkusen": ("bayer-04-leverkusen", 15),
    "VfB Stuttgart": ("vfb-stuttgart", 79),
    "Stuttgart": ("vfb-stuttgart", 79),
    "Eintracht Frankfurt": ("eintracht-frankfurt", 24),
    "Frankfurt": ("eintracht-frankfurt", 24),
    "SC Freiburg": ("sc-freiburg", 60),
    "Freiburg": ("sc-freiburg", 60),
    "VfL Wolfsburg": ("vfl-wolfsburg", 82),
    "Wolfsburg": ("vfl-wolfsburg", 82),
    "Borussia Mönchengladbach": ("borussia-monchengladbach", 18),
    "Mönchengladbach": ("borussia-monchengladbach", 18),
    "M'gladbach": ("borussia-monchengladbach", 18),
    "Union Berlin": ("1-fc-union-berlin", 89),
    "TSG Hoffenheim": ("tsg-1899-hoffenheim", 533),
    "Hoffenheim": ("tsg-1899-hoffenheim", 533),
    "FC Augsburg": ("fc-augsburg", 167),
    "Augsburg": ("fc-augsburg", 167),
    "Werder Bremen": ("sv-werder-bremen", 86),
    "Bremen": ("sv-werder-bremen", 86),
    "1. FSV Mainz 05": ("1-fsv-mainz-05", 39),
    "Mainz": ("1-fsv-mainz-05", 39),
    "VfL Bochum": ("vfl-bochum", 80),
    "Bochum": ("vfl-bochum", 80),
    "1. FC Heidenheim": ("1-fc-heidenheim-1846", 2036),
    "Heidenheim": ("1-fc-heidenheim-1846", 2036),
    "FC St. Pauli": ("fc-st-pauli", 35),
    "St. Pauli": ("fc-st-pauli", 35),
    "Holstein Kiel": ("holstein-kiel", 820),
    "Kiel": ("holstein-kiel", 820),
    # Germany - 2. Bundesliga (using exact Norsk Tipping names)
    "1 FC Kaiserslautern": ("1-fc-kaiserslautern", 123),
    "1. FC Kaiserslautern": ("1-fc-kaiserslautern", 123),
    "Kaiserslautern": ("1-fc-kaiserslautern", 123),
    "1. FC Nürnberg": ("1-fc-nurnberg", 4),
    "Nürnberg": ("1-fc-nurnberg", 4),
    "1. FC Magdeburg": ("1-fc-magdeburg", 23),
    "Magdeburg": ("1-fc-magdeburg", 23),
    "Hamburger SV": ("hamburger-sv", 41),
    "Hamburg": ("hamburger-sv", 41),
    "Hertha Berlin": ("hertha-bsc", 44),
    "Hertha BSC": ("hertha-bsc", 44),
    "Hertha": ("hertha-bsc", 44),
    "Karlsruher SC": ("karlsruher-sc", 84),
    "Karlsruhe": ("karlsruher-sc", 84),
    "FC Köln": ("1-fc-koln", 3),
    "Köln": ("1-fc-koln", 3),
    "Koln": ("1-fc-koln", 3),
    "SC Paderborn": ("sc-paderborn-07", 127),
    "Paderborn": ("sc-paderborn-07", 127),
    "Greuther Fürth": ("spvgg-greuther-furth", 115),
    "Fürth": ("spvgg-greuther-furth", 115),
    "Eintracht Braunschweig": ("eintracht-braunschweig", 74),
    "Braunschweig": ("eintracht-braunschweig", 74),
    "Jahn Regensburg": ("ssv-jahn-regensburg", 109),
    "Regensburg": ("ssv-jahn-regensburg", 109),
    "SV Darmstadt 98": ("sv-darmstadt-98", 105),
    "Darmstadt": ("sv-darmstadt-98", 105),
    "SV 07 Elversberg": ("sv-07-elversberg", 244),
    "Elversberg": ("sv-07-elversberg", 244),
    "SSV Ulm 1846": ("ssv-ulm-1846", 170),
    "Ulm": ("ssv-ulm-1846", 170),
    "Arminia Bielefeld": ("arminia-bielefeld", 10),
    "Bielefeld": ("arminia-bielefeld", 10),
    "Dynamo Dresden": ("sg-dynamo-dresden", 106),
    "Dresden": ("sg-dynamo-dresden", 106),
    "SV Wehen Wiesbaden": ("sv-wehen-wiesbaden", 125),
    "Wiesbaden": ("sv-wehen-wiesbaden", 125),
    
    # Spain - La Liga
    "Real Madrid": ("real-madrid", 418),
    "FC Barcelona": ("fc-barcelona", 131),
    "Barcelona": ("fc-barcelona", 131),
    "Atletico Madrid": ("atletico-madrid", 13),
    "Atlético Madrid": ("atletico-madrid", 13),
    "Sevilla": ("fc-sevilla", 368),
    "Real Sociedad": ("real-sociedad-san-sebastian", 681),
    "Sociedad": ("real-sociedad-san-sebastian", 681),
    "Real Betis": ("real-betis-sevilla", 150),
    "Betis": ("real-betis-sevilla", 150),
    "Athletic Club Bilbao": ("athletic-bilbao", 621),
    "Athletic Bilbao": ("athletic-bilbao", 621),
    "Ath Bilbao": ("athletic-bilbao", 621),
    "Villarreal": ("fc-villarreal", 1050),
    "Valencia": ("fc-valencia", 1049),
    "Osasuna": ("ca-osasuna", 331),
    "Celta Vigo": ("celta-de-vigo", 940),
    "Celta": ("celta-de-vigo", 940),
    "Getafe": ("fc-getafe", 3709),
    "Espanyol": ("rcd-espanyol-barcelona", 714),
    "Rayo Vallecano": ("rayo-vallecano", 367),
    "Vallecano": ("rayo-vallecano", 367),
    "Mallorca": ("rcd-mallorca", 237),
    "Girona": ("fc-girona", 12321),
    "Las Palmas": ("ud-las-palmas", 472),
    "Alaves": ("deportivo-alaves", 1108),
    "Alavés": ("deportivo-alaves", 1108),
    "Leganes": ("cd-leganes", 1244),
    "Leganés": ("cd-leganes", 1244),
    "Valladolid": ("real-valladolid", 366),
    "Real Valladolid": ("real-valladolid", 366),
    # Spain - Segunda Division (using exact Norsk Tipping names)
    "Elche": ("elche-cf", 1531),
    "Levante": ("levante-ud", 3368),
    "Granada": ("granada-cf", 16795),
    "Almeria": ("ud-almeria", 3302),
    "Malaga": ("fc-malaga", 1084),
    "Huesca": ("sd-huesca", 2919),
    "Cadiz": ("cadiz-cf", 2687),
    "Eibar": ("sd-eibar", 1533),
    "Real Oviedo": ("real-oviedo", 331),
    "Oviedo": ("real-oviedo", 331),
    "Real Zaragoza": ("real-saragossa", 237),
    "Zaragoza": ("real-saragossa", 237),
    "Racing Santander": ("racing-santander", 1123),
    "Deportivo La Coruna": ("deportivo-la-coruna", 897),
    "Deportivo": ("deportivo-la-coruna", 897),
    "Gijon": ("sporting-gijon", 2448),
    "Sporting Gijon": ("sporting-gijon", 2448),
    "Albacete": ("albacete-balompie", 2285),
    "Burgos": ("burgos-cf", 10205),
    "Mirandes": ("cd-mirandes", 6536),
    "Real Sociedad B": ("real-sociedad-san-sebastian-b", 10271),
    "Tenerife": ("cd-tenerife", 321),
    "Ferrol": ("racing-ferrol", 1955),
    "Castellon": ("cd-castellon", 1551),
    "Cordoba": ("cordoba-cf", 2559),
    
    # Italy - Serie A (using exact Norsk Tipping names)
    "Inter Milan": ("inter-mailand", 46),
    "Inter": ("inter-mailand", 46),
    "AC Milan": ("ac-mailand", 5),
    "Milan": ("ac-mailand", 5),
    "Juventus": ("juventus-turin", 506),
    "Napoli": ("ssc-neapel", 6195),
    "Atalanta": ("atalanta-bergamo", 800),
    "Atalanta BC": ("atalanta-bergamo", 800),
    "Roma": ("as-rom", 12),
    "AS Roma": ("as-rom", 12),
    "Lazio": ("lazio-rom", 398),
    "Fiorentina": ("ac-florenz", 430),
    "Bologna": ("fc-bologna", 1025),
    "Torino FC": ("fc-turin", 416),
    "Torino": ("fc-turin", 416),
    "Udinese": ("udinese-calcio", 410),
    "Empoli FC": ("fc-empoli", 749),
    "Empoli": ("fc-empoli", 749),
    "Cagliari": ("cagliari-calcio", 1390),
    "Hellas Verona": ("hellas-verona", 276),
    "Verona": ("hellas-verona", 276),
    "Parma": ("parma-calcio-1913", 130),
    "Parma Calcio": ("parma-calcio-1913", 130),
    "Genoa": ("cfc-genua-1893", 252),
    "Lecce": ("us-lecce", 1005),
    "AC Monza": ("ac-monza", 2919),
    "Monza": ("ac-monza", 2919),
    "Venezia": ("fc-venedig", 607),
    "Venezia FC": ("fc-venedig", 607),
    "Como": ("como-1907", 1047),
    "Como 1907": ("como-1907", 1047),
    # Italy - Serie B (using exact Norsk Tipping names)
    "Sampdoria Genoa": ("uc-sampdoria", 1038),
    "Sampdoria": ("uc-sampdoria", 1038),
    "Sassuolo Calcio": ("us-sassuolo", 6574),
    "Sassuolo": ("us-sassuolo", 6574),
    "Frosinone Calcio": ("frosinone-calcio", 10244),
    "Frosinone": ("frosinone-calcio", 10244),
    "Spezia Calcio": ("spezia-calcio", 3522),
    "Spezia": ("spezia-calcio", 3522),
    "Palermo FC": ("palermo-fc", 166),
    "Palermo": ("palermo-fc", 166),
    "SSC Bari": ("ssc-bari", 304),
    "Bari": ("ssc-bari", 304),
    "Modena FC": ("modena-fc-2018", 1032),
    "Modena": ("modena-fc-2018", 1032),
    "Reggiana 1919": ("ac-reggiana-1919", 2468),
    "Reggiana": ("ac-reggiana-1919", 2468),
    "Cesena FC": ("cesena-fc", 304),
    "Cesena": ("cesena-fc", 304),
    "Mantova 1911": ("mantova-fc", 10197),
    "Mantova": ("mantova-fc", 10197),
    "Carrarese Calcio": ("carrarese-calcio", 5936),
    "Carrarese": ("carrarese-calcio", 5936),
    "US Catanzaro": ("us-catanzaro-1929", 8927),
    "Catanzaro": ("us-catanzaro-1929", 8927),
    "Juve Stabia": ("ss-juve-stabia", 5081),
    "Pisa SC": ("pisa-sc", 5169),
    "Pisa": ("pisa-sc", 5169),
    "US Cremonese": ("us-cremonese", 2469),
    "Cremonese": ("us-cremonese", 2469),
    "Salernitana": ("us-salernitana-1919", 380),
    "Cittadella": ("as-cittadella", 2470),
    "Brescia": ("brescia-calcio", 1004),
    "Cosenza": ("cosenza-calcio", 4965),
    
    # France - Ligue 1 (using exact Norsk Tipping names)
    "Paris Saint-Germain": ("fc-paris-saint-germain", 583),
    "PSG": ("fc-paris-saint-germain", 583),
    "Olympique Marseille": ("olympique-marseille", 244),
    "Marseille": ("olympique-marseille", 244),
    "AS Monaco": ("as-monaco", 162),
    "Monaco": ("as-monaco", 162),
    "Olympique Lyon": ("olympique-lyon", 1041),
    "Lyon": ("olympique-lyon", 1041),
    "Lille": ("losc-lille", 1082),
    "LOSC Lille": ("losc-lille", 1082),
    "OGC Nice": ("ogc-nizza", 417),
    "Nice": ("ogc-nizza", 417),
    "RC Lens": ("rc-lens", 826),
    "Lens": ("rc-lens", 826),
    "Stade Rennes": ("stade-rennes", 273),
    "Rennes": ("stade-rennes", 273),
    "Stade Reims": ("stade-reims", 1421),
    "Reims": ("stade-reims", 1421),
    "Toulouse": ("fc-toulouse", 415),
    "RC Strasbourg": ("rc-strassburg-alsace", 667),
    "Strasbourg": ("rc-strassburg-alsace", 667),
    "Strasbourg Alsace": ("rc-strassburg-alsace", 667),
    "FC Nantes": ("fc-nantes", 995),
    "Nantes": ("fc-nantes", 995),
    "Montpellier HSC": ("montpellier-hsc", 969),
    "Montpellier": ("montpellier-hsc", 969),
    "Angers SCO": ("sco-angers", 1420),
    "Angers": ("sco-angers", 1420),
    "Le Havre": ("le-havre-ac", 738),
    "Auxerre": ("aj-auxerre", 290),
    "AS Saint-Etienne": ("as-saint-etienne", 618),
    "Saint-Etienne": ("as-saint-etienne", 618),
    # France - Ligue 2 (using exact Norsk Tipping names)
    "FC Metz": ("fc-metz", 347),
    "Metz": ("fc-metz", 347),
    "Paris FC": ("paris-fc", 982),
    "FC Lorient": ("fc-lorient", 1158),
    "Lorient": ("fc-lorient", 1158),
    "Stade Lavallois MFC": ("stade-laval", 1086),
    "Laval": ("stade-laval", 1086),
    "Pau FC": ("pau-fc", 9928),
    "Pau": ("pau-fc", 9928),
    "Troyes": ("estac-troyes", 415),
    "Rodez Aveyron Football": ("rodez-aveyron-football", 10210),
    "Rodez": ("rodez-aveyron-football", 10210),
    "USL Dunkerque": ("usl-dunkerque", 10219),
    "Dunkerque": ("usl-dunkerque", 10219),
    
    # Netherlands - Eredivisie (using exact Norsk Tipping names)
    "Ajax Amsterdam": ("afc-ajax", 610),
    "Ajax": ("afc-ajax", 610),
    "PSV Eindhoven": ("psv-eindhoven", 383),
    "PSV": ("psv-eindhoven", 383),
    "Feyenoord Rotterdam": ("feyenoord-rotterdam", 234),
    "Feyenoord": ("feyenoord-rotterdam", 234),
    "AZ Alkmaar": ("az-alkmaar", 1090),
    "AZ": ("az-alkmaar", 1090),
    "FC Twente Enschede": ("fc-twente-enschede", 317),
    "FC Twente": ("fc-twente-enschede", 317),
    "Twente": ("fc-twente-enschede", 317),
    "FC Utrecht": ("fc-utrecht", 200),
    "Utrecht": ("fc-utrecht", 200),
    "Sparta Rotterdam": ("sparta-rotterdam", 468),
    "Sparta": ("sparta-rotterdam", 468),
    "SC Heerenveen": ("sc-heerenveen", 306),
    "Heerenveen": ("sc-heerenveen", 306),
    "NEC Nijmegen": ("nec-nijmegen", 466),
    "NEC": ("nec-nijmegen", 466),
    "Go Ahead Eagles": ("go-ahead-eagles-deventer", 1090),
    "Vitesse Arnhem": ("vitesse-arnhem", 499),
    "Vitesse": ("vitesse-arnhem", 499),
    "Fortuna Sittard": ("fortuna-sittard", 472),
    "Heracles Almelo": ("heracles-almelo", 1304),
    "Heracles": ("heracles-almelo", 1304),
    "Willem II Tilburg": ("willem-ii-tilburg", 403),
    "Willem II": ("willem-ii-tilburg", 403),
    "RKC Waalwijk": ("rkc-waalwijk", 316),
    "RKC": ("rkc-waalwijk", 316),
    "PEC Zwolle": ("pec-zwolle", 1269),
    "Zwolle": ("pec-zwolle", 1269),
    "FC Groningen": ("fc-groningen", 202),
    "Groningen": ("fc-groningen", 202),
    "NAC Breda": ("nac-breda", 207),
    "NAC": ("nac-breda", 207),
    "Almere City": ("almere-city-fc", 6498),
    "FC Volendam": ("fc-volendam", 1010),
    "Volendam": ("fc-volendam", 1010),
    
    # Belgium - Pro League (using exact Norsk Tipping names)
    "Club Brugge": ("fc-brugge", 2282),
    "Brugge": ("fc-brugge", 2282),
    "RSC Anderlecht": ("rsc-anderlecht", 58),
    "Anderlecht": ("rsc-anderlecht", 58),
    "RSC Anderlecht Futures": ("rsc-anderlecht-ii", 20854),
    "Royal Antwerp FC": ("royal-antwerp-fc", 43),
    "Antwerp": ("royal-antwerp-fc", 43),
    "Union Saint-Gilloise": ("royale-union-saint-gilloise", 1766),
    "Union SG": ("royale-union-saint-gilloise", 1766),
    "KAA Gent": ("kaa-gent", 395),
    "Gent": ("kaa-gent", 395),
    "Cercle Brugge": ("cercle-brugge", 208),
    "Standard Liège": ("standard-luttich", 237),
    "Standard Liege": ("standard-luttich", 237),
    "Standard": ("standard-luttich", 237),
    "KRC Genk": ("krc-genk", 323),
    "Genk": ("krc-genk", 323),
    "Yellow-Red KV Mechelen": ("kv-mechelen", 322),
    "Mechelen": ("kv-mechelen", 322),
    "KVC Westerlo": ("kvc-westerlo", 373),
    "Westerlo": ("kvc-westerlo", 373),
    "OH Leuven": ("oud-heverlee-leuven", 2646),
    "Oud-Heverlee Leuven": ("oud-heverlee-leuven", 2646),
    "Leuven": ("oud-heverlee-leuven", 2646),
    "KV Kortrijk": ("kv-kortrijk", 301),
    "Kortrijk": ("kv-kortrijk", 301),
    "K Beerschot VA": ("beerschot-va", 2580),
    "Beerschot": ("beerschot-va", 2580),
    "Royal Charleroi SC": ("sporting-charleroi", 374),
    "Charleroi": ("sporting-charleroi", 374),
    "SV Zulte Waregem": ("sv-zulte-waregem", 302),
    "Zulte Waregem": ("sv-zulte-waregem", 302),
    "St. Truidense VV": ("sint-truidense-vv", 302),
    "Sint-Truiden": ("sint-truidense-vv", 302),
    "Jong KRC Genk": ("krc-genk-ii", 33803),
    "Club NXT": ("club-brugge-nxt", 20862),
    
    # Turkey - Super Lig (using exact Norsk Tipping names)
    "Galatasaray": ("galatasaray-istanbul", 141),
    "Fenerbahce Istanbul": ("fenerbahce-istanbul", 36),
    "Fenerbahce": ("fenerbahce-istanbul", 36),
    "Besiktas Istanbul": ("besiktas-istanbul", 114),
    "Besiktas": ("besiktas-istanbul", 114),
    "Trabzonspor": ("trabzonspor", 449),
    "Istanbul Basaksehir": ("istanbul-basaksehir-fk", 6890),
    "Basaksehir": ("istanbul-basaksehir-fk", 6890),
    "Konyaspor": ("konyaspor", 2381),
    "Alanyaspor": ("alanyaspor", 10484),
    "Antalyaspor": ("antalyaspor", 589),
    "Sivasspor": ("sivasspor", 2384),
    "Kasimpasa Istanbul": ("kasimpasa-sk", 3205),
    "Kasimpasa": ("kasimpasa-sk", 3205),
    "Caykur Rizespor": ("caykur-rizespor", 126),
    "Rizespor": ("caykur-rizespor", 126),
    "Gaziantep FK": ("gaziantep-fk", 15753),
    "Gaziantep": ("gaziantep-fk", 15753),
    "Kayserispor": ("kayserispor", 3575),
    "Samsunspor": ("samsunspor", 142),
    "Hatayspor": ("hatayspor", 2879),
    "Eyupspor": ("eyupspor", 11586),
    "Goztepe Izmir": ("goztepe-izmir", 556),
    "Goztepe": ("goztepe-izmir", 556),
    "Bodrumspor": ("bodrumspor", 10809),
    "Fatih Karagumruk Istanbul": ("fatih-karagumruk-istanbul", 5004),
    "Karagumruk": ("fatih-karagumruk-istanbul", 5004),
    "Genclerbirligi": ("genclerbirligi-ankara", 358),
    "Kocaelispor": ("kocaelispor", 2879),
    
    # Portugal - Primeira Liga (using exact Norsk Tipping names)
    "SL Benfica": ("sl-benfica", 294),
    "Benfica": ("sl-benfica", 294),
    "SL Benfica B": ("sl-benfica-b", 8554),
    "Benfica B": ("sl-benfica-b", 8554),
    "FC Porto": ("fc-porto", 720),
    "Porto": ("fc-porto", 720),
    "FC Porto B": ("fc-porto-b", 8555),
    "Porto B": ("fc-porto-b", 8555),
    "Sporting CP": ("sporting-lissabon", 336),
    "Sporting Lisbon": ("sporting-lissabon", 336),
    "Sporting Lisbon B": ("sporting-cp-b", 8616),
    "Sporting": ("sporting-lissabon", 336),
    "SC Braga": ("sporting-braga", 1075),
    "Braga": ("sporting-braga", 1075),
    "Vitória Guimarães": ("vitoria-sc", 2420),
    "Vitoria SC Guimaraes": ("vitoria-sc", 2420),
    "Guimaraes": ("vitoria-sc", 2420),
    "Rio Ave": ("rio-ave-fc", 1538),
    "Famalicão": ("fc-famalicao", 3524),
    "Famalicao": ("fc-famalicao", 3524),
    "Gil Vicente": ("gil-vicente-fc", 2449),
    "Gil Vicente Barcelos": ("gil-vicente-fc", 2449),
    "Boavista": ("boavista-porto", 746),
    "Moreirense": ("moreirense-fc", 2447),
    "Casa Pia": ("casa-pia-ac", 8616),
    "Casa Pia Lisbon": ("casa-pia-ac", 8616),
    "Estoril Praia": ("gd-estoril-praia", 2259),
    "FC Arouca": ("fc-arouca", 5639),
    "Arouca": ("fc-arouca", 5639),
    "Santa Clara": ("cd-santa-clara", 2463),
    "Santa Clara Azores": ("cd-santa-clara", 2463),
    "Nacional Funchal": ("cd-nacional", 2465),
    "CD Nacional Madeira": ("cd-nacional", 2465),
    "Nacional": ("cd-nacional", 2465),
    "Estrela Amadora": ("cf-estrela-da-amadora", 3463),
    "AVS Futebol SAD": ("avs-futebol-sad", 27755),
    "Farense": ("sc-farense", 8107),
    
    # Scotland - Premiership (using exact Norsk Tipping names)
    "Celtic Glasgow": ("celtic-glasgow", 371),
    "Celtic": ("celtic-glasgow", 371),
    "Rangers": ("rangers-fc", 124),
    "Rangers FC": ("rangers-fc", 124),
    "Glasgow Rangers": ("rangers-fc", 124),
    "Aberdeen": ("fc-aberdeen", 1147),
    "Aberdeen FC": ("fc-aberdeen", 1147),
    "Hearts": ("heart-of-midlothian-fc", 1244),
    "Heart of Midlothian": ("heart-of-midlothian-fc", 1244),
    "Heart of Midlothian FC": ("heart-of-midlothian-fc", 1244),
    "Hibernian FC": ("hibernian-edinburgh", 338),
    "Hibernian": ("hibernian-edinburgh", 338),
    "Dundee United": ("dundee-united-fc", 447),
    "Dundee FC": ("dundee-fc", 448),
    "Dundee": ("dundee-fc", 448),
    "St Mirren FC": ("fc-st-mirren", 1012),
    "St Mirren": ("fc-st-mirren", 1012),
    "Motherwell FC": ("fc-motherwell", 695),
    "Motherwell": ("fc-motherwell", 695),
    "Kilmarnock FC": ("kilmarnock-fc", 1078),
    "Kilmarnock": ("kilmarnock-fc", 1078),
    "Ross County FC": ("ross-county-fc", 4333),
    "Ross County": ("ross-county-fc", 4333),
    "St Johnstone": ("fc-st-johnstone", 992),
    "St. Johnstone FC": ("fc-st-johnstone", 992),
    "Livingston FC": ("fc-livingston", 3215),
    "Livingston": ("fc-livingston", 3215),
    "Partick Thistle FC": ("partick-thistle-fc", 1113),
    "Partick Thistle": ("partick-thistle-fc", 1113),
    "Falkirk FC": ("fc-falkirk", 1098),
    "Falkirk": ("fc-falkirk", 1098),
    "Hamilton Academical FC": ("hamilton-academical-fc", 1112),
    "Hamilton": ("hamilton-academical-fc", 1112),
    "Raith Rovers FC": ("raith-rovers-fc", 6206),
    "Raith Rovers": ("raith-rovers-fc", 6206),
    "Ayr United FC": ("ayr-united-fc", 6213),
    "Ayr United": ("ayr-united-fc", 6213),
    "Queen of the South FC": ("queen-of-the-south-fc", 6202),
    "Inverness Caledonian Thistle FC": ("inverness-ct", 2285),
    "Inverness CT": ("inverness-ct", 2285),
    "Dunfermline Athletic FC": ("dunfermline-athletic-fc", 1097),
    "Dunfermline": ("dunfermline-athletic-fc", 1097),
    "Kelty Hearts FC": ("kelty-hearts-fc", 32666),
    "Kelty Hearts": ("kelty-hearts-fc", 32666),
    "Cove Rangers FC": ("cove-rangers-fc", 20857),
    "Cove Rangers": ("cove-rangers-fc", 20857),
    "Montrose FC": ("montrose-fc", 6215),
    "Montrose": ("montrose-fc", 6215),
    "Alloa Athletic FC": ("alloa-athletic-fc", 6219),
    "Alloa Athletic": ("alloa-athletic-fc", 6219),
    "Arbroath FC": ("arbroath-fc", 6195),
    "Arbroath": ("arbroath-fc", 6195),
    "Airdrieonians FC": ("airdrieonians-fc", 10283),
    "Airdrieonians": ("airdrieonians-fc", 10283),
    "Stenhousemuir FC": ("stenhousemuir-fc", 9685),
    "Stenhousemuir": ("stenhousemuir-fc", 9685),
    "Stirling Albion FC": ("stirling-albion-fc", 9684),
    "Stirling Albion": ("stirling-albion-fc", 9684),
    "Stranraer FC": ("stranraer-fc", 6222),
    "Stranraer": ("stranraer-fc", 6222),
    "Elgin City FC": ("elgin-city-fc", 6224),
    "Elgin City": ("elgin-city-fc", 6224),
    "Forfar Athletic FC": ("forfar-athletic-fc", 6223),
    "Forfar Athletic": ("forfar-athletic-fc", 6223),
    "Peterhead FC": ("peterhead-fc", 9667),
    "Peterhead": ("peterhead-fc", 9667),
    "Annan Athletic FC": ("annan-athletic-fc", 10282),
    "Annan Athletic": ("annan-athletic-fc", 10282),
    "Dumbarton FC": ("dumbarton-fc", 6217),
    "Dumbarton": ("dumbarton-fc", 6217),
    "Clyde FC": ("clyde-fc", 6208),
    "Clyde": ("clyde-fc", 6208),
    "East Fife FC": ("east-fife-fc", 6220),
    "East Fife": ("east-fife-fc", 6220),
    "Edinburgh City FC": ("edinburgh-city-fc", 20856),
    "Edinburgh City": ("edinburgh-city-fc", 20856),
    "East Kilbride FC": ("east-kilbride-fc", 18629),
    "East Kilbride": ("east-kilbride-fc", 18629),
    "Spartans FC": ("spartans-fc", 25206),
    "Spartans": ("spartans-fc", 25206),
}


def _get_transfermarkt_scraper():
    """Get the TransfermarktScraper from advanced_features_scraper."""
    try:
        from .advanced_features_scraper import TransfermarktScraper
        return TransfermarktScraper()
    except ImportError:
        return None


def _is_cache_valid() -> bool:
    """Check if the injury cache is still valid."""
    global _CACHE_TIMESTAMP
    if _CACHE_TIMESTAMP is None:
        return False
    age = datetime.now() - _CACHE_TIMESTAMP
    return age.total_seconds() < _CACHE_TTL_HOURS * 3600


def get_team_injuries(team_name: str, scraper=None) -> Dict[str, int]:
    """
    Get injury/suspension counts for a team.
    
    Returns dict with:
        - injured_players: count of injured players
        - suspended_players: count of suspended players
        - total_out: total unavailable players
    """
    global _INJURY_CACHE, _CACHE_TIMESTAMP
    
    # Check cache first
    if _is_cache_valid() and team_name in _INJURY_CACHE:
        return _INJURY_CACHE[team_name]
    
    # Get team info from mapping
    team_info = TEAM_TRANSFERMARKT_MAP.get(team_name)
    if not team_info:
        # Try partial match
        for key, value in TEAM_TRANSFERMARKT_MAP.items():
            if team_name.lower() in key.lower() or key.lower() in team_name.lower():
                team_info = value
                break
    
    if not team_info:
        return {"injured_players": 0, "suspended_players": 0, "total_out": 0}
    
    team_slug, team_id = team_info
    
    # Use the scraper if available
    if scraper:
        try:
            injuries = scraper.get_team_injuries(team_name)
            result = {
                "injured_players": len([i for i in injuries if i.injury_type == "Injury"]),
                "suspended_players": len([i for i in injuries if i.injury_type == "Suspension"]),
                "total_out": len(injuries),
            }
            _INJURY_CACHE[team_name] = result
            _CACHE_TIMESTAMP = datetime.now()
            return result
        except Exception as e:
            print(f"    Error fetching injuries for {team_name}: {e}")
    
    return {"injured_players": 0, "suspended_players": 0, "total_out": 0}


def fetch_injuries_for_matches(
    odds_df: pd.DataFrame,
    home_col: str = "home_team",
    away_col: str = "away_team",
) -> pd.DataFrame:
    """
    Fetch current injury/suspension data for all teams in upcoming matches.
    
    Adds columns:
        - home_injured: number of injured home players
        - home_suspended: number of suspended home players  
        - away_injured: number of injured away players
        - away_suspended: number of suspended away players
        - injury_advantage: (away_total_out - home_total_out) positive = home advantage
    
    Args:
        odds_df: DataFrame with upcoming matches (must have home_team, away_team columns)
        home_col: Name of home team column
        away_col: Name of away team column
    
    Returns:
        DataFrame with injury columns added
    """
    if odds_df.empty:
        return odds_df
    
    result = odds_df.copy()
    
    # Initialize columns
    result["home_injured"] = 0
    result["home_suspended"] = 0
    result["away_injured"] = 0
    result["away_suspended"] = 0
    result["injury_advantage"] = 0.0
    
    # Get unique teams
    all_teams = set(result[home_col].unique()) | set(result[away_col].unique())
    
    # Try to get scraper
    scraper = _get_transfermarkt_scraper()
    
    if scraper is None:
        print("Warning: TransfermarktScraper not available. Skipping injury data.")
        return result
    
    print("\n" + "=" * 60)
    print("FETCHING INJURY/SUSPENSION DATA")
    print("=" * 60)
    
    # Fetch injuries for each team
    team_injuries = {}
    teams_found = 0
    
    for team in sorted(all_teams):
        # Check if team is in our mapping
        if team not in TEAM_TRANSFERMARKT_MAP:
            # Try partial match
            found = False
            for key in TEAM_TRANSFERMARKT_MAP:
                if team.lower() in key.lower() or key.lower() in team.lower():
                    found = True
                    break
            if not found:
                continue
        
        injury_data = get_team_injuries(team, scraper)
        if injury_data["total_out"] > 0:
            teams_found += 1
            print(f"  {team}: {injury_data['injured_players']} injured, {injury_data['suspended_players']} suspended")
        team_injuries[team] = injury_data
    
    # Close scraper driver
    try:
        scraper._close_driver()
    except:
        pass
    
    # Apply to DataFrame
    for idx, row in result.iterrows():
        home_team = row[home_col]
        away_team = row[away_col]
        
        home_data = team_injuries.get(home_team, {"injured_players": 0, "suspended_players": 0, "total_out": 0})
        away_data = team_injuries.get(away_team, {"injured_players": 0, "suspended_players": 0, "total_out": 0})
        
        result.at[idx, "home_injured"] = home_data["injured_players"]
        result.at[idx, "home_suspended"] = home_data["suspended_players"]
        result.at[idx, "away_injured"] = away_data["injured_players"]
        result.at[idx, "away_suspended"] = away_data["suspended_players"]
        # Positive = more away players out = home advantage
        result.at[idx, "injury_advantage"] = away_data["total_out"] - home_data["total_out"]
    
    matches_with_data = (result["home_injured"] + result["home_suspended"] + 
                         result["away_injured"] + result["away_suspended"] > 0).sum()
    
    print(f"\nInjury data fetched for {teams_found} teams")
    print(f"Matches with injury data: {matches_with_data}/{len(result)}")
    
    return result


if __name__ == "__main__":
    # Test the injury fetcher
    import pandas as pd
    
    test_matches = pd.DataFrame({
        "home_team": ["Liverpool", "Arsenal", "Bayern München"],
        "away_team": ["Chelsea", "Manchester City", "Borussia Dortmund"],
    })
    
    result = fetch_injuries_for_matches(test_matches)
    print("\nTest results:")
    print(result[["home_team", "away_team", "home_injured", "away_injured", "injury_advantage"]])
