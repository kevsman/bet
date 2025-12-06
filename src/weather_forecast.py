"""
Weather forecast fetcher for upcoming football matches.

Uses the free Open-Meteo API to get weather forecasts for match venues.
No API key required.
"""

import requests
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import pandas as pd
from pathlib import Path

# Stadium coordinates (latitude, longitude) for major teams
# Format: 'team_name': (lat, lon)
VENUE_COORDINATES = {
    # England - Premier League
    'Arsenal': (51.5549, -0.1084),  # Emirates Stadium
    'Aston Villa': (52.5092, -1.8847),  # Villa Park
    'Bournemouth': (50.7352, -1.8384),  # Vitality Stadium
    'Brentford': (51.4907, -0.2886),  # Gtech Community Stadium
    'Brighton': (50.8619, -0.0837),  # Amex Stadium
    'Chelsea': (51.4817, -0.1910),  # Stamford Bridge
    'Crystal Palace': (51.3983, -0.0856),  # Selhurst Park
    'Everton': (53.4389, -2.9664),  # Goodison Park
    'Fulham': (51.4749, -0.2217),  # Craven Cottage
    'Ipswich': (52.0545, 1.1449),  # Portman Road
    'Ipswich Town': (52.0545, 1.1449),
    'Leicester': (52.6203, -1.1422),  # King Power Stadium
    'Leicester City': (52.6203, -1.1422),
    'Liverpool': (53.4308, -2.9608),  # Anfield
    'Man City': (53.4831, -2.2004),  # Etihad Stadium
    'Manchester City': (53.4831, -2.2004),
    'Man United': (53.4631, -2.2913),  # Old Trafford
    'Manchester United': (53.4631, -2.2913),
    'Newcastle': (54.9756, -1.6217),  # St James' Park
    'Newcastle United': (54.9756, -1.6217),
    "Nott'm Forest": (52.9400, -1.1328),  # City Ground
    'Nottingham Forest': (52.9400, -1.1328),
    'Southampton': (50.9058, -1.3909),  # St Mary's Stadium
    'Tottenham': (51.6042, -0.0662),  # Tottenham Hotspur Stadium
    'Tottenham Hotspur': (51.6042, -0.0662),
    'West Ham': (51.5387, -0.0166),  # London Stadium
    'West Ham United': (51.5387, -0.0166),
    'Wolves': (52.5901, -2.1306),  # Molineux
    'Wolverhampton': (52.5901, -2.1306),
    
    # England - Championship
    'Burnley': (53.7890, -2.2303),  # Turf Moor
    'Leeds': (53.7778, -1.5722),  # Elland Road
    'Leeds United': (53.7778, -1.5722),
    'Sunderland': (54.9146, -1.3883),  # Stadium of Light
    'Sheffield United': (53.3703, -1.4708),  # Bramall Lane
    'Sheffield Utd': (53.3703, -1.4708),
    'Middlesbrough': (54.5783, -1.2167),  # Riverside Stadium
    'West Brom': (52.5090, -1.9641),  # The Hawthorns
    'Watford': (51.6500, -0.4017),  # Vicarage Road
    'Norwich': (52.6222, 1.3089),  # Carrow Road
    'Coventry': (52.4489, -1.4956),  # Coventry Building Society Arena
    'Blackburn': (53.7286, -2.4892),  # Ewood Park
    'Stoke': (52.9884, -2.1756),  # bet365 Stadium
    'Hull': (53.7461, -0.3678),  # MKM Stadium
    'Preston': (53.7722, -2.6881),  # Deepdale
    'Luton': (51.8842, -0.4317),  # Kenilworth Road
    'Bristol City': (51.4400, -2.6203),  # Ashton Gate
    'Swansea': (51.6428, -3.9350),  # Swansea.com Stadium
    'Millwall': (51.4861, -0.0508),  # The Den
    'Cardiff': (51.4728, -3.2031),  # Cardiff City Stadium
    'Plymouth': (50.3883, -4.1508),  # Home Park
    'Derby': (52.9147, -1.4472),  # Pride Park
    'QPR': (51.5093, -0.2322),  # Loftus Road
    'Portsmouth': (50.7961, -1.0639),  # Fratton Park
    'Oxford': (51.7161, -1.2081),  # Kassam Stadium
    
    # Germany - Bundesliga
    'Bayern Munich': (48.2188, 11.6247),  # Allianz Arena
    'Dortmund': (51.4926, 7.4519),  # Signal Iduna Park
    'Borussia Dortmund': (51.4926, 7.4519),
    'RB Leipzig': (51.3458, 12.3483),  # Red Bull Arena
    'Leverkusen': (51.0383, 7.0022),  # BayArena
    'Bayer Leverkusen': (51.0383, 7.0022),
    'Ein Frankfurt': (50.0686, 8.6453),  # Deutsche Bank Park
    'Eintracht Frankfurt': (50.0686, 8.6453),
    'Stuttgart': (48.7922, 9.2319),  # MHPArena
    'VfB Stuttgart': (48.7922, 9.2319),
    'Freiburg': (47.9894, 7.8989),  # Europa-Park Stadion
    'SC Freiburg': (47.9894, 7.8989),
    'Hoffenheim': (49.2383, 8.8883),  # PreZero Arena
    'TSG Hoffenheim': (49.2383, 8.8883),
    'Wolfsburg': (52.4319, 10.8039),  # Volkswagen Arena
    'VfL Wolfsburg': (52.4319, 10.8039),
    "M'gladbach": (51.1747, 6.3856),  # Borussia-Park
    "Borussia M'gladbach": (51.1747, 6.3856),
    'Mainz': (49.9844, 8.2244),  # Mewa Arena
    '1. FSV Mainz 05': (49.9844, 8.2244),
    'Union Berlin': (52.4572, 13.5681),  # An der Alten Försterei
    '1. FC Union Berlin': (52.4572, 13.5681),
    'Werder Bremen': (53.0664, 8.8378),  # Weserstadion
    'Augsburg': (48.3236, 10.8864),  # WWK Arena
    'FC Augsburg': (48.3236, 10.8864),
    'Bochum': (51.4900, 7.2364),  # Vonovia Ruhrstadion
    'VfL Bochum': (51.4900, 7.2364),
    'St Pauli': (53.5544, 9.9675),  # Millerntor-Stadion
    'FC St. Pauli': (53.5544, 9.9675),
    'Heidenheim': (48.6728, 10.1417),  # Voith-Arena
    '1. FC Heidenheim': (48.6728, 10.1417),
    'Holstein Kiel': (54.3489, 10.1228),  # Holstein-Stadion
    
    # Spain - La Liga
    'Real Madrid': (40.4531, -3.6883),  # Santiago Bernabéu
    'Barcelona': (41.3809, 2.1228),  # Camp Nou (temporary: Montjuïc)
    'Ath Madrid': (40.4361, -3.5994),  # Metropolitano
    'Atletico Madrid': (40.4361, -3.5994),
    'Sevilla': (37.3840, -5.9706),  # Ramón Sánchez-Pizjuán
    'Ath Bilbao': (43.2642, -2.9494),  # San Mamés
    'Athletic Club Bilbao': (43.2642, -2.9494),
    'Sociedad': (43.3017, -1.9736),  # Reale Arena
    'Real Sociedad': (43.3017, -1.9736),
    'Betis': (37.3567, -5.9817),  # Benito Villamarín
    'Real Betis': (37.3567, -5.9817),
    'Villarreal': (39.9442, -0.1036),  # Estadio de la Cerámica
    'Valencia': (39.4747, -0.3583),  # Mestalla
    'Celta': (42.2119, -8.7397),  # Balaídos
    'Celta Vigo': (42.2119, -8.7397),
    'Osasuna': (42.7967, -1.6369),  # El Sadar
    'Getafe': (40.3256, -3.7147),  # Coliseum Alfonso Pérez
    'Mallorca': (39.5903, 2.6308),  # Son Moix
    'Girona': (41.9608, 2.8283),  # Montilivi
    'Las Palmas': (28.1003, -15.4567),  # Gran Canaria
    'Rayo Vallecano': (40.3919, -3.6589),  # Vallecas
    'Alaves': (42.8372, -2.6878),  # Mendizorroza
    'Leganes': (40.3258, -3.7594),  # Butarque
    'Valladolid': (41.6444, -4.7611),  # José Zorrilla
    'Real Valladolid': (41.6444, -4.7611),
    'Espanyol': (41.3478, 2.0756),  # RCDE Stadium
    
    # Spain - Segunda
    'Elche': (38.2653, -0.6611),  # Martínez Valero
    'Granada': (37.1283, -3.5953),  # Nuevo Los Cármenes
    'Eibar': (43.1817, -2.4756),  # Ipurua
    'Huesca': (42.1350, -0.4081),  # El Alcoraz
    'Sporting Gijon': (43.5361, -5.6372),  # El Molinón
    'Almeria': (36.8403, -2.4344),  # Power Horse Stadium
    'Cadiz': (36.5028, -6.2736),  # Nuevo Mirandilla
    'Tenerife': (28.4653, -16.2533),  # Heliodoro Rodríguez López
    'Zaragoza': (41.6364, -0.9017),  # La Romareda
    'Levante': (39.4944, -0.3639),  # Ciutat de València
    'Racing Santander': (43.4722, -3.7917),  # El Sardinero
    'Oviedo': (43.3647, -5.8675),  # Carlos Tartiere
    'Albacete': (38.9917, -1.8600),  # Carlos Belmonte
    'Ceuta': (35.8894, -5.3167),  # Alfonso Murube
    
    # Italy - Serie A
    'Inter': (45.4781, 9.1239),  # San Siro
    'Milan': (45.4781, 9.1239),  # San Siro
    'AC Milan': (45.4781, 9.1239),
    'Juventus': (45.1097, 7.6411),  # Allianz Stadium
    'Napoli': (40.8279, 14.1931),  # Diego Armando Maradona
    'SSC Napoli': (40.8279, 14.1931),
    'Roma': (41.9339, 12.4547),  # Olimpico
    'AS Roma': (41.9339, 12.4547),
    'Lazio': (41.9339, 12.4547),  # Olimpico
    'SS Lazio': (41.9339, 12.4547),
    'Atalanta': (45.7089, 9.6808),  # Gewiss Stadium
    'Atalanta BC': (45.7089, 9.6808),
    'Fiorentina': (43.7806, 11.2822),  # Artemio Franchi
    'ACF Fiorentina': (43.7806, 11.2822),
    'Bologna': (44.4922, 11.3097),  # Renato Dall'Ara
    'Bologna FC 1909': (44.4922, 11.3097),
    'Torino': (45.0419, 7.6500),  # Olimpico Grande Torino
    'Torino FC': (45.0419, 7.6500),
    'Udinese': (46.0817, 13.1997),  # Bluenergy Stadium
    'Udinese Calcio': (46.0817, 13.1997),
    'Genoa': (44.4167, 8.9525),  # Luigi Ferraris
    'Genoa CFC': (44.4167, 8.9525),
    'Sampdoria': (44.4167, 8.9525),  # Luigi Ferraris
    'Cagliari': (39.1997, 9.1378),  # Unipol Domus
    'Cagliari Calcio': (39.1997, 9.1378),
    'Verona': (45.4353, 10.9686),  # Bentegodi
    'Hellas Verona': (45.4353, 10.9686),
    'Parma': (44.7953, 10.3381),  # Ennio Tardini
    'Parma Calcio 1913': (44.7953, 10.3381),
    'Lecce': (40.3606, 18.1719),  # Via del Mare
    'US Lecce': (40.3606, 18.1719),
    'Empoli': (43.7267, 10.9519),  # Carlo Castellani
    'Empoli FC': (43.7267, 10.9519),
    'Monza': (45.5836, 9.3031),  # U-Power Stadium
    'AC Monza': (45.5836, 9.3031),
    'Como': (45.8150, 9.0894),  # Giuseppe Sinigaglia
    'Como 1907': (45.8150, 9.0894),
    'Venezia': (45.4564, 12.3478),  # Pier Luigi Penzo
    'Venezia FC': (45.4564, 12.3478),
    
    # France - Ligue 1
    'Paris SG': (48.8414, 2.2530),  # Parc des Princes
    'Paris Saint Germain': (48.8414, 2.2530),
    'Marseille': (43.2697, 5.3958),  # Vélodrome
    'Olympique Marseille': (43.2697, 5.3958),
    'Lyon': (45.7653, 4.9822),  # Groupama Stadium
    'Olympique Lyon': (45.7653, 4.9822),
    'Monaco': (43.7275, 7.4156),  # Louis II
    'AS Monaco': (43.7275, 7.4156),
    'Lille': (50.6119, 3.1306),  # Pierre Mauroy
    'LOSC Lille': (50.6119, 3.1306),
    'Nice': (43.7050, 7.1925),  # Allianz Riviera
    'OGC Nice': (43.7050, 7.1925),
    'Lens': (50.4328, 2.8153),  # Bollaert-Delelis
    'RC Lens': (50.4328, 2.8153),
    'Rennes': (48.1075, -1.7128),  # Roazhon Park
    'Stade Rennais': (48.1075, -1.7128),
    'Strasbourg': (48.5600, 7.7550),  # Meinau
    'RC Strasbourg': (48.5600, 7.7550),
    'Nantes': (47.2558, -1.5250),  # Beaujoire
    'FC Nantes': (47.2558, -1.5250),
    'Brest': (48.4028, -4.4617),  # Francis-Le Blé
    'Stade Brestois 29': (48.4028, -4.4617),
    'Toulouse': (43.5833, 1.4344),  # Stadium de Toulouse
    'Toulouse FC': (43.5833, 1.4344),
    'Reims': (49.2467, 4.0250),  # Auguste-Delaune
    'Stade de Reims': (49.2467, 4.0250),
    'Montpellier': (43.6222, 3.8119),  # Mosson
    'Montpellier HSC': (43.6222, 3.8119),
    'Le Havre': (49.4986, 0.1689),  # Océane
    'Le Havre AC': (49.4986, 0.1689),
    'St Etienne': (45.4608, 4.3903),  # Geoffroy-Guichard
    'AS Saint-Étienne': (45.4608, 4.3903),
    'Auxerre': (47.7917, 3.5872),  # Abbé-Deschamps
    'AJ Auxerre': (47.7917, 3.5872),
    'Angers': (47.4606, -0.5306),  # Raymond Kopa
    'Angers SCO': (47.4606, -0.5306),
    
    # France - Ligue 2
    'Pau FC': (43.3117, -0.3606),  # Nouste Camp
    
    # Netherlands - Eredivisie
    'Ajax': (52.3142, 4.9419),  # Johan Cruyff Arena
    'PSV': (51.4417, 5.4675),  # Philips Stadion
    'PSV Eindhoven': (51.4417, 5.4675),
    'Feyenoord': (51.8939, 4.5231),  # De Kuip
    'AZ Alkmaar': (52.6131, 4.7408),  # AFAS Stadion
    'Twente': (52.2367, 6.8378),  # De Grolsch Veste
    'FC Twente': (52.2367, 6.8378),
    'Utrecht': (52.0786, 5.1456),  # Stadion Galgenwaard
    'FC Utrecht': (52.0786, 5.1456),
    'Vitesse': (51.9633, 5.8933),  # GelreDome
    'Heerenveen': (52.9600, 5.9417),  # Abe Lenstra Stadion
    'FC Groningen': (53.2067, 6.5878),  # Euroborg
    'Sparta Rotterdam': (51.8942, 4.4311),  # Sparta Stadion Het Kasteel
    'NEC Nijmegen': (51.8392, 5.8683),  # Goffertstadion
    'Go Ahead Eagles': (52.2619, 6.1578),  # De Adelaarshorst
    'NAC Breda': (51.5875, 4.7833),  # Rat Verlegh Stadion
    'Heracles': (52.2214, 6.8203),  # Erve Asito
    'Waalwijk': (51.6833, 5.0500),  # Mandemakers Stadion
    'RKC Waalwijk': (51.6833, 5.0500),
    'Almere City': (52.3733, 5.2278),  # Yanmar Stadion
    'Fortuna Sittard': (50.9933, 5.8583),  # Fortuna Sittard Stadion
    'Willem II': (51.5550, 5.0806),  # Koning Willem II Stadion
    'PEC Zwolle': (52.5167, 6.1167),  # MAC³PARK stadion
    
    # Turkey - Süper Lig
    'Galatasaray': (41.1036, 28.9914),  # NEF Stadyumu
    'Fenerbahce': (40.9878, 29.0369),  # Şükrü Saracoğlu
    'Besiktas': (41.0394, 28.9947),  # Vodafone Park
    'Trabzonspor': (40.9944, 39.7786),  # Papara Park
    'Basaksehir': (41.1211, 28.8089),  # Başakşehir Fatih Terim
    'Antalyaspor': (36.8900, 30.6833),  # Corendon Airlines Park
    'Konyaspor': (37.8667, 32.4833),  # Konya Büyükşehir Stadyumu
    'Alanyaspor': (36.5422, 32.0022),  # Bahçeşehir Okulları Stadyumu
    'Rizespor': (41.0208, 40.5178),  # Çaykur Didi Stadyumu
    'Sivasspor': (39.7508, 37.0156),  # Yeni 4 Eylül Stadyumu
    'Kasimpasa': (41.0478, 28.9692),  # Recep Tayyip Erdoğan Stadyumu
    'Gaziantep': (37.0769, 37.3797),  # Kalyon Stadyumu
    'Kayserispor': (38.7206, 35.5028),  # RHG Enertürk Enerji Stadyumu
    'Ankaragucu': (39.8667, 32.8333),  # Eryaman Stadyumu
    'Samsunspor': (41.2889, 36.3306),  # Samsun 19 Mayıs Stadyumu
    'Goztepe': (38.4361, 27.1367),  # Gürsel Aksel Stadyumu
    'Hatayspor': (36.2028, 36.1606),  # Yeni Hatay Stadyumu
    'Eyupspor': (41.0478, 28.9339),  # Alibeyköy Stadyumu
    'Bodrumspor': (37.0383, 27.4306),  # Bodrum Stadyumu
    
    # Portugal - Primeira Liga
    'Benfica': (38.7528, -9.1847),  # Estádio da Luz
    'SL Benfica': (38.7528, -9.1847),
    'Porto': (41.1617, -8.5833),  # Estádio do Dragão
    'FC Porto': (41.1617, -8.5833),
    'Sporting CP': (38.7614, -9.1608),  # Estádio José Alvalade
    'Braga': (41.5622, -8.4297),  # Estádio Municipal de Braga
    
    # Scotland - Premiership
    'Celtic': (55.8497, -4.2056),  # Celtic Park
    'Rangers': (55.8533, -4.3092),  # Ibrox
    'Hearts': (55.9392, -3.2322),  # Tynecastle
    'Aberdeen': (57.1597, -2.0883),  # Pittodrie
    'Hibernian': (55.9617, -3.1656),  # Easter Road
    'Dundee': (56.4750, -2.9689),  # Dens Park
    'Dundee United': (56.4747, -2.9656),  # Tannadice
    'Kilmarnock': (55.6047, -4.5081),  # Rugby Park
    'St Mirren': (55.8539, -4.4281),  # SMiSA Stadium
    'Motherwell': (55.7800, -3.9817),  # Fir Park
    'Ross County': (57.5961, -4.4228),  # Global Energy Stadium
    'St Johnstone': (56.4089, -3.4747),  # McDiarmid Park
    
    # Belgium - Pro League
    'Club Brugge': (51.1933, 3.1806),  # Jan Breydel
    'Anderlecht': (50.8344, 4.2981),  # Lotto Park
    'Genk': (50.9906, 5.5336),  # Cegeka Arena
    'Antwerp': (51.1883, 4.4161),  # Bosuilstadion
    'St. Gilloise': (50.8044, 4.3328),  # Stade Joseph Marien
    'Gent': (51.0347, 3.7050),  # Ghelamco Arena
    'Standard Liege': (50.6103, 5.5519),  # Stade de Sclessin
    'Cercle Brugge': (51.1933, 3.1806),  # Jan Breydel
    'Mechelen': (51.0247, 4.4756),  # AFAS-stadion
    'Westerlo': (51.0917, 4.9194),  # Het Kuipje
    'Charleroi': (50.4114, 4.4536),  # Stade du Pays de Charleroi
    'OH Leuven': (50.8836, 4.7042),  # Den Dreef
    'Kortrijk': (50.8133, 3.2444),  # Guldensporenstadion
    'STVV': (50.9778, 5.4722),  # Stayen
    'Beerschot': (51.1833, 4.4167),  # Olympisch Stadion
    'Dender': (50.9247, 4.0797),  # Stadion Dender
}


def get_weather_forecast(lat: float, lon: float, date: str, hour: int = 15) -> Optional[Dict]:
    """
    Fetch weather forecast from Open-Meteo API.
    
    Args:
        lat: Latitude of the venue
        lon: Longitude of the venue
        date: Date in YYYY-MM-DD format
        hour: Hour of the match (24h format)
    
    Returns:
        Dictionary with weather data or None if unavailable
    """
    try:
        # Open-Meteo API - free, no API key needed
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,wind_speed_10m,precipitation_probability,precipitation",
            "start_date": date,
            "end_date": date,
            "timezone": "auto"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Get the hourly data for the match time
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        
        # Find the index for the requested hour
        target_time = f"{date}T{hour:02d}:00"
        if target_time in times:
            idx = times.index(target_time)
        else:
            # Use closest hour
            idx = min(hour, len(times) - 1) if times else 0
        
        if not times:
            return None
            
        temp = hourly.get("temperature_2m", [None])[idx]
        wind = hourly.get("wind_speed_10m", [None])[idx]
        precip_prob = hourly.get("precipitation_probability", [None])[idx]
        precip = hourly.get("precipitation", [None])[idx]
        
        return {
            "weather_temp": temp,
            "weather_wind_speed": wind,
            "weather_precip_prob": precip_prob,
            "weather_precipitation": precip,
            "weather_is_cold": 1 if temp is not None and temp < 5 else 0,
            "weather_is_hot": 1 if temp is not None and temp > 28 else 0,
            "weather_is_rainy": 1 if precip is not None and precip > 1 else 0,
        }
        
    except Exception as e:
        print(f"    Weather fetch error: {e}")
        return None


def get_venue_coordinates(team_name: str) -> Optional[Tuple[float, float]]:
    """Get venue coordinates for a team."""
    # Try exact match first
    if team_name in VENUE_COORDINATES:
        return VENUE_COORDINATES[team_name]
    
    # Try case-insensitive match
    team_lower = team_name.lower()
    for key, coords in VENUE_COORDINATES.items():
        if key.lower() == team_lower:
            return coords
    
    # Try partial match
    for key, coords in VENUE_COORDINATES.items():
        if team_lower in key.lower() or key.lower() in team_lower:
            return coords
    
    return None


def fetch_weather_for_matches(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch weather forecasts for all upcoming matches.
    
    Args:
        matches_df: DataFrame with 'home_team', 'date', 'time' columns
    
    Returns:
        DataFrame with weather columns added
    """
    print("\n============================================================")
    print("FETCHING WEATHER FORECASTS")
    print("============================================================")
    
    weather_data = []
    teams_without_coords = set()
    
    for idx, row in matches_df.iterrows():
        home_team = row['home_team']
        match_date = row['date']
        match_time = row.get('time', '15:00')
        
        # Parse hour from time
        try:
            hour = int(match_time.split(':')[0])
        except:
            hour = 15
        
        # Get venue coordinates
        coords = get_venue_coordinates(home_team)
        
        if coords is None:
            teams_without_coords.add(home_team)
            weather_data.append({
                "weather_temp": None,
                "weather_wind_speed": None,
                "weather_precip_prob": None,
                "weather_precipitation": None,
                "weather_is_cold": 0,
                "weather_is_hot": 0,
                "weather_is_rainy": 0,
            })
            continue
        
        lat, lon = coords
        weather = get_weather_forecast(lat, lon, str(match_date), hour)
        
        if weather:
            print(f"  {home_team}: {weather['weather_temp']:.1f}°C, wind {weather['weather_wind_speed']:.1f} km/h")
            weather_data.append(weather)
        else:
            weather_data.append({
                "weather_temp": None,
                "weather_wind_speed": None,
                "weather_precip_prob": None,
                "weather_precipitation": None,
                "weather_is_cold": 0,
                "weather_is_hot": 0,
                "weather_is_rainy": 0,
            })
    
    # Create weather DataFrame
    weather_df = pd.DataFrame(weather_data)
    
    # Add weather columns to matches
    result = pd.concat([matches_df.reset_index(drop=True), weather_df], axis=1)
    
    # Summary
    with_weather = weather_df['weather_temp'].notna().sum()
    print(f"\nWeather fetched for {with_weather}/{len(matches_df)} matches")
    
    if teams_without_coords:
        print(f"Teams without venue coordinates: {len(teams_without_coords)}")
        for team in sorted(teams_without_coords)[:10]:
            print(f"  - {team}")
        if len(teams_without_coords) > 10:
            print(f"  ... and {len(teams_without_coords) - 10} more")
    
    return result


if __name__ == "__main__":
    # Test with sample data
    test_df = pd.DataFrame({
        'home_team': ['Chelsea', 'Bayern Munich', 'Real Madrid', 'Unknown Team'],
        'date': ['2025-12-07', '2025-12-07', '2025-12-07', '2025-12-07'],
        'time': ['15:30', '15:30', '21:00', '15:00']
    })
    
    result = fetch_weather_for_matches(test_df)
    print("\nResult:")
    print(result[['home_team', 'weather_temp', 'weather_wind_speed', 'weather_is_cold']].to_string())
