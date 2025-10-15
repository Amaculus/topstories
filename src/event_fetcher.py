# src/event_fetcher.py
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from zoneinfo import ZoneInfo

def get_games_for_date(sport: str = "nfl", target_date: datetime = None) -> List[Dict]:
    """Fetch all games for a specific date."""
    try:
        sport_map = {
            "nfl": "football/nfl",
            "nba": "basketball/nba",
            "mlb": "baseball/mlb",
            "nhl": "hockey/nhl",
            "ncaaf": "football/college-football",
            "ncaab": "basketball/mens-college-basketball",
        }
        
        sport_path = sport_map.get(sport.lower())
        if not sport_path:
            return []
        
        # Format date for ESPN API (YYYYMMDD)
        if target_date is None:
            target_date = datetime.now(ZoneInfo("America/New_York"))
        
        date_str = target_date.strftime("%Y%m%d")
        url = f"http://site.api.espn.com/apis/site/v2/sports/{sport_path}/scoreboard?dates={date_str}"
        
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        events = data.get("events", [])
        if not events:
            return []
        
        games = []
        for game in events:
            competitions = game.get("competitions", [{}])[0]
            competitors = competitions.get("competitors", [])
            
            if len(competitors) < 2:
                continue
            
            # Find home/away
            home = competitors[0] if competitors[0].get("homeAway") == "home" else competitors[1]
            away = competitors[1] if competitors[0].get("homeAway") == "home" else competitors[0]
            
            # Get broadcast info
            broadcasts = competitions.get("broadcasts", [])
            network = broadcasts[0].get("names", [""])[0] if broadcasts else ""
            
            # Parse game time
            game_time = game.get("date", "")
            try:
                dt = datetime.fromisoformat(game_time.replace("Z", "+00:00"))
                dt_et = dt.astimezone(ZoneInfo("America/New_York"))
            except:
                dt_et = None
            
            games.append({
                "id": game.get("id", ""),
                "home_team": home.get("team", {}).get("displayName", ""),
                "away_team": away.get("team", {}).get("displayName", ""),
                "start_time": game_time,
                "start_time_et": dt_et,
                "network": network,
                "headline": game.get("name", ""),
                "short_name": game.get("shortName", ""),
                "sport": sport.upper(),
            })
        
        return games
    
    except Exception as e:
        print(f"Failed to fetch {sport} games: {e}")
        return []


def filter_prime_time_games(games: List[Dict]) -> List[Dict]:
    """Filter for prime time games (evening games after 6 PM ET)."""
    if not games:
        return []
    
    prime_time = []
    
    for game in games:
        dt_et = game.get("start_time_et")
        if not dt_et:
            continue
        
        hour = dt_et.hour
        
        # Prime time = 6 PM or later (18:00+)
        # This catches SNF (8:20 PM), MNF (8:15 PM), TNF (8:15 PM)
        if hour >= 18:
            prime_time.append(game)
    
    # Sort by time (latest first for most premium slots)
    prime_time.sort(key=lambda g: g.get("start_time_et", datetime.min), reverse=True)
    
    return prime_time


def get_featured_game(sport: str = "nfl", target_date: datetime = None) -> Optional[Dict]:
    """Fetch featured game (prime time preferred) for a sport on a specific date."""
    games = get_games_for_date(sport, target_date)
    
    if not games:
        return None
    
    # Try to get prime time game first
    prime_games = filter_prime_time_games(games)
    if prime_games:
        return prime_games[0]  # Return latest prime time game
    
    # Fallback to first game of the day
    return games[0]


def format_event_for_prompt(game: Optional[Dict], reference_date: datetime = None) -> str:
    """Format game data into natural text for prompts."""
    if not game:
        return ""
    
    try:
        dt_et = game.get("start_time_et")
        if not dt_et:
            return f"{game.get('away_team', '')} vs. {game.get('home_team', '')}"
        
        # Determine day context relative to reference date
        if reference_date is None:
            reference_date = datetime.now(ZoneInfo("America/New_York"))
        
        ref_date = reference_date.date()
        game_date = dt_et.date()
        
        if game_date == ref_date:
            day_context = "tonight"
        elif game_date == (ref_date + timedelta(days=1)):
            day_context = "tomorrow night"
        else:
            day_context = dt_et.strftime("%A night")
        
        time_str = dt_et.strftime("%I:%M %p ET").lstrip("0")
        
        # Determine if it's a marquee game
        hour = dt_et.hour
        if hour >= 20:  # 8 PM or later
            if game.get("sport") == "NFL":
                if dt_et.weekday() == 6:  # Sunday
                    marquee = "Sunday Night Football"
                elif dt_et.weekday() == 0:  # Monday
                    marquee = "Monday Night Football"
                elif dt_et.weekday() == 3:  # Thursday
                    marquee = "Thursday Night Football"
                else:
                    marquee = None
            else:
                marquee = None
        else:
            marquee = None
        
        # Build context string
        parts = [f"{game['away_team']} vs. {game['home_team']}"]
        
        if marquee:
            parts.append(f"on {marquee}")
        
        parts.append(day_context)
        
        if game.get('network'):
            parts.append(f"at {time_str} on {game['network']}")
        else:
            parts.append(f"at {time_str}")
        
        return " ".join(parts)
    
    except Exception as e:
        print(f"Failed to format event: {e}")
        return f"{game.get('away_team', '')} vs. {game.get('home_team', '')}"


def format_game_for_dropdown(game: Dict) -> str:
    """Format game for display in dropdown selector."""
    dt_et = game.get("start_time_et")
    if dt_et:
        time_str = dt_et.strftime("%I:%M %p ET").lstrip("0")
        return f"{game['away_team']} @ {game['home_team']} - {time_str} ({game.get('network', 'TBD')})"
    return f"{game['away_team']} @ {game['home_team']}"