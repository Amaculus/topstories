"""
Charlotte API Odds Fetcher
Fetches live odds from charlotte.rotogrinders.com (ScoresAndOdds)
"""

import requests
from typing import Optional, Dict, List, Any
from datetime import datetime


class CharlotteOddsFetcher:
    """Fetch odds data from Charlotte/RotoGrinders API"""

    API_KEY = "ZV03bez9ie16w553FeuM8Z49Djw67pt6"

    # Sport mapping for Charlotte API endpoints
    SPORT_PATHS = {
        "nfl": "nfl",
        "ncaaf": "ncaaf",  # Charlotte API uses "ncaaf" not "cfb"
        "ncaab": "ncaab",  # Assuming same pattern for basketball
        "nba": "nba",
        "mlb": "mlb",
        "nhl": "nhl"
    }
    
    # Sportsbook name mapping (friendly names)
    SPORTSBOOK_NAMES = {
        "draftkings": "DraftKings",
        "fanduel": "FanDuel", 
        "betmgm": "BetMGM",
        "caesars": "Caesars",
        "bet365": "Bet365",
        "hardrock": "Hard Rock",
        "fanatics": "Fanatics",
        "riverscasino": "Rivers Casino"
    }
    
    def __init__(self, sport: str = "nfl"):
        """
        Initialize odds fetcher for a specific sport

        Args:
            sport: Sport code (nfl, ncaaf, ncaab, nba, mlb, nhl)
        """
        self.sport = sport.lower()
        self.sport_path = self.SPORT_PATHS.get(self.sport, "nfl")
        self.base_url = f"https://charlotte.rotogrinders.com/sports/{self.sport_path}/extended"
        self.games_cache = None
        self.cache_timestamp = None
        
    def fetch_week_odds(self, week: str = "2025-reg-13") -> Dict[str, Any]:
        """
        Fetch odds for an entire week
        
        Args:
            week: Week identifier (e.g., "2025-reg-13", "2025-post-1")
            
        Returns:
            Dict with full API response
        """
        params = {
            "role": "scoresandodds",
            "week": week,
            "key": self.API_KEY
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Cache the games
            self.games_cache = data.get('data', [])
            self.cache_timestamp = datetime.now()
            
            return data
        except requests.exceptions.RequestException as e:
            print(f"Error fetching odds: {e}")
            return {}
            
    def get_all_games(self, week: str = "2025-reg-13") -> List[Dict[str, Any]]:
        """Get list of all games with odds for a week"""
        if not self.games_cache:
            self.fetch_week_odds(week)
        return self.games_cache or []
    
    def find_game_by_teams(self, away_team: str, home_team: str) -> Optional[Dict[str, Any]]:
        """
        Find a game by team names/keys
        
        Args:
            away_team: Away team key (e.g., "KC", "GB", "Chiefs", "Packers")
            home_team: Home team key
            
        Returns:
            Game dict if found, None otherwise
        """
        if not self.games_cache:
            self.fetch_week_odds()
            
        away_lower = away_team.lower()
        home_lower = home_team.lower()
        
        for game in self.games_cache:
            away = game['away']
            home = game['home']
            
            # Match by key or mascot
            away_match = (
                away['key'].lower() == away_lower or 
                away['mascot'].lower() == away_lower
            )
            home_match = (
                home['key'].lower() == home_lower or 
                home['mascot'].lower() == home_lower
            )
            
            if away_match and home_match:
                return game
                
        return None
    
    def find_game_by_id(self, game_id: int) -> Optional[Dict[str, Any]]:
        """Find a game by Charlotte game ID"""
        if not self.games_cache:
            self.fetch_week_odds()
            
        for game in self.games_cache:
            if game['id'] == game_id:
                return game
        return None
    
    def get_spread_odds(
        self, 
        game: Dict[str, Any], 
        sportsbook: str = "draftkings"
    ) -> Optional[Dict[str, Any]]:
        """
        Get spread odds for a specific sportsbook
        
        Args:
            game: Game dict from Charlotte API
            sportsbook: Sportsbook key (e.g., "draftkings", "fanduel")
            
        Returns:
            Dict with spread odds: {
                'line': 2.5,
                'favorite': 'home',
                'away_odds': -105,
                'home_odds': -115,
                'away_team': 'Packers',
                'home_team': 'Lions'
            }
        """
        try:
            odds = game['odds']['current']['spread']
            comparison = odds.get('comparison', {})
            
            # Get odds for specified book (fallback to default)
            book_odds = comparison.get(sportsbook.lower(), odds)
            
            return {
                'line': book_odds['value'],
                'favorite': book_odds['favorite'],
                'away_odds': book_odds['away'],
                'home_odds': book_odds['home'],
                'away_team': game['away']['mascot'],
                'home_team': game['home']['mascot'],
                'away_key': game['away']['key'],
                'home_key': game['home']['key']
            }
        except (KeyError, TypeError):
            return None
    
    def get_moneyline_odds(
        self, 
        game: Dict[str, Any], 
        sportsbook: str = "draftkings"
    ) -> Optional[Dict[str, Any]]:
        """Get moneyline odds for a specific sportsbook"""
        try:
            odds = game['odds']['current']['moneyline']
            comparison = odds.get('comparison', {})
            book_odds = comparison.get(sportsbook.lower(), odds)
            
            return {
                'away_odds': book_odds['away'],
                'home_odds': book_odds['home'],
                'favorite': book_odds['favorite'],
                'away_team': game['away']['mascot'],
                'home_team': game['home']['mascot'],
                'away_key': game['away']['key'],
                'home_key': game['home']['key']
            }
        except (KeyError, TypeError):
            return None
    
    def get_total_odds(
        self, 
        game: Dict[str, Any], 
        sportsbook: str = "draftkings"
    ) -> Optional[Dict[str, Any]]:
        """Get over/under total odds for a specific sportsbook"""
        try:
            odds = game['odds']['current']['total']
            comparison = odds.get('comparison', {})
            book_odds = comparison.get(sportsbook.lower(), odds)
            
            return {
                'line': book_odds['value'],
                'over_odds': book_odds['over'],
                'under_odds': book_odds['under'],
                'favorite': book_odds['favorite']
            }
        except (KeyError, TypeError):
            return None
    
    def format_spread_text(
        self, 
        spread_odds: Dict[str, Any], 
        sportsbook: str = "draftkings"
    ) -> str:
        """
        Format spread odds as human-readable text
        
        Example: "Lions -2.5 (-115) vs Packers +2.5 (-105) at DraftKings"
        """
        if not spread_odds:
            return "Odds unavailable"
            
        line = spread_odds['line']
        favorite = spread_odds['favorite']
        away_team = spread_odds['away_team']
        home_team = spread_odds['home_team']
        away_odds = spread_odds['away_odds']
        home_odds = spread_odds['home_odds']
        
        book_name = self.SPORTSBOOK_NAMES.get(sportsbook.lower(), sportsbook.title())
        
        if favorite == 'home':
            return (
                f"{home_team} -{line} ({home_odds:+d}) vs "
                f"{away_team} +{line} ({away_odds:+d}) at {book_name}"
            )
        else:
            return (
                f"{away_team} -{line} ({away_odds:+d}) vs "
                f"{home_team} +{line} ({home_odds:+d}) at {book_name}"
            )
    
    def format_moneyline_text(
        self, 
        ml_odds: Dict[str, Any], 
        sportsbook: str = "draftkings"
    ) -> str:
        """
        Format moneyline odds as text
        
        Example: "Lions -142 vs Packers +120 at DraftKings"
        """
        if not ml_odds:
            return "Odds unavailable"
            
        book_name = self.SPORTSBOOK_NAMES.get(sportsbook.lower(), sportsbook.title())
        
        return (
            f"{ml_odds['home_team']} {ml_odds['home_odds']:+d} vs "
            f"{ml_odds['away_team']} {ml_odds['away_odds']:+d} at {book_name}"
        )
    
    def format_total_text(
        self, 
        total_odds: Dict[str, Any],
        sportsbook: str = "draftkings"
    ) -> str:
        """
        Format over/under total as text
        
        Example: "Over/Under 48.5 (O: -110 / U: -110) at DraftKings"
        """
        if not total_odds:
            return "Odds unavailable"
            
        book_name = self.SPORTSBOOK_NAMES.get(sportsbook.lower(), sportsbook.title())
        
        return (
            f"Over/Under {total_odds['line']} "
            f"(O: {total_odds['over_odds']:+d} / U: {total_odds['under_odds']:+d}) "
            f"at {book_name}"
        )
    
    def get_all_odds_for_game(
        self, 
        game: Dict[str, Any], 
        sportsbook: str = "draftkings"
    ) -> Dict[str, str]:
        """
        Get all formatted odds for a game at a specific sportsbook
        
        Returns:
            Dict with 'spread', 'moneyline', 'total' keys
        """
        spread = self.get_spread_odds(game, sportsbook)
        moneyline = self.get_moneyline_odds(game, sportsbook)
        total = self.get_total_odds(game, sportsbook)
        
        return {
            'spread': self.format_spread_text(spread, sportsbook),
            'moneyline': self.format_moneyline_text(moneyline, sportsbook),
            'total': self.format_total_text(total, sportsbook),
            'spread_raw': spread,
            'moneyline_raw': moneyline,
            'total_raw': total
        }
    
    def get_available_sportsbooks(self, game: Dict[str, Any]) -> List[str]:
        """Get list of available sportsbooks for a game"""
        try:
            comparison = game['odds']['current']['spread'].get('comparison', {})
            return list(comparison.keys())
        except (KeyError, TypeError):
            return []


# Convenience function for quick usage
def get_game_odds(away_team: str, home_team: str, sportsbook: str = "draftkings", week: str = "2025-reg-13", sport: str = "nfl"):
    """
    Quick function to get odds for a game

    Usage:
        odds = get_game_odds("Chiefs", "Cowboys", "draftkings", sport="nfl")
        odds = get_game_odds("Georgia", "Alabama", "draftkings", sport="ncaaf")
        print(odds['spread'])
    """
    fetcher = CharlotteOddsFetcher(sport=sport)
    fetcher.fetch_week_odds(week)
    game = fetcher.find_game_by_teams(away_team, home_team)

    if not game:
        return None

    return fetcher.get_all_odds_for_game(game, sportsbook)


if __name__ == "__main__":
    # Test the fetcher
    fetcher = CharlotteOddsFetcher(sport="nfl")
    fetcher.fetch_week_odds("2025-reg-13")
    
    # Test with Packers @ Lions
    game = fetcher.find_game_by_teams("Packers", "Lions")
    if game:
        print(f"Found game: {game['away']['mascot']} @ {game['home']['mascot']}")
        print(f"Game ID: {game['id']}")
        
        # Get odds for DraftKings
        odds_dk = fetcher.get_all_odds_for_game(game, "draftkings")
        print(f"\n=== DraftKings ===")
        print(f"Spread: {odds_dk['spread']}")
        print(f"Moneyline: {odds_dk['moneyline']}")
        print(f"Total: {odds_dk['total']}")
        
        # Get odds for FanDuel
        odds_fd = fetcher.get_all_odds_for_game(game, "fanduel")
        print(f"\n=== FanDuel ===")
        print(f"Spread: {odds_fd['spread']}")
        print(f"Moneyline: {odds_fd['moneyline']}")
        print(f"Total: {odds_fd['total']}")
        
        # Show available books
        books = fetcher.get_available_sportsbooks(game)
        print(f"\nAvailable sportsbooks: {', '.join(books)}")