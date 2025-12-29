"""
Test CFB integration end-to-end
"""

from src.odds_fetcher import CharlotteOddsFetcher

print("Testing CFB Integration...")
print("=" * 70)

# Test NCAAF odds fetcher
print("\n1. Initializing CFB odds fetcher...")
fetcher = CharlotteOddsFetcher(sport="ncaaf")
print(f"   Sport: {fetcher.sport}")
print(f"   API URL: {fetcher.base_url}")

# Fetch current week odds
print("\n2. Fetching week 14 odds...")
result = fetcher.fetch_week_odds("2025-reg-14")

if result:
    games = fetcher.games_cache
    print(f"   SUCCESS - Found {len(games)} games")

    if games:
        print("\n3. Sample games:")
        for i, game in enumerate(games[:5]):  # Show first 5 games
            away = game.get('away', {})
            home = game.get('home', {})
            print(f"   {i+1}. {away.get('mascot', 'N/A')} @ {home.get('mascot', 'N/A')}")

        # Test finding a game
        print(f"\n4. Testing game lookup...")
        first_game = games[0]
        away_team = first_game['away']['mascot']
        home_team = first_game['home']['mascot']

        found_game = fetcher.find_game_by_teams(away_team, home_team)
        if found_game:
            print(f"   Found: {away_team} @ {home_team}")

            # Test odds retrieval
            print(f"\n5. Testing odds retrieval for DraftKings...")
            odds = fetcher.get_all_odds_for_game(found_game, "draftkings")
            print(f"   Spread: {odds['spread']}")
            print(f"   Moneyline: {odds['moneyline']}")
            print(f"   Total: {odds['total']}")

            print(f"\n6. Available sportsbooks:")
            books = fetcher.get_available_sportsbooks(found_game)
            print(f"   {', '.join(books)}")

            print("\n" + "=" * 70)
            print("CFB Integration Test: PASSED")
            print("=" * 70)
        else:
            print("   ERROR: Could not find game")
    else:
        print("   WARNING: No games found")
else:
    print("   ERROR: Failed to fetch odds")
