"""
Test Charlotte API endpoints for CFB support
"""

import requests
import json

API_KEY = "ZV03bez9ie16w553FeuM8Z49Djw67pt6"

# Test different possible CFB endpoints
endpoints_to_test = [
    ("CFB", "https://charlotte.rotogrinders.com/sports/cfb/extended"),
    ("NCAAF", "https://charlotte.rotogrinders.com/sports/ncaaf/extended"),
    ("College Football", "https://charlotte.rotogrinders.com/sports/college-football/extended"),
]

# Test different week formats
week_formats = ["2025-w14", "2025-14", "14", "2025-reg-14"]

print("Testing Charlotte API for CFB support...\n")
print("=" * 70)

for name, base_url in endpoints_to_test:
    print(f"\nTesting {name}: {base_url}")
    print("-" * 70)

    for week_format in week_formats:
        params = {
            "role": "scoresandodds",
            "week": week_format,
            "key": API_KEY
        }

        try:
            response = requests.get(base_url, params=params, timeout=10)

            print(f"\nWeek format: {week_format}")
            print(f"Status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                games = data.get('data', [])
                print(f"SUCCESS - Found {len(games)} games")

                if games:
                    print(f"\nSample game:")
                    game = games[0]
                    print(f"  {game.get('away', {}).get('mascot', 'N/A')} @ {game.get('home', {}).get('mascot', 'N/A')}")
                    print(f"  Game ID: {game.get('id', 'N/A')}")

                    # Check if odds are available
                    if 'odds' in game:
                        print(f"  Odds data available: YES")
                    else:
                        print(f"  Odds data available: NO")

                    # Found working endpoint!
                    print(f"\n*** WORKING ENDPOINT FOUND! ***")
                    print(f"   URL: {base_url}")
                    print(f"   Week format: {week_format}")
                    break
            elif response.status_code == 404:
                print(f"FAILED - 404 - Endpoint not found")
            else:
                print(f"WARNING - Status {response.status_code}")
                print(f"Response: {response.text[:200]}")

        except requests.exceptions.RequestException as e:
            print(f"ERROR: {e}")

    # If we found a working endpoint, stop testing
    if 'response' in locals() and response.status_code == 200 and games:
        break

print("\n" + "=" * 70)
print("\nTest complete!")
