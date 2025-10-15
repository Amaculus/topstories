import json
import os

# Check what's in the evergreen index
index_file = "storage/evergreen_index.json"
if os.path.exists(index_file):
    with open(index_file) as f:
        items = json.load(f)
    
    print(f"Total articles in index: {len(items)}")
    
    # Categorize
    reviews = [i for i in items if "/reviews/" in i.get("url", "")]
    odds = [i for i in items if "/odds" in i.get("url", "")]
    education = [i for i in items if "/education/" in i.get("url", "")]
    
    print(f"- Operator reviews: {len(reviews)}")
    print(f"- Odds pages: {len(odds)}")
    print(f"- Education content: {len(education)}")
    print(f"- Other: {len(items) - len(reviews) - len(odds) - len(education)}")
    
    # Show some examples
    print("\nSample operator reviews:")
    for r in reviews[:3]:
        print(f"  - {r.get('title')} → {r.get('url')}")
    
    print("\nSample odds pages:")
    for o in odds[:3]:
        print(f"  - {o.get('title')} → {o.get('url')}")
    
    print("\nSample education:")
    for e in education[:3]:
        print(f"  - {e.get('title')} → {e.get('url')}")
else:
    print("No evergreen index found!")