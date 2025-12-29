"""
BAM (Better Ad Management) API Offers Fetcher
Fetches promotional offers from the BAM API
"""

import requests
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import os

# Cache configuration
CACHE_FILE = "data/bam_offers_cache.pkl"
CACHE_DURATION = timedelta(hours=6)

# BAM API Configuration
BAM_API_URL = "https://b.bet-links.com/v1/affiliate/properties/1/placements/2037/promotions"
BAM_PARAMS = {
    "user_parent_book_ids": "",
    "context": "web-article-top-stories"  # Match Google Sheets context
}
BAM_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}


class BAMOffersFetcher:
    """Fetches and caches promotional offers from BAM API"""

    def __init__(self, cache_duration_hours: int = 6):
        self.cache_duration = timedelta(hours=cache_duration_hours)
        self.cache_file = CACHE_FILE

    def fetch_offers(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Fetch offers from BAM API with caching

        Args:
            force_refresh: If True, bypass cache and fetch fresh data

        Returns:
            List of offer dictionaries in standardized format
        """
        # Check cache first (unless force refresh)
        if not force_refresh and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    cache_age = datetime.now() - cached_data['timestamp']

                    if cache_age < self.cache_duration:
                        print(f"Using cached offers (age: {cache_age.seconds // 60} min)")
                        return cached_data['offers']
            except Exception as e:
                print(f"Cache read failed: {e}")

        # Fetch fresh data from API
        print("Fetching fresh offers from BAM API...")
        try:
            response = requests.get(
                BAM_API_URL,
                params=BAM_PARAMS,
                headers=BAM_HEADERS,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            # Parse promotions
            promotions = data.get('promotions', [])
            offers = [self._parse_promotion(promo) for promo in promotions]

            # Cache the results
            cache_data = {
                'timestamp': datetime.now(),
                'offers': offers
            }
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)

            print(f"Fetched {len(offers)} offers from BAM API")
            return offers

        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch from BAM API: {e}")
            return []

    def _parse_promotion(self, promo: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse a single promotion from BAM API into standardized format

        Args:
            promo: Raw promotion dict from BAM API

        Returns:
            Standardized offer dict compatible with existing code
        """
        affiliate = promo.get('affiliate', {})
        additional_attrs = promo.get('additional_attributes', {})
        campaign = promo.get('campaign', {})

        # Extract images
        images = promo.get('images', [])
        logo_url = ""
        promo_image_url = ""
        for img in images:
            img_type = img.get('image_type', {}).get('name', '')
            if img_type == 'Logo' and not logo_url:
                logo_url = img.get('image_url', '')
            elif img_type == 'Promo' and not promo_image_url:
                promo_image_url = img.get('image_url', '')

        # Get impression/tracking URL
        impression_url = additional_attrs.get('impression_url', '')
        impression_urls = additional_attrs.get('impression_urls', [])
        if not impression_url and impression_urls:
            impression_url = impression_urls[0]

        # Build switchboard link in the correct format
        # Format: https://switchboard.actionnetwork.com/offers?affiliateId=1895&campaignId=6641&context=web-article-top-stories&stateCode=
        affiliate_id = affiliate.get('id', '1')
        campaign_id = campaign.get('id', '')
        switchboard_link = (
            f"https://switchboard.actionnetwork.com/offers?"
            f"affiliateId={affiliate_id}&"
            f"campaignId={campaign_id}&"
            f"context=web-article-top-stories&"
            f"stateCode="
        )

        # Build standardized offer dict
        offer = {
            # BAM-specific fields
            'bam_id': promo.get('id'),
            'bam_campaign_id': campaign.get('id'),

            # Core offer info
            'brand': affiliate.get('display_name', affiliate.get('name', '')),
            'offer_text': promo.get('title', ''),
            'affiliate_offer': promo.get('title', ''),
            'bonus_code': additional_attrs.get('bonus_code', ''),
            'dollar_amount': promo.get('dollar_amount', ''),

            # Terms and conditions
            'terms': promo.get('terms', ''),
            'affiliate_terms': additional_attrs.get('affiliate_terms', promo.get('terms', '')),

            # Links
            'switchboard_link': switchboard_link,
            'url': switchboard_link,
            'impression_url': impression_url,  # Keep original for reference

            # Images
            'logo_url': logo_url,
            'promo_image_url': promo_image_url,

            # Metadata
            'expires_at': promo.get('expires_at', ''),
            'internal_identifiers': promo.get('internal_identifiers', []),
            'affiliate_type': affiliate.get('affiliate_type', ''),
            'rating': affiliate.get('additional_attributes', {}).get('rating', 0),
            'features': affiliate.get('additional_attributes', {}).get('features', []),

            # Build shortcode
            'shortcode_type': 'Promo Card',
            'shortcode': self._build_shortcode(promo),
        }

        return offer

    def _build_shortcode(self, promo: Dict[str, Any]) -> str:
        """
        Build WordPress BAM shortcode for the promotion

        Args:
            promo: Raw promotion dict from BAM API

        Returns:
            WordPress shortcode string in BAM format
            Example: [bam-inline-promotion placement-id="2037" property-id="1"
                     context="web-article-top-stories" internal-id="evergreen"
                     affiliate-type="sportsbook" affiliate="betmgm"]
        """
        affiliate = promo.get('affiliate', {})
        internal_identifiers = promo.get('internal_identifiers', [])

        # Extract required fields
        affiliate_name = affiliate.get('name', '').lower()
        affiliate_type = affiliate.get('affiliate_type', 'sportsbook').lower()

        # Get primary internal identifier
        # Prefer specific identifiers over generic ones
        # Generic identifiers: 'sportsbook', 'bonus-code', 'canada', 'mo'
        # Specific identifiers: 'evergreen', 'evergreen2', 'fbo', 'bet-get', 'lpb', etc.
        internal_id = 'evergreen'
        if internal_identifiers:
            # Define priority order (most specific first)
            priority_ids = ['fbo', 'bet-get', 'lpb', 'evergreen', 'evergreen2']
            generic_ids = {'sportsbook', 'bonus-code', 'canada', 'mo'}

            # First, try to find a priority ID
            for priority_id in priority_ids:
                if priority_id in internal_identifiers:
                    internal_id = priority_id
                    break
            else:
                # If no priority ID found, use first non-generic ID
                for id_str in internal_identifiers:
                    if id_str not in generic_ids:
                        internal_id = id_str
                        break
                else:
                    # Fallback to first ID if all are generic
                    internal_id = internal_identifiers[0]

        # Build BAM shortcode
        # Format: [bam-inline-promotion placement-id="2037" property-id="1" ...]
        shortcode = (
            f'[bam-inline-promotion '
            f'placement-id="2037" '
            f'property-id="1" '
            f'context="web-article-top-stories" '
            f'internal-id="{internal_id}" '
            f'affiliate-type="{affiliate_type}" '
            f'affiliate="{affiliate_name}"]'
        )

        return shortcode

    def get_offer_by_brand(self, brand_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the first offer for a specific sportsbook brand

        Args:
            brand_name: Brand name (e.g., "DraftKings", "BetMGM", "FanDuel")

        Returns:
            Offer dict if found, None otherwise
        """
        offers = self.fetch_offers()
        brand_lower = brand_name.lower()

        for offer in offers:
            if offer.get('brand', '').lower() == brand_lower:
                return offer

        return None

    def get_all_brands(self) -> List[str]:
        """
        Get list of all available sportsbook brands

        Returns:
            List of brand names
        """
        offers = self.fetch_offers()
        brands = list(set(offer.get('brand', '') for offer in offers if offer.get('brand')))
        return sorted(brands)


# Convenience functions
def get_bam_offers(force_refresh: bool = False) -> List[Dict[str, Any]]:
    """
    Get all offers from BAM API

    Args:
        force_refresh: If True, bypass cache

    Returns:
        List of offer dicts
    """
    fetcher = BAMOffersFetcher()
    return fetcher.fetch_offers(force_refresh=force_refresh)


def get_offer_by_brand(brand_name: str) -> Optional[Dict[str, Any]]:
    """
    Get offer for a specific brand

    Args:
        brand_name: Brand name (e.g., "DraftKings")

    Returns:
        Offer dict or None
    """
    fetcher = BAMOffersFetcher()
    return fetcher.get_offer_by_brand(brand_name)


if __name__ == "__main__":
    # Test the fetcher
    print("Testing BAM Offers Fetcher...")
    print("=" * 70)

    fetcher = BAMOffersFetcher()
    offers = fetcher.fetch_offers(force_refresh=True)

    print(f"\nFetched {len(offers)} offers")
    print("\nAvailable brands:")
    for brand in fetcher.get_all_brands():
        print(f"  - {brand}")

    print("\n" + "=" * 70)
    print("Sample offers:")
    print("=" * 70)

    for i, offer in enumerate(offers[:3]):
        print(f"\n{i+1}. {offer['brand']}")
        print(f"   Offer: {offer['offer_text']}")
        print(f"   Code: {offer['bonus_code']}")
        print(f"   Amount: ${offer['dollar_amount']}")
        print(f"   Shortcode: {offer['shortcode']}")
