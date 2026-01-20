"""
BAM (Better Ad Management) API Offers Fetcher
Fetches promotional offers from the BAM API for multiple properties
"""

import requests
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import os

# Cache configuration
CACHE_DIR = "data"
CACHE_DURATION = timedelta(hours=6)

# Property configurations
# Each property has: property_id, placement_id, switchboard_domain, name, default_context
PROPERTIES = {
    "action_network": {
        "property_id": "1",
        "placement_id": "2037",
        "switchboard_domain": "switchboard.actionnetwork.com",
        "name": "Action Network",
        "default_context": "web-article-top-stories"
    },
    "vegas_insider": {
        "property_id": "2",
        "placement_id": "2035",
        "switchboard_domain": "switchboard.vegasinsider.com",
        "name": "VegasInsider",
        "default_context": "web-article-top-stories"
    },
    "rotogrinders": {
        "property_id": "3",
        "placement_id": "2039",
        "switchboard_domain": "switchboard.rotogrinders.com",
        "name": "RotoGrinders",
        "default_context": "web-article-top-stories"
    },
    "scores_and_odds": {
        "property_id": "4",
        "placement_id": "2029",
        "switchboard_domain": "switchboard.scoresandodds.com",
        "name": "ScoresAndOdds",
        "default_context": "web-article-top-stories"
    },
    "fantasy_labs": {
        "property_id": "11",
        "placement_id": "2041",
        "switchboard_domain": "switchboard.actionnetwork.com",
        "name": "FantasyLabs",
        "default_context": "web-article-top-stories"
    }
}

# Default property
DEFAULT_PROPERTY = "action_network"

BAM_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}


class BAMOffersFetcher:
    """Fetches and caches promotional offers from BAM API"""

    def __init__(self, property_key: str = DEFAULT_PROPERTY, cache_duration_hours: int = 6):
        """
        Initialize the fetcher for a specific property

        Args:
            property_key: Key from PROPERTIES dict (e.g., "action_network", "vegas_insider")
            cache_duration_hours: How long to cache offers (default 6 hours)
        """
        if property_key not in PROPERTIES:
            raise ValueError(f"Unknown property: {property_key}. Available: {list(PROPERTIES.keys())}")

        self.property_key = property_key
        self.property_config = PROPERTIES[property_key]
        self.cache_duration = timedelta(hours=cache_duration_hours)
        self.cache_file = os.path.join(CACHE_DIR, f"bam_offers_{property_key}.pkl")

        # Build API URL for this property
        self.api_url = (
            f"https://b.bet-links.com/v1/affiliate/properties/"
            f"{self.property_config['property_id']}/placements/"
            f"{self.property_config['placement_id']}/promotions"
        )

    def fetch_offers(self, force_refresh: bool = False, context: str = None) -> List[Dict[str, Any]]:
        """
        Fetch offers from BAM API with caching

        Args:
            force_refresh: If True, bypass cache and fetch fresh data
            context: Override default context (e.g., "web-article-top-stories")

        Returns:
            List of offer dictionaries in standardized format
        """
        context = context or self.property_config['default_context']

        # Check cache first (unless force refresh)
        if not force_refresh and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    cache_age = datetime.now() - cached_data['timestamp']

                    if cache_age < self.cache_duration:
                        print(f"Using cached {self.property_config['name']} offers (age: {cache_age.seconds // 60} min)")
                        return cached_data['offers']
            except Exception as e:
                print(f"Cache read failed: {e}")

        # Fetch fresh data from API
        print(f"Fetching fresh offers from BAM API ({self.property_config['name']})...")
        try:
            params = {
                "user_parent_book_ids": "",
                "context": context
            }

            response = requests.get(
                self.api_url,
                params=params,
                headers=BAM_HEADERS,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            # Parse promotions with property context
            promotions = data.get('promotions', [])
            offers = [self._parse_promotion(promo, context) for promo in promotions]

            # Cache the results
            cache_data = {
                'timestamp': datetime.now(),
                'offers': offers,
                'context': context
            }
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)

            print(f"Fetched {len(offers)} offers from BAM API ({self.property_config['name']})")
            return offers

        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch from BAM API: {e}")
            return []

    def _parse_promotion(self, promo: Dict[str, Any], context: str) -> Dict[str, Any]:
        """
        Parse a single promotion from BAM API into standardized format

        Args:
            promo: Raw promotion dict from BAM API
            context: The context used for this request

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

        # Build switchboard link with property-specific domain
        affiliate_id = affiliate.get('id', '1')
        campaign_id = campaign.get('id', '')
        switchboard_link = (
            f"https://{self.property_config['switchboard_domain']}/offers?"
            f"affiliateId={affiliate_id}&"
            f"campaignId={campaign_id}&"
            f"context={context}&"
            f"stateCode=&"
            f"propertyId={self.property_config['property_id']}"
        )

        # Build standardized offer dict
        offer = {
            # BAM-specific fields
            'bam_id': promo.get('id'),
            'bam_campaign_id': campaign.get('id'),
            'property': self.property_config['name'],
            'property_key': self.property_key,

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
            'impression_url': impression_url,

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
            'shortcode': self._build_shortcode(promo, context),
        }

        return offer

    def _build_shortcode(self, promo: Dict[str, Any], context: str) -> str:
        """
        Build WordPress BAM shortcode for the promotion

        Args:
            promo: Raw promotion dict from BAM API
            context: The context for this shortcode

        Returns:
            WordPress shortcode string in BAM format
        """
        affiliate = promo.get('affiliate', {})
        internal_identifiers = promo.get('internal_identifiers', [])

        # Extract required fields
        affiliate_name = affiliate.get('name', '').lower()
        affiliate_type = affiliate.get('affiliate_type', 'sportsbook').lower()

        # Get primary internal identifier
        # Prefer specific identifiers over generic ones
        internal_id = self._select_internal_id(internal_identifiers)

        # Build BAM shortcode with property-specific values
        shortcode = (
            f'[bam-inline-promotion '
            f'placement-id="{self.property_config["placement_id"]}" '
            f'property-id="{self.property_config["property_id"]}" '
            f'context="{context}" '
            f'internal-id="{internal_id}" '
            f'affiliate-type="{affiliate_type}" '
            f'affiliate="{affiliate_name}"]'
        )

        return shortcode

    def _select_internal_id(self, internal_identifiers: List[str]) -> str:
        """
        Select the best internal identifier from the list

        Args:
            internal_identifiers: List of available internal IDs

        Returns:
            The best internal ID to use
        """
        if not internal_identifiers:
            return 'evergreen'

        # Define priority order (most specific first)
        priority_ids = ['fbo', 'bet-get', 'lpb', 'omni', 'evergreen', 'evergreen2']
        generic_ids = {'sportsbook', 'bonus-code', 'canada', 'mo'}

        # First, try to find a priority ID
        for priority_id in priority_ids:
            if priority_id in internal_identifiers:
                return priority_id

        # If no priority ID found, use first non-generic ID
        for id_str in internal_identifiers:
            if id_str not in generic_ids:
                return id_str

        # Fallback to first ID if all are generic
        return internal_identifiers[0]

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
def get_bam_offers(
    property_key: str = DEFAULT_PROPERTY,
    force_refresh: bool = False,
    context: str = None
) -> List[Dict[str, Any]]:
    """
    Get all offers from BAM API for a specific property

    Args:
        property_key: Property to fetch from (default: action_network)
        force_refresh: If True, bypass cache
        context: Override default context

    Returns:
        List of offer dicts
    """
    fetcher = BAMOffersFetcher(property_key=property_key)
    return fetcher.fetch_offers(force_refresh=force_refresh, context=context)


def get_offer_by_brand(brand_name: str, property_key: str = DEFAULT_PROPERTY) -> Optional[Dict[str, Any]]:
    """
    Get offer for a specific brand from a property

    Args:
        brand_name: Brand name (e.g., "DraftKings")
        property_key: Property to fetch from

    Returns:
        Offer dict or None
    """
    fetcher = BAMOffersFetcher(property_key=property_key)
    return fetcher.get_offer_by_brand(brand_name)


def get_available_properties() -> Dict[str, str]:
    """
    Get dict of available properties {key: display_name}

    Returns:
        Dict mapping property keys to display names
    """
    return {key: config['name'] for key, config in PROPERTIES.items()}


def get_property_config(property_key: str) -> Dict[str, str]:
    """
    Get configuration for a specific property

    Args:
        property_key: Property key

    Returns:
        Property config dict
    """
    if property_key not in PROPERTIES:
        raise ValueError(f"Unknown property: {property_key}")
    return PROPERTIES[property_key].copy()


if __name__ == "__main__":
    # Test the fetcher with multiple properties
    print("Testing BAM Offers Fetcher - Multi-Property Support")
    print("=" * 70)

    for prop_key, prop_config in PROPERTIES.items():
        print(f"\n{prop_config['name']} (property_id={prop_config['property_id']})")
        print("-" * 50)

        try:
            fetcher = BAMOffersFetcher(property_key=prop_key)
            offers = fetcher.fetch_offers(force_refresh=True)

            print(f"  Fetched {len(offers)} offers")

            if offers:
                # Show first offer
                offer = offers[0]
                print(f"  Sample: {offer['brand']} - {offer['offer_text'][:40]}...")
                print(f"  Shortcode: {offer['shortcode'][:80]}...")

        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n" + "=" * 70)
    print("Test complete!")
