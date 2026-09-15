"""Google Lens Visual Search Service using SerpAPI.

Performs real HTTPS requests to SerpAPI using the Google Lens engine for uploaded images.
All mock data fallbacks are removed. If SERPAPI_KEY is missing or invalid, raises an
explicit HTTP 400 HTTPException.
"""

import io
import os
import time
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import requests
from fastapi import HTTPException, status
from PIL import Image

from app.config import settings
from app.models import LensSearchResponse, LensVisualMatch
import re

def _clean_source_name(raw_source: Optional[str]) -> str:
    """Normalize and clean merchant/retailer name to clean store labels.

    Examples:
        'Amazon.com' -> 'Amazon'
        'IKEA US' -> 'IKEA'
        'Wayfair LLC' -> 'Wayfair'
        'target.com' -> 'Target'
    """
    if not raw_source or not str(raw_source).strip():
        return "Store"
    s = str(raw_source).strip()
    lower = s.lower()
    if "amazon" in lower:
        return "Amazon"
    if "ikea" in lower:
        return "IKEA"
    if "wayfair" in lower:
        return "Wayfair"
    if "target" in lower:
        return "Target"
    if "walmart" in lower:
        return "Walmart"
    if "west elm" in lower or "westelm" in lower:
        return "West Elm"
    if "cb2" in lower:
        return "CB2"
    if "crate and barrel" in lower or "crate & barrel" in lower:
        return "Crate & Barrel"
    if "pottery barn" in lower:
        return "Pottery Barn"
    if "overstock" in lower or "bed bath" in lower:
        return "Overstock"
    if "article" in lower:
        return "Article"
    if "ashley" in lower:
        return "Ashley Furniture"
    if "williams-sonoma" in lower or "williams sonoma" in lower:
        return "Williams Sonoma"
    if "home depot" in lower or "homedepot" in lower:
        return "The Home Depot"
    if "lowes" in lower or "lowe's" in lower:
        return "Lowe's"
    if "hayneedle" in lower:
        return "Hayneedle"
    if "houzz" in lower:
        return "Houzz"
    if "etsy" in lower:
        return "Etsy"
    if "ebay" in lower:
        return "eBay"

    # Remove domain extensions e.g. .com, .co.uk, .org, .net, etc.
    cleaned = re.sub(r"\.(com|org|net|co|us|uk|ca|io|ai)(\.[a-z]{2})?$", "", s, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(inc|llc|ltd|corp)\b\.?", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned.strip() or s


MIN_MATCH_CONFIDENCE = 0.55  # 55% score cutoff
DISALLOWED_SOURCES = {"instagram", "reddit", "pinterest", "tiktok", "twitter", "facebook"}


class LensService:
    """Service to execute real Google Lens visual searches via SerpAPI.

    Strictly forces live HTTPS communication. Raises HTTP 400 on missing or invalid keys.
    Zero mock fallbacks.
    """

    SERPAPI_IMAGE_URL = "https://serpapi.com/image"
    SERPAPI_SEARCH_URL = "https://serpapi.com/search"

    def __init__(self, api_key: Optional[str] = None):
        """Initialize LensService with an optional API key override."""
        self._api_key = api_key

    @property
    def api_key(self) -> str:
        """Resolve API key from instance, environment variable, or app settings."""
        if self._api_key is not None:
            key = self._api_key
        else:
            key = os.getenv("SERPAPI_KEY") or settings.SERPAPI_KEY or ""
        return key.strip() if key else ""

    def validate_api_key(self) -> str:
        """Validate presence of a non-empty, non-placeholder SerpAPI key.

        Raises:
            HTTPException(400): If the key is missing or blank.
        """
        key = self.api_key
        invalid_placeholders = {
            "",
            "your_serpapi_key",
            "mock",
            "test_key",
            "placeholder",
            "none",
            "null",
        }
        if not key or key.lower() in invalid_placeholders:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "SERPAPI_KEY is missing or unconfigured. "
                    "Please provide a valid SerpAPI key in your environment or configuration."
                ),
            )
        return key

    def upload_image_to_serpapi(
        self,
        image_bytes: bytes,
        filename: str = "query.jpg",
        content_type: str = "image/jpeg",
        timeout: int = 30,
    ) -> str:
        """Upload image bytes to SerpAPI's dedicated Image API to obtain an image_id.

        Args:
            image_bytes: Raw binary content of the image.
            filename: Name for the multipart file upload.
            content_type: MIME type of the image.
            timeout: Request timeout in seconds.

        Returns:
            str: The SerpAPI image_id for subsequent Google Lens queries.

        Raises:
            HTTPException(400): If API key is missing/invalid or upload fails.
        """
        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot search with empty image content.",
            )

        api_key = self.validate_api_key()

        try:
            response = requests.post(
                self.SERPAPI_IMAGE_URL,
                data={"api_key": api_key},
                files={"image": (filename, image_bytes, content_type)},
                timeout=timeout,
            )
        except requests.exceptions.RequestException as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Network error connecting to SerpAPI image upload endpoint: {str(exc)}",
            )

        # Handle unauthorized or invalid key
        if response.status_code in (401, 403):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid SERPAPI_KEY: Authentication failed with SerpAPI. Please verify your API key.",
            )

        if response.status_code != 200:
            error_detail = response.text
            try:
                err_json = response.json()
                if "error" in err_json:
                    error_detail = err_json["error"]
            except Exception:
                pass

            if "invalid api key" in error_detail.lower():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid SERPAPI_KEY: {error_detail}",
                )

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"SerpAPI Image Upload failed (HTTP {response.status_code}): {error_detail}",
            )

        try:
            data = response.json()
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to parse SerpAPI upload response JSON: {str(exc)}",
            )

        image_id = data.get("image_id")
        if not image_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"SerpAPI did not return an image_id. Response: {data}",
            )

        return image_id

    def _query_raw_by_image_id(
        self,
        image_id: str,
        country: str = "us",
        language: str = "en",
        timeout: int = 30,
    ) -> List[Dict[str, Any]]:
        """Query Google Lens engine on SerpAPI using a previously uploaded image_id.

        Args:
            image_id: The SerpAPI image identifier.
            country: Two-letter country code (default 'us').
            language: Language code (default 'en').
            timeout: Request timeout in seconds.

        Returns:
            List[LensVisualMatch]: Parsed real visual matches from Google Lens.

        Raises:
            HTTPException(400): If API key is invalid or SerpAPI returns an error.
        """
        api_key = self.validate_api_key()

        params = {
            "engine": "google_lens",
            "image_id": image_id,
            "api_key": api_key,
            "gl": country,
            "hl": language,
        }

        try:
            response = requests.get(
                self.SERPAPI_SEARCH_URL,
                params=params,
                timeout=timeout,
            )
        except requests.exceptions.RequestException as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Network error querying SerpAPI Google Lens: {str(exc)}",
            )

        if response.status_code in (401, 403):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid SERPAPI_KEY: Authentication failed during Google Lens search.",
            )

        if response.status_code != 200:
            error_detail = response.text
            try:
                err_json = response.json()
                if "error" in err_json:
                    error_detail = err_json["error"]
            except Exception:
                pass

            if "invalid api key" in error_detail.lower():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid SERPAPI_KEY: {error_detail}",
                )

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"SerpAPI Google Lens search failed (HTTP {response.status_code}): {error_detail}",
            )

        try:
            data = response.json()
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to parse SerpAPI Google Lens JSON response: {str(exc)}",
            )

        if "error" in data:
            error_msg = data["error"]
            if "invalid api key" in str(error_msg).lower():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid SERPAPI_KEY: {error_msg}",
                )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"SerpAPI Google Lens returned error: {error_msg}",
            )

        raw_matches = data.get("visual_matches", [])
        return raw_matches

    def search_by_image_id(
        self,
        image_id: str,
        country: str = "us",
        language: str = "en",
        timeout: int = 30,
    ) -> List[LensVisualMatch]:
        """Query Google Lens engine on SerpAPI using image_id and return typed models."""
        raw_matches = self._query_raw_by_image_id(
            image_id=image_id,
            country=country,
            language=language,
            timeout=timeout,
        )
        return self._parse_visual_matches(raw_matches)

    def _query_raw_by_url(
        self,
        image_url: str,
        country: str = "us",
        language: str = "en",
        timeout: int = 30,
    ) -> List[Dict[str, Any]]:
        """Query Google Lens directly using a public URL and return raw visual_matches."""
        api_key = self.validate_api_key()

        params = {
            "engine": "google_lens",
            "url": image_url,
            "api_key": api_key,
            "gl": country,
            "hl": language,
        }

        try:
            response = requests.get(
                self.SERPAPI_SEARCH_URL,
                params=params,
                timeout=timeout,
            )
        except requests.exceptions.RequestException as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Network error querying SerpAPI Google Lens URL: {str(exc)}",
            )

        if response.status_code in (401, 403):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid SERPAPI_KEY: Authentication failed during Google Lens search.",
            )

        if response.status_code != 200:
            error_detail = response.text
            try:
                err_json = response.json()
                if "error" in err_json:
                    error_detail = err_json["error"]
            except Exception:
                pass
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"SerpAPI Google Lens search failed (HTTP {response.status_code}): {error_detail}",
            )

        data = response.json()
        if "error" in data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"SerpAPI Google Lens returned error: {data['error']}",
            )

        return data.get("visual_matches", [])

    def search_by_image_url(
        self,
        image_url: str,
        country: str = "us",
        language: str = "en",
        timeout: int = 30,
    ) -> List[LensVisualMatch]:
        """Query Google Lens engine on SerpAPI directly with a public image URL."""
        raw_matches = self._query_raw_by_url(image_url, country=country, language=language, timeout=timeout)
        return self._parse_visual_matches(raw_matches)

    def get_raw_visual_matches(
        self,
        image_input: Union[bytes, io.BytesIO, Image.Image, str],
        filename: str = "query.jpg",
        content_type: str = "image/jpeg",
        country: str = "us",
        language: str = "en",
    ) -> List[Dict[str, Any]]:
        """Perform live HTTPS Google Lens search and return raw visual_matches list from SerpAPI."""
        self.validate_api_key()

        # Handle public image URL string
        if isinstance(image_input, str) and (image_input.startswith("http://") or image_input.startswith("https://")):
            return self._query_raw_by_url(image_input, country=country, language=language)

        # Convert input to raw image bytes
        if isinstance(image_input, bytes):
            image_bytes = image_input
        elif isinstance(image_input, io.BytesIO):
            image_bytes = image_input.getvalue()
        elif isinstance(image_input, Image.Image):
            buf = io.BytesIO()
            image_input.convert("RGB").save(buf, format="JPEG", quality=95)
            image_bytes = buf.getvalue()
        elif isinstance(image_input, str) and os.path.exists(image_input):
            with open(image_input, "rb") as f:
                image_bytes = f.read()
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported image input type: {type(image_input)}",
            )

        # Upload to SerpAPI
        image_id = self.upload_image_to_serpapi(
            image_bytes=image_bytes,
            filename=filename,
            content_type=content_type,
        )

        return self._query_raw_by_image_id(
            image_id=image_id,
            country=country,
            language=language,
        )

    def get_visual_matches(
        self,
        image_input: Union[bytes, io.BytesIO, Image.Image, str],
        filename: str = "query.jpg",
        content_type: str = "image/jpeg",
        country: str = "us",
        language: str = "en",
    ) -> List[Dict[str, Any]]:
        """Execute live visual search and return clean, provider-sanitized results for the API.

        Format:
        [
            {
                "title": match.get("title"),
                "source": "Store Name",
                "price": "$...",
                "link": "https://...",
                "thumbnail": "https://..."
            }, ...
        ]
        """
        raw_matches = self.get_raw_visual_matches(
            image_input=image_input,
            filename=filename,
            content_type=content_type,
            country=country,
            language=language,
        )

        results: List[Dict[str, Any]] = []
        for match in raw_matches:
            price_data = match.get("price")
            if isinstance(price_data, dict):
                extracted = price_data.get("extracted_value")
                val_str = price_data.get("value")
                if val_str:
                    price_val = val_str
                elif extracted is not None:
                    price_val = f"${extracted:,.2f}"
                else:
                    price_val = "Check Site"
            elif isinstance(price_data, str) and price_data.strip():
                price_val = price_data.strip()
            elif isinstance(price_data, (int, float)):
                price_val = f"${price_data:,.2f}"
            else:
                price_val = "Check Site"

            clean_source = _clean_source_name(match.get("source"))

            results.append({
                "title": match.get("title") or "Visual Match",
                "source": clean_source,
                "price": price_val,
                "link": match.get("link") or "",
                "thumbnail": match.get("thumbnail") or match.get("image") or "",
            })

        return results

    def search_lens(
        self,
        image_input: Union[bytes, io.BytesIO, Image.Image, str],
        filename: str = "query.jpg",
        content_type: str = "image/jpeg",
        country: str = "us",
        language: str = "en",
    ) -> List[LensVisualMatch]:
        """Perform end-to-end visual search and return typed LensVisualMatch models."""
        raw_matches = self.get_raw_visual_matches(
            image_input=image_input,
            filename=filename,
            content_type=content_type,
            country=country,
            language=language,
        )
        return self._parse_visual_matches(raw_matches)

    def _parse_visual_matches(self, raw_matches: List[Dict[str, Any]]) -> List[LensVisualMatch]:
        """Parse raw visual matches into typed LensVisualMatch models.

        Zero mock data. Only sanitized authentic listings are included.
        """
        results: List[LensVisualMatch] = []

        for idx, item in enumerate(raw_matches, start=1):
            title = item.get("title") or "Visual Match"
            link = item.get("link") or ""
            source = _clean_source_name(item.get("source"))
            thumbnail = item.get("thumbnail") or None
            image = item.get("image") or None

            # Parse price if present
            price_val: Optional[str] = None
            extracted_num: Optional[float] = None
            price_data = item.get("price")
            if isinstance(price_data, dict):
                extracted_num = price_data.get("extracted_value")
                price_val = price_data.get("value") or (f"${extracted_num:,.2f}" if extracted_num else None)
            elif isinstance(price_data, (str, int, float)):
                price_val = str(price_data)
                try:
                    clean_str = "".join(c for c in price_val if c.isdigit() or c == ".")
                    extracted_num = float(clean_str) if clean_str else None
                except Exception:
                    pass

            results.append(
                LensVisualMatch(
                    position=item.get("position", idx),
                    title=title,
                    link=link,
                    source=source,
                    price=price_val,
                    extracted_price=extracted_num,
                    thumbnail=thumbnail,
                    image=image,
                )
            )

        return results

    def search_multi_crops(
        self,
        crops: List[Union[Image.Image, Dict[str, Any]]],
        top_matches_per_crop: int = 3,
        min_confidence: float = MIN_MATCH_CONFIDENCE,
    ) -> List[Dict[str, Any]]:
        """Unified visual search across multiple cropped image regions.

        Iterates over each crop, queries Google Lens for live web visual matches,
        extracts the top 3 store listings, and derives a smart descriptive label.
        Discards low-confidence items below min_confidence (default 55%) and social media links.

        Args:
            crops: List of PIL Images or crop metadata dictionaries from extract_all_furniture_crops.
            top_matches_per_crop: Number of top store matches to keep per crop (default: 3).
            min_confidence: Minimum detection confidence threshold (default: 0.55).

        Returns:
            List of structured items ready for the API response:
            [
                {
                    "item_id": 1,
                    "detected_name": "Teal Tufted Leather Sofa",
                    "category": "Sofa",
                    "bbox": [x1, y1, x2, y2],
                    "confidence": 0.89,
                    "matches": [ ...top store buy links... ],
                }, ...
            ]
        """
        items: List[Dict[str, Any]] = []

        for idx, crop_item in enumerate(crops, start=1):
            if isinstance(crop_item, dict):
                crop_img = crop_item.get("cropped_image")
                category = crop_item.get("category", "Furniture")
                label = crop_item.get("label", "furniture")
                bbox = crop_item.get("bbox")
                confidence = crop_item.get("confidence")
            elif isinstance(crop_item, Image.Image):
                crop_img = crop_item
                category = "Furniture"
                label = "furniture"
                bbox = None
                confidence = None
            else:
                continue

            if crop_img is None:
                continue

            # Quality filter: Discard items with confidence below 55% threshold
            if confidence is not None and confidence < min_confidence:
                continue

            # Convert crop PIL Image to JPEG bytes
            buf = io.BytesIO()
            crop_img.convert("RGB").save(buf, format="JPEG", quality=95)
            crop_bytes = buf.getvalue()

            # Live visual search
            try:
                raw_matches = self.get_visual_matches(crop_bytes)
                # Filter out irrelevant social media platforms (Instagram, Reddit, Pinterest)
                cleaned_matches = [
                    m for m in raw_matches
                    if not any(
                        dis in (m.get("source") or "").lower() or dis in (m.get("link") or "").lower()
                        for dis in DISALLOWED_SOURCES
                    )
                ]
                matches = cleaned_matches[:top_matches_per_crop]
            except Exception as exc:
                print(f"Warning: Visual search failed for crop #{idx} ({exc})")
                matches = []

            smart_name = _derive_smart_name(matches, fallback_label=label, fallback_category=category)

            # Determine / refine category
            item_cat = category
            if matches:
                first_title = matches[0].get("title", "").lower()
                if any(w in first_title for w in ["sofa", "couch", "sectional", "loveseat"]):
                    item_cat = "Sofa"
                elif any(w in first_title for w in ["table", "desk", "coffee table"]):
                    item_cat = "Table"
                elif any(w in first_title for w in ["chair", "armchair", "recliner", "lounge"]):
                    item_cat = "Chair"
                elif any(w in first_title for w in ["tv", "console", "media", "stand"]):
                    item_cat = "TV & Media"
                elif any(w in first_title for w in ["plant", "planter", "vase", "lamp", "clock"]):
                    item_cat = "Decor"

            items.append({
                "item_id": len(items) + 1,
                "detected_name": smart_name,
                "category": item_cat,
                "bbox": bbox,
                "confidence": confidence,
                "matches": matches,
            })

        return items


def _derive_smart_name(matches: List[Dict[str, Any]], fallback_label: str, fallback_category: str) -> str:
    """Derive a clean, smart product title for a detected region from its top visual match.

    Examples:
        "Article Sven Teal Tufted Velvet 88\" Sectional Sofa" -> "Teal Tufted Velvet Sectional Sofa"
        "IKEA LACK Modern Coffee Table in White" -> "Modern Coffee Table in White"
    """
    if not matches:
        return f"Modern {fallback_category.title()}" if fallback_category else "Interior Furniture"

    top_title = matches[0].get("title", "").strip()
    if not top_title:
        return f"Modern {fallback_category.title()}"

    # Strip store prefixes/suffixes e.g. "Wayfair.com: ", "Amazon.com | ", "- IKEA"
    cleaned = re.sub(r"^.*?:\s*", "", top_title)
    cleaned = re.sub(r"\s*\|.*$", "", cleaned)
    cleaned = re.sub(r"\s*-\s*(Wayfair|Amazon|IKEA|Target|Walmart|West Elm|CB2|Overstock).*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip()

    # Truncate if excessively verbose
    words = cleaned.split()
    if len(words) > 7:
        cleaned = " ".join(words[:7])

    return cleaned if len(cleaned) >= 5 else top_title[:45]


# Global singleton instance
lens_service = LensService()
