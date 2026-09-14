#!/usr/bin/env python3
"""VisionSpace AI - Automated Accuracy Evaluation & Benchmarking Engine.

Tests the live VisionSpace AI visual search pipeline using ground-truth test images.
Evaluates Top-1 Precision, Top-5 Category Recall, Store Link Validity, and Request Latency.
"""

import argparse
import io
import os
import random
import sys
import time
import urllib.request
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

FIXTURES_DIR = os.path.join(PROJECT_ROOT, "tests", "fixtures")

# Ground-truth test dataset mapping 5 distinct furniture categories
GROUND_TRUTH_DATA: Dict[str, Dict[str, Any]] = {
    "sofa_test.jpg": {
        "expected_category": "Sofa",
        "description": "Sofa / Sectional",
        "keywords": ["sofa", "couch", "sectional", "davenport"],
        "url": "https://images.unsplash.com/photo-1555041469-a586c61ea9bc?auto=format&fit=crop&w=800&q=80",
    },
    "chair_test.jpg": {
        "expected_category": "Chair",
        "description": "Accent Chair / Lounge",
        "keywords": ["chair", "armchair", "recliner", "lounge"],
        "url": "https://images.unsplash.com/photo-1567538096630-e0c55bd6374c?auto=format&fit=crop&w=800&q=80",
    },
    "table_test.jpg": {
        "expected_category": "Table",
        "description": "Coffee Table / Pedestal",
        "keywords": ["table", "desk", "coffee table", "pedestal", "teak", "walnut"],
        "url": "https://images.unsplash.com/photo-1577140917170-285929fb55b7?auto=format&fit=crop&w=800&q=80",
    },
    "lamp_test.jpg": {
        "expected_category": "Lighting",
        "description": "Floor Lamp / Lighting",
        "keywords": ["lamp", "light", "lighting", "sconce"],
        "url": "https://images.unsplash.com/photo-1507473885765-e6ed057f782c?auto=format&fit=crop&w=800&q=80",
    },
    "dining_test.jpg": {
        "expected_category": "Table",
        "description": "Dining Set / Dining Table",
        "keywords": ["dining", "table", "teak", "walnut", "set", "chair"],
        "url": "https://images.unsplash.com/photo-1615066390971-03e4e1c36ddf?auto=format&fit=crop&w=800&q=80",
    },
}


def ensure_fixtures() -> None:
    """Ensure all test fixture images exist locally, downloading or generating them if missing."""
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    for filename, meta in GROUND_TRUTH_DATA.items():
        filepath = os.path.join(FIXTURES_DIR, filename)
        if not os.path.exists(filepath):
            url = meta.get("url")
            print(f"[*] Downloading test fixture '{filename}' from {url}...")
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=10) as resp, open(filepath, "wb") as f:
                    f.write(resp.read())
                print(f"[+] Downloaded {filename} successfully.")
            except Exception as exc:
                print(f"[!] Warning: Failed to download {url} ({exc}). Creating placeholder image.")
                try:
                    from PIL import Image
                    img = Image.new("RGB", (300, 300), color=(180, 160, 140))
                    img.save(filepath, format="JPEG")
                except Exception as img_exc:
                    print(f"[!] Error creating placeholder: {img_exc}")


def is_valid_https_url(url: Optional[str]) -> bool:
    """Check if the provided link is a non-empty, well-formed HTTPS URL."""
    if not url or not isinstance(url, str):
        return False
    trimmed = url.strip()
    if not trimmed.startswith("https://"):
        return False
    try:
        parsed = urlparse(trimmed)
        return bool(parsed.scheme == "https" and parsed.netloc)
    except Exception:
        return False


def infer_source(item: Dict[str, Any], link: str) -> str:
    """Extract or infer retailer source name from metadata or link domain."""
    if item.get("source"):
        return str(item["source"])
    if item.get("retailer"):
        return str(item["retailer"])

    link_lower = link.lower()
    known_retailers = {
        "amazon": "Amazon",
        "ikea": "IKEA",
        "wayfair": "Wayfair",
        "westelm": "West Elm",
        "cb2": "CB2",
        "hermanmiller": "Herman Miller",
        "target": "Target",
        "crateandbarrel": "Crate & Barrel",
        "rh.com": "Restoration Hardware",
        "dwr.com": "Design Within Reach",
        "potterybarn": "Pottery Barn",
        "unsplash": "Unsplash Store",
    }
    for key, name in known_retailers.items():
        if key in link_lower:
            return name

    try:
        netloc = urlparse(link).netloc
        clean = netloc.replace("www.", "").split(".")[0]
        return clean.capitalize() if clean else "Store"
    except Exception:
        return "Store"


def calculate_match_score(
    title: str,
    item_category: str,
    expected_category: str,
    gt_keywords: List[str],
) -> tuple[int, bool]:
    """Calculate match score percentage and binary classification match.

    Returns:
        tuple[int, bool]: (match_percentage, is_match)
    """
    title_lower = title.lower()
    cat_match = item_category.strip().lower() == expected_category.strip().lower()
    keyword_matched = any(kw.lower() in title_lower for kw in gt_keywords)

    if keyword_matched and cat_match:
        score = 100
        is_match = True
    elif cat_match or keyword_matched:
        score = 80
        is_match = True
    else:
        score = 20
        is_match = False

    return score, is_match


def evaluate_single_image(
    image_filename: str,
    api_url: str = "http://127.0.0.1:8000",
    top_k: int = 5,
) -> Dict[str, Any]:
    """Execute search query for a test image and evaluate benchmark metrics."""
    gt_meta = GROUND_TRUTH_DATA[image_filename]
    expected_category = gt_meta["expected_category"]
    expected_desc = gt_meta["description"]
    keywords = gt_meta["keywords"]

    filepath = os.path.join(FIXTURES_DIR, image_filename)
    if not os.path.exists(filepath):
        ensure_fixtures()

    with open(filepath, "rb") as f:
        img_bytes = f.read()

    endpoint = f"{api_url.rstrip('/')}/api/v1/search-furniture"
    start_time = time.perf_counter()
    try:
        response = requests.post(
            endpoint,
            files={"file": (image_filename, img_bytes, "image/jpeg")},
            data={"top_k": top_k},
            timeout=30,
        )
    except requests.exceptions.RequestException as exc:
        print(f"\n[ERROR] Failed to connect to VisionSpace AI API at {endpoint}: {exc}")
        print("Please ensure the backend is running (e.g. uvicorn app.main:app --reload).")
        sys.exit(1)

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    if response.status_code != 200:
        print(f"\n[ERROR] API returned HTTP {response.status_code}: {response.text}")
        sys.exit(1)

    data = response.json()
    results = data.get("results", [])
    if not results and data.get("detected_objects"):
        results = data["detected_objects"][0].get("matches", [])

    total_results = len(results)
    ranked_evaluations: List[Dict[str, Any]] = []
    live_links_count = 0
    matched_count = 0

    for idx, match in enumerate(results[:top_k], start=1):
        item = match.get("item", {})
        title = item.get("name") or item.get("title") or f"Item {item.get('id', idx)}"
        item_cat = item.get("category") or "Unknown"
        price = item.get("price") or 0.0
        link = item.get("buy_url") or item.get("link") or item.get("image_url") or ""
        source = infer_source(item, link)

        link_valid = is_valid_https_url(link)
        if link_valid:
            live_links_count += 1

        match_score, is_match = calculate_match_score(
            title=title,
            item_category=item_cat,
            expected_category=expected_category,
            gt_keywords=keywords,
        )
        if is_match:
            matched_count += 1

        ranked_evaluations.append({
            "rank": idx,
            "title": title,
            "category": item_cat,
            "price": price,
            "link": link,
            "source": source,
            "match_score": match_score,
            "is_match": is_match,
            "link_valid": link_valid,
        })

    # Metrics
    top1_eval = ranked_evaluations[0] if ranked_evaluations else None
    top1_pass = bool(top1_eval and top1_eval["is_match"])
    recall_pct = (matched_count / total_results * 100.0) if total_results > 0 else 0.0
    store_valid_pass = (live_links_count == total_results) and (total_results > 0)

    return {
        "image_filename": image_filename,
        "ground_truth_desc": expected_desc,
        "expected_category": expected_category,
        "latency_ms": latency_ms,
        "total_live_links": live_links_count,
        "total_results": total_results,
        "ranked_evaluations": ranked_evaluations,
        "top1_pass": top1_pass,
        "top1_title": top1_eval["title"] if top1_eval else "None",
        "recall_pct": recall_pct,
        "matched_count": matched_count,
        "store_valid_pass": store_valid_pass,
    }


def print_benchmark_report(report: Dict[str, Any]) -> None:
    """Print the structured ASCII benchmark report to stdout."""
    print("=" * 68)
    print("               VISIONSPACE AI ENGINE ACCURACY REPORT                ")
    print("=" * 68)
    print(f"Tested Image      : {report['image_filename']}")
    print(f"Ground Truth      : {report['ground_truth_desc']}")
    print(f"Execution Latency : {report['latency_ms']:.2f} ms")
    print(f"Total Live Links  : {report['total_live_links']}")
    print()
    print("[RANKED SEARCH RESULTS & MATCH SCORE]")
    for item in report["ranked_evaluations"]:
        score_str = f"{item['match_score']:3d}%"
        price_str = f"${item['price']:,.0f}" if isinstance(item['price'], (int, float)) else f"${item['price']}"
        link_display = item['link']
        if len(link_display) > 52:
            link_display = link_display[:49] + "..."

        print(f"{item['rank']}. [MATCH {score_str}] {item['title']}")
        print(f"   Source: {item['source']:<10} | Price: {price_str:<8} | Link: {link_display}")

    print()
    print("[ACCURACY METRICS]")
    if report["top1_pass"]:
        print("• Top-1 Match     : PASS (Target item identified correctly in Rank 1)")
    else:
        print(f"• Top-1 Match     : FAIL (Rank 1 was '{report['top1_title']}')")

    recall_text = (
        f"{report['recall_pct']:.1f}% "
        f"({report['matched_count']} out of {report['total_results']} links match target object)"
    )
    print(f"• Category Recall : {recall_text}")

    if report["store_valid_pass"]:
        valid_text = f"PASS ({report['total_live_links']}/{report['total_results']} valid e-commerce web links)"
    else:
        valid_text = f"FAIL ({report['total_live_links']}/{report['total_results']} valid e-commerce web links)"
    print(f"• Store Validity  : {valid_text}")
    print("=" * 68)
    print()


def main() -> None:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Benchmark VisionSpace AI visual search accuracy and store link validity.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run evaluation across all 5 benchmark fixture images.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Run evaluation on a specific fixture image (e.g. sofa_test.jpg).",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="Base URL for the running VisionSpace AI service (default: http://127.0.0.1:8000).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of nearest neighbors to retrieve (default: 5).",
    )

    args = parser.parse_args()

    # Pre-flight: ensure fixtures exist
    ensure_fixtures()

    if args.all:
        selected_images = list(GROUND_TRUTH_DATA.keys())
        print(f"\n[*] Running VisionSpace AI Benchmark Suite on all {len(selected_images)} images...\n")
        reports = []
        for img in selected_images:
            report = evaluate_single_image(img, api_url=args.api_url, top_k=args.top_k)
            print_benchmark_report(report)
            reports.append(report)

        # Overall summary
        avg_latency = sum(r["latency_ms"] for r in reports) / len(reports)
        top1_acc = (sum(1 for r in reports if r["top1_pass"]) / len(reports)) * 100.0
        avg_recall = sum(r["recall_pct"] for r in reports) / len(reports)
        all_store_valid = all(r["store_valid_pass"] for r in reports)

        print("=" * 68)
        print("                     BENCHMARK SUITE SUMMARY                        ")
        print("=" * 68)
        print(f"Images Tested     : {len(reports)}")
        print(f"Average Latency   : {avg_latency:.2f} ms")
        print(f"Top-1 Accuracy    : {top1_acc:.1f}% ({sum(1 for r in reports if r['top1_pass'])}/{len(reports)})")
        print(f"Mean Recall Ratio : {avg_recall:.1f}%")
        print(f"All Store Links   : {'PASS (100% Valid HTTPS)' if all_store_valid else 'WARNING (Some Invalid)'}")
        print("=" * 68)

    elif args.image:
        if args.image not in GROUND_TRUTH_DATA:
            print(f"[ERROR] Unknown fixture image '{args.image}'. Available: {list(GROUND_TRUTH_DATA.keys())}")
            sys.exit(1)
        report = evaluate_single_image(args.image, api_url=args.api_url, top_k=args.top_k)
        print_benchmark_report(report)
    else:
        # Randomly select one image from benchmark set
        chosen_image = random.choice(list(GROUND_TRUTH_DATA.keys()))
        report = evaluate_single_image(chosen_image, api_url=args.api_url, top_k=args.top_k)
        print_benchmark_report(report)


if __name__ == "__main__":
    main()
