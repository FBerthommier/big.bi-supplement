#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""
embed_video_in_html.py — Embed video (MP4), poster (PNG), and captions
(WebVTT) directly into HTML as base64, to produce a fully self-contained
page (a single .html file to open).

Usage:
    python scripts/embed_video_in_html.py <base_dir> [en|fr]

<base_dir> must contain index.html, dual/dual_progressive.mp4,
dual/poster.png, dual/captions.vtt and, optionally, the comparison
figure under polar/static/. The second argument selects the page-title
variant ("en" rewrites the title to "Embedded Video"; "fr" keeps it).
"""
from __future__ import annotations
import base64
import sys
from pathlib import Path

# Select target
if len(sys.argv) < 2:
    sys.exit("usage: embed_video_in_html.py <base_dir> [en|fr]")
BASE = Path(sys.argv[1])
TARGET = sys.argv[2] if len(sys.argv) > 2 else "en"

SRC_HTML = BASE / "index.html"
OUT_HTML = BASE / "index_embedded.html"
VIDEO_PATH = BASE / "dual" / "dual_progressive.mp4"
POSTER_PATH = BASE / "dual" / "poster.png"
CAPTIONS_PATH = BASE / "dual" / "captions.vtt"
COMPARISON_PATH = BASE / "polar" / "static" / "comparison_polar.png"
ARTICLE_FIG_PATH = BASE / "article_fig4.png"  # embedded only if present (fr variant)
VIDEO_SRC = "dual/dual_progressive.mp4"
COMPARISON_SRC = "polar/static/comparison_polar.png"
TITLE_OLD = "Simulation big.bi → bi.gbi — Accessible Video (arXiv:2307.02299 §4)"
TITLE_NEW = "Simulation big.bi → bi.gbi — Embedded Video (arXiv:2307.02299 §4)"
if TARGET != "en":
    TITLE_OLD = TITLE_NEW = None


def encode_b64(path: Path, mime: str) -> str:
    """Encode a file as a base64 data URI."""
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def main() -> int:
    print("=" * 60)
    print(f"Embedding video + assets into HTML (base64) — target: {TARGET}")
    print("=" * 60)

    html = SRC_HTML.read_text(encoding="utf-8")
    print(f"Source HTML: {len(html):,} chars")

    # 1. Video
    print(f"\n[1/4] Video: {VIDEO_PATH.name} ({VIDEO_PATH.stat().st_size:,} bytes)")
    video_uri = encode_b64(VIDEO_PATH, "video/mp4")
    html = html.replace(f'src="{VIDEO_SRC}"',
                        f'src="{video_uri}"')
    html = html.replace(f'href="{VIDEO_SRC}" download',
                        f'href="{video_uri}" download="dual_progressive.mp4"')
    # Fallback link inside <video> tag (English version)
    html = html.replace(f'href="{VIDEO_SRC}">Download',
                        f'href="{video_uri}" download="dual_progressive.mp4">Download')

    # 2. Poster
    print(f"[2/4] Poster: {POSTER_PATH.name} ({POSTER_PATH.stat().st_size:,} bytes)")
    poster_uri = encode_b64(POSTER_PATH, "image/png")
    html = html.replace('poster="dual/poster.png"',
                        f'poster="{poster_uri}"')

    # 3. Captions (WebVTT as data URI)
    print(f"[3/4] Captions: {CAPTIONS_PATH.name} ({CAPTIONS_PATH.stat().st_size:,} bytes)")
    captions_uri = encode_b64(CAPTIONS_PATH, "text/vtt")
    html = html.replace('src="dual/captions.vtt"',
                        f'src="{captions_uri}"')

    # 4. Comparison figure
    if COMPARISON_PATH and COMPARISON_PATH.exists():
        print(f"[4/4] Comparison: {COMPARISON_PATH.name} "
              f"({COMPARISON_PATH.stat().st_size:,} bytes)")
        comp_uri = encode_b64(COMPARISON_PATH, "image/png")
        html = html.replace(f'src="{COMPARISON_SRC}"',
                            f'src="{comp_uri}"')
        html = html.replace(
            f'href="{COMPARISON_SRC}" download',
            f'href="{comp_uri}" download="comparison.png"')
    else:
        print("[4/4] Comparison figure not found, skipping")

    # 5. Article Figure 4 (fr version only)
    if ARTICLE_FIG_PATH and ARTICLE_FIG_PATH.exists():
        print(f"[5] Article Fig 4: {ARTICLE_FIG_PATH.name} "
              f"({ARTICLE_FIG_PATH.stat().st_size:,} bytes)")
        art_uri = encode_b64(ARTICLE_FIG_PATH, "image/png")
        html = html.replace('href="article_fig4.png" download',
                            f'href="{art_uri}" download="article_fig4.png"')

    # Update title
    if TITLE_OLD and TITLE_NEW:
        html = html.replace(f"<title>{TITLE_OLD}</title>",
                            f"<title>{TITLE_NEW}</title>")

    # Write output
    OUT_HTML.write_text(html, encoding="utf-8")
    size_mb = OUT_HTML.stat().st_size / (1024 * 1024)
    print(f"\n✓ Self-contained HTML saved: {OUT_HTML}")
    print(f"  Size: {size_mb:.2f} MB ({OUT_HTML.stat().st_size:,} bytes)")
    print(f"  Video embedded: {VIDEO_PATH.stat().st_size / 1024:.0f} KB")
    print(f"  Poster embedded: {POSTER_PATH.stat().st_size / 1024:.0f} KB")
    print(f"  Captions embedded: {CAPTIONS_PATH.stat().st_size / 1024:.1f} KB")
    if COMPARISON_PATH and COMPARISON_PATH.exists():
        print(f"  Comparison embedded: {COMPARISON_PATH.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

