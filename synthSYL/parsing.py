# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""
parsing.py — Text analysis and syllable structure for the synthSYL model.

This is Step 1 of the synthesis pipeline.  The public entry point is
``parse_input()``, which returns a ``ParseResult`` dataclass.

Pipeline:
    parse_input → tokenize + build syllables → accumulate → build boundaries → finalise
"""

from __future__ import annotations

from .phonology import PANPHON_DB, SIMPLE_TO_IPA, feature_index
from .types import ParseResult, SyllableBoundary


# ═══════════════════════════════════════════════════════════════════════
# Pre-computed constants (Directive 7)
# ═══════════════════════════════════════════════════════════════════════

_SYL_FEATURE_INDEX: int = feature_index("syl")
"""Pre-computed index of the 'syl' feature in PanPhon vectors."""


# ═══════════════════════════════════════════════════════════════════════
# Business-rule helpers (Directive 6)
# ═══════════════════════════════════════════════════════════════════════

def _is_vowel_seg(seg: str) -> bool:
    """Check if a segment is a vowel using PanPhon features.

    Uses the pre-computed ``_SYL_FEATURE_INDEX`` instead of calling
    ``feature_index("syl")`` on every invocation.
    """
    pv = PANPHON_DB.get(seg)
    if pv is None:
        return False
    return pv[_SYL_FEATURE_INDEX] == 1


def _is_pause_marker(token: str) -> bool:
    """Check if a token is a pause marker ('|')."""
    return token == '|'


def _is_syllable_separator(char: str) -> bool:
    """Check if a character is an explicit syllable separator ('.' or '-')."""
    return char in ('.', '-')


def _is_valid_segment(segment: str) -> bool:
    """Check if a segment is a recognized IPA segment.

    A segment is valid if it is either in the PanPhon database
    directly or has a simple-to-IPA mapping that resolves to a
    known PanPhon entry.
    """
    return segment in PANPHON_DB or segment in SIMPLE_TO_IPA


# ═══════════════════════════════════════════════════════════════════════
# Longest-match tokenisation (Directive 5)
# ═══════════════════════════════════════════════════════════════════════

def _match_longest_token(text: str, pos: int) -> tuple[str, int] | None:
    """Find the longest recognised IPA segment starting at *pos*.

    Searches ``SIMPLE_TO_IPA`` first, then ``PANPHON_DB`` for the
    longest matching segment (up to 3 characters).  This preserves
    the original lookup order: simple mapping takes priority over
    direct IPA match.

    Returns (segment, length) if a match is found, or None.
    """
    for length in [3, 2, 1]:
        if pos + length > len(text):
            continue
        candidate = text[pos:pos + length]
        # Check simple-to-IPA mapping first
        if candidate in SIMPLE_TO_IPA:
            mapped = SIMPLE_TO_IPA[candidate]
            if mapped not in ('.', ' ') and mapped in PANPHON_DB:
                return mapped, length
        # Check direct IPA match
        if candidate in PANPHON_DB and length == len(candidate):
            return candidate, length
    return None


def _tokenize_word(word: str) -> list[str]:
    """Tokenize a single word into IPA segments and dot separators.

    Syllable separators ('.' and '-') are converted to '.' tokens.
    Unrecognised characters are silently consumed.
    """
    tokens: list[str] = []
    pos = 0
    while pos < len(word):
        if _is_syllable_separator(word[pos]):
            tokens.append('.')
            pos += 1
            continue
        result = _match_longest_token(word, pos)
        if result is not None:
            seg, length = result
            tokens.append(seg)
            pos += length
        else:
            pos += 1  # Skip unknown character
    return tokens


# ═══════════════════════════════════════════════════════════════════════
# Syllable construction from tokens (Directive 4)
# ═══════════════════════════════════════════════════════════════════════

def _should_split_at_dot(prev_seg: str | None, next_seg: str | None) -> bool:
    """Determine whether to split at a dot separator.

    Returns True if the dot should be kept as a syllable boundary,
    False if the syllables should be merged.

    The phonological rule is:
      - C.C → split (keep the boundary)
      - C.V, V.C, V.V → merge (ignore the dot)

    If either surrounding segment is unknown, the default is to
    split (keep the boundary).
    """
    if prev_seg is None or next_seg is None:
        return True
    prev_v = _is_vowel_seg(prev_seg)
    next_v = _is_vowel_seg(next_seg)
    # C.C → split; C.V, V.C, V.V → merge
    return not prev_v and not next_v


def _build_syllables(tokens: list[str]) -> list[list[str]]:
    """Build syllables from a list of tokens (phonemes and dots).

    Dots are handled as explicit syllable separators.  The split/merge
    decision is delegated to ``_should_split_at_dot()``.
    """
    word_sylls: list[list[str]] = []
    current_syll: list[str] = []

    i_tok = 0
    while i_tok < len(tokens):
        tok = tokens[i_tok]
        if tok == '.':
            # Determine surrounding segments
            prev_seg: str | None = None
            if current_syll:
                prev_seg = current_syll[-1]
            elif word_sylls:
                prev_seg = word_sylls[-1][-1]

            next_seg: str | None = None
            for j in range(i_tok + 1, len(tokens)):
                if tokens[j] != '.':
                    next_seg = tokens[j]
                    break

            if _should_split_at_dot(prev_seg, next_seg):
                if current_syll:
                    word_sylls.append(current_syll)
                    current_syll = []
            # If not splitting, ignore the dot (continue in same syllable)

            i_tok += 1
            continue

        # Phoneme token: add to current syllable
        current_syll.append(tok)
        i_tok += 1

    if current_syll:
        word_sylls.append(current_syll)

    return word_sylls


# ═══════════════════════════════════════════════════════════════════════
# Pipeline helpers (Directive 3)
# ═══════════════════════════════════════════════════════════════════════

def _parse_word(word: str) -> list[list[str]]:
    """Parse a single word into a list of syllables.

    Tokenizes the word, then builds syllables from the tokens
    using the dot-separator merge logic.
    """
    tokens = _tokenize_word(word)
    if not tokens:
        return []
    return _build_syllables(tokens)


def _flush_current_block(blocks: list[list[str]],
                        syllables_per_word: list[list[list[str]]],
                        current_block: list[str],
                        current_word_sylls: list[list[str]]) -> None:
    """Flush the current block and word syllables into the result lists.

    Called when a pause is encountered or at the end of parsing.
    Modifies *blocks* and *syllables_per_word* in place.
    """
    if current_block:
        blocks.append(current_block[:])
        syllables_per_word.append(current_word_sylls[:])


def _build_syllable_boundaries(
    syllables_per_word: list[list[list[str]]],
    flat: list[str],
) -> list[SyllableBoundary]:
    """Build syllable boundary information from the parsed data.

    For each inter-syllable boundary within a word, records the
    flat-list index, the previous syllable's segments, and the
    current syllable's segments.

    Returns a list of ``SyllableBoundary`` instances.
    """
    syl_boundary_info: list[SyllableBoundary] = []
    used_indices: set[int] = set()
    for word_syls in syllables_per_word:
        for si in range(1, len(word_syls)):
            target_seg = word_syls[si][0]
            for fi, seg in enumerate(flat):
                if seg == target_seg and fi not in used_indices:
                    syl_boundary_info.append(SyllableBoundary(
                        flat_idx=fi,
                        prev_segments=word_syls[si - 1],
                        curr_segments=word_syls[si],
                    ))
                    used_indices.add(fi)
                    break
    return syl_boundary_info


def _finalize_parse(
    flat: list[str],
    blocks: list[list[str]],
    syllables_per_word: list[list[list[str]]],
    word_starts: list[int],
    syl_boundaries: set[int],
    syl_boundary_info: list[SyllableBoundary],
) -> ParseResult:
    """Create the final ``ParseResult`` from accumulated parse data."""
    return ParseResult(
        flat=flat,
        blocks=blocks,
        syllables_per_word=syllables_per_word,
        word_starts=word_starts,
        syllable_boundaries=syl_boundaries,
        syllable_boundary_info=syl_boundary_info,
    )


# ═══════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════

def parse_input(text: str) -> ParseResult:
    """Parse phonetic input text into a structured representation.

    The input text uses synthSYL notation: IPA segments separated by
    spaces, ``|`` for pauses, and ``.`` or ``-`` for explicit syllable
    breaks.

    Returns a ``ParseResult`` dataclass with the following fields:

        flat:                  Flat list of IPA segments and markers.
        blocks:                Word-level blocks (list of segment lists).
        syllables_per_word:    Syllables grouped by word.
        word_starts:           Flat-list indices where each word begins.
        syllable_boundaries:   Set of flat-list indices at syllable boundaries.
        syllable_boundary_info: Detailed boundary descriptors.
    """
    words_raw = text.strip().split()

    flat: list[str] = []
    blocks: list[list[str]] = []
    syllables_per_word: list[list[list[str]]] = []
    current_block: list[str] = []
    current_word_sylls: list[list[str]] = []
    word_starts: list[int] = []
    syl_boundaries: set[int] = set()
    syl_boundary_info: list[SyllableBoundary] = []

    for w in words_raw:
        if _is_pause_marker(w):
            flat.append('|')
            _flush_current_block(blocks, syllables_per_word,
                                current_block, current_word_sylls)
            current_block = []
            current_word_sylls = []
            continue

        # Insert GAP between words (before the current word)
        if flat and flat[-1] != '|':
            flat.append('GAP')

        word_starts.append(len(flat))

        # Parse the word into syllables
        word_sylls = _parse_word(w)

        # Insert segments into flat and record syllable boundaries
        word_syll_starts: list[tuple[int, list[str]]] = []
        for segs in word_sylls:
            syl_start = len(flat)
            word_syll_starts.append((syl_start, segs))
            flat.extend(segs)
            current_block.extend(segs)

        # Mark syllable boundaries (all except the first in the word)
        for idx_syl, (start, segs) in enumerate(word_syll_starts):
            if idx_syl > 0:
                syl_boundaries.add(start)

        # Build syl_boundary_info (pairs of adjacent syllables)
        for sb_idx in range(len(word_syll_starts) - 1):
            prev_start, prev_segs = word_syll_starts[sb_idx]
            curr_start, curr_segs = word_syll_starts[sb_idx + 1]
            syl_boundary_info.append(SyllableBoundary(
                flat_idx=curr_start,
                prev_segments=prev_segs,
                curr_segments=curr_segs,
            ))

        # Accumulate syllables for the current word
        current_word_sylls.extend(word_sylls)

    # Flush remaining block
    _flush_current_block(blocks, syllables_per_word,
                        current_block, current_word_sylls)

    # Add trailing short pause if needed
    if flat and flat[-1] not in ('|', 'GAP'):
        flat.append('GAP')

    return _finalize_parse(
        flat, blocks, syllables_per_word,
        word_starts, syl_boundaries, syl_boundary_info)
