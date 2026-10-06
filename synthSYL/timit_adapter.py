# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""
timit_adapter.py — TIMIT block adapter for the synthSYL package.

Groups phrases into blocks of 5 separated by long pauses (|), suitable
for batch video generation.  Reads an input file (format: ``index: text``)
and produces an output file with:

  - Systematic dot ``.`` between words (no C+V fusion across word
    boundaries)
  - Post-processing: dots inserted after the 1st consonant in 4+
    consonant sequences and after the 1st vowel in 3+ vowel sequences

Consonant/vowel classification is derived from the package's
phonological database (``PANPHON_DB``, ``SIMPLE_TO_IPA``), with
character-level overrides for glide characters (w, j) that are treated
as consonants in this adapter, consistent with the original script's
behaviour.

Usage::

    python -m panphon_timit_plugin.timit_adapter input.txt output.txt
"""

from __future__ import annotations

import sys

from .phonology import PANPHON_DB, SIMPLE_TO_IPA, feature_index


# ============================================================================
# Character-level consonant/vowel classification (derived from package data)
# ============================================================================

# Build consonant character set from the package's phonological database.
# Any IPA segment whose PanPhon ``cons`` feature is +1 is a consonant.
_CONSONANT_CHARS: set[str] = set()
for _ipa, _features in PANPHON_DB.items():
    if _features[feature_index("cons")] == 1:
        _CONSONANT_CHARS.add(_ipa)

# Add single-character synthSYL notation keys that map to consonants.
# This lets the adapter work with both IPA and synthSYL notation.
for _key, _ipa in SIMPLE_TO_IPA.items():
    if _ipa in _CONSONANT_CHARS and len(_key) == 1:
        _CONSONANT_CHARS.add(_key)

# Additional characters from the original adapter not present in
# PANPHON_DB (h, G, ð, θ, ŋ).  These are treated as consonants.
_CONSONANT_CHARS.update('hGðθŋ')

# Override: w and j are treated as consonants at the character level
# in this adapter.  In the package's segment-level classification they
# are glides (syl=+1), but the original TIMIT adapter classifies them
# as consonants; keeping this preserves identical dot-insertion behaviour.
_CONSONANT_CHARS.update('wj')

# Separator characters (neither consonant nor vowel)
_SEPARATORS: set[str] = set(' .|,;:!?')


def _is_consonant(char: str) -> bool:
    """Return True if *char* is classified as a consonant."""
    return char in _CONSONANT_CHARS


def _is_vowel(char: str) -> bool:
    """Return True if *char* is classified as a vowel.

    A character is a vowel if it is neither a consonant nor a separator.
    """
    if char in _SEPARATORS:
        return False
    return not _is_consonant(char)


# ============================================================================
# Post-processing: insert dots to limit long consonant/vowel sequences
# ============================================================================

def _insert_dots_for_long_sequences(text: str) -> str:
    """Insert dots to break up long consonant or vowel sequences.

    Scans *text* and inserts a dot whenever a sequence of 4+ consonants
    or 3+ vowels is detected:

      - 4+ consonants: dot after the 1st consonant (C.CCC...)
      - 3+ vowels:     dot after the 1st vowel    (V.VV...)
    """
    result: list[str] = []
    i = 0
    n = len(text)

    while i < n:
        c = text[i]

        # Existing dots are preserved as-is
        if c == '.':
            result.append(c)
            i += 1
            continue

        # Detect a consonant sequence
        if _is_consonant(c):
            start = i
            while i < n and _is_consonant(text[i]):
                i += 1
            seq = text[start:i]
            if len(seq) >= 4:
                # Keep the 1st consonant, insert a dot, then the rest
                result.append(seq[0])
                result.append('.')
                result.append(seq[1:])
            else:
                result.append(seq)
            continue

        # Detect a vowel sequence
        if _is_vowel(c):
            start = i
            while i < n and _is_vowel(text[i]):
                i += 1
            seq = text[start:i]
            if len(seq) >= 3:
                # Keep the 1st vowel, insert a dot, then the rest
                result.append(seq[0])
                result.append('.')
                result.append(seq[1:])
            else:
                result.append(seq)
            continue

        # Other characters (should not normally occur)
        result.append(c)
        i += 1

    return ''.join(result)


# ============================================================================
# Phrase transformation (concatenation + post-processing)
# ============================================================================

def transform_phrase(phrase: str) -> str:
    """Apply concatenation rules and post-processing to a phrase.

    Step 1: insert a systematic dot between words (no C+V fusion).
    Step 2: insert dots to limit long consonant/vowel sequences.
    """
    words = phrase.split()
    if len(words) <= 1:
        return phrase

    # Step 1: always a dot between words (C+V fusion suppressed)
    joined = '.'.join(words)

    # Step 2: post-processing to limit long clusters
    return _insert_dots_for_long_sequences(joined)


# ============================================================================
# Main entry point
# ============================================================================

def main() -> None:
    """Read an input file, transform phrases, and write blocks of 5."""
    if len(sys.argv) < 3:
        print("Usage: python -m syntSYL.timit_adapter <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Read phrases from input file
    phrases: list[str] = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ':' in line:
                _, phrase = line.split(':', 1)
                phrase = phrase.strip()
                if phrase:
                    phrases.append(transform_phrase(phrase))
            else:
                phrases.append(transform_phrase(line))

    if not phrases:
        print("Error: no valid phrases found.")
        sys.exit(1)

    print(f"Phrases read: {len(phrases)}")

    # Display transformation examples
    print("\n=== Transformation examples (systematic dot) ===")
    examples = [
        "zis waz izi fOr as",
        "Zén mé èn mOr mani ba wèkin fard",
        "wa jèl Or wèi ovè sili atamz",
        "brat sanSan Simèz an za oSan",
        "nasin iz èz afènsiv èz inasans",
    ]
    for ex in examples:
        transformed = transform_phrase(ex)
        print(f"  Before: {ex}")
        print(f"  After:  {transformed}")
        print()

    # Group into blocks of 5 phrases separated by long pauses
    BLOCK_SIZE = 5
    output_lines: list[str] = []
    for i in range(0, len(phrases), BLOCK_SIZE):
        block = phrases[i:i + BLOCK_SIZE]
        block_index = i // BLOCK_SIZE + 1
        block_str = f"{block_index}: | " + " | ".join(block) + " |"
        output_lines.append(block_str)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(output_lines))

    print(f"Output file generated: {output_file}")
    print(f"Blocks created:        {len(output_lines)}")
    print(f"Total phrases:         {len(phrases)}")


if __name__ == "__main__":
    main()
