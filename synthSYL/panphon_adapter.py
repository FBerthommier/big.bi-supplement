# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""
panphon_adapter.py — English text → synthSYL simplified phoneme converter.

Uses the CMU pronouncing dictionary (via NLTK) to convert English text into
words composed of IPA phonemes in the synthSYL simplified notation.  The
output is directly compatible with the package's parsing pipeline
(``parse_input``, ``panphon_pipeline``) via the ``SIMPLE_TO_IPA`` mapping.

The adapter bridges the gap between English orthography and the package's
articulatory synthesis system:

  1. English text → ARPAbet phonemes (CMU dictionary lookup)
  2. ARPAbet phonemes → synthSYL simplified notation (ARPABET_TO_SIMPLE)
  3. Simplified notation → IPA segments (``SIMPLE_TO_IPA`` in phonology.py)
  4. IPA segments → polar coordinates → Maeda trajectories (pipeline)

Unknown words are silently skipped (or replaced by a placeholder if
requested).  The first CMU pronunciation variant is used when multiple
pronunciations exist.

Installation::

    pip install nltk
    python -c "import nltk; nltk.download('cmudict')"

Usage::

    python -m panphon_timit_plugin.panphon_adapter input.txt output.txt
    python -m panphon_timit_plugin.panphon_adapter input.txt output.txt --verbose
"""

from __future__ import annotations

import re
import sys
import argparse

# ── Package imports ──────────────────────────────────────────────────
from .phonology import SIMPLE_TO_IPA, PANPHON_DB


# ============================================================================
# NLTK / CMU dictionary loading
# ============================================================================

try:
    import nltk
    from nltk.corpus import cmudict
    nltk.data.path.append(__import__('os').path.expanduser('~/nltk_data'))
    try:
        cmu = cmudict.dict()
        print(f"CMU dictionary loaded: {len(cmu)} words.", file=sys.stderr)
    except LookupError:
        print("Downloading CMU dictionary...", file=sys.stderr)
        nltk.download('cmudict')
        cmu = cmudict.dict()
        print(f"CMU dictionary loaded: {len(cmu)} words.", file=sys.stderr)
except ImportError:
    print("Error: nltk is not installed. Run: pip install nltk", file=sys.stderr)
    sys.exit(1)


# ============================================================================
# ARPAbet → synthSYL simplified notation mapping (corrected version)
# ============================================================================
#
# Each ARPAbet symbol maps to a synthSYL simplified key that is also
# present in the package's ``SIMPLE_TO_IPA`` dictionary.  This ensures
# the adapter's output can be directly processed by the package pipeline.
#
# The mapping preserves the same phonemic categories as the package:
#   - Vowels:         a, è, O, i, u, é, o
#   - Diphthongs:     mapped to their primary vowel quality
#   - Consonants:     b, d, g, p, t, k, m, n, f, v, s, z, l, r, S, Z
#   - Glides:         w, j
#
# Notable ARPAbet conventions:
#   - HH (voiceless glottal fricative) → 'f': the package's simplified
#     notation does not distinguish /h/ from /f/ at the segment level;
#     both map to the same fricative class.  This is a known limitation
#     of the simplified notation.
#   - DH (voiced dental fricative /ð/) → 'z': the package does not
#     have a separate /ð/ entry; it is merged with /z/.
#   - TH (voiceless dental fricative /θ/) → 's': similarly merged with /s/.
#   - NG (velar nasal /ŋ/) → 'n': the package does not distinguish /ŋ/
#     from /n/ in its simplified notation.

ARPABET_TO_SIMPLE: dict[str, str] = {
    # Vowels
    'AA': 'a',   'AE': 'è', 'AH': 'a',  'AO': 'O',
    'EH': 'è',   'ER': 'è', 'IH': 'i',  'IY': 'i',
    'UH': 'u',   'UW': 'u',
    # Diphthongs (mapped to primary vowel quality)
    'AW': 'a',   'AY': 'a', 'EY': 'é',  'OW': 'o',  'OY': 'o',
    # Consonants
    'B':  'b',   'CH': 'S', 'D':  'd',  'DH': 'z',  'F':  'f',
    'G':  'g',   'HH': 'f', 'JH': 'Z',  'K':  'k',  'L':  'l',
    'M':  'm',   'N':  'n', 'NG': 'n',  'P':  'p',  'R':  'r',
    'S':  's',   'SH': 'S', 'T':  't',  'TH': 's',  'V':  'v',
    'W':  'w',   'Y':  'j', 'Z':  'z',  'ZH': 'Z',
}

# Phonemes to ignore (stress markers from ARPAbet: 0=unstressed, 1=primary, 2=secondary)
IGNORE_PHONEMES: set[str] = {'0', '1', '2'}

# Validate that all output symbols of ARPABET_TO_SIMPLE are recognised
# by the package.  The pipeline's tokeniser (_match_longest_token in
# parsing.py) checks both SIMPLE_TO_IPA and PANPHON_DB, so a symbol is
# valid if it appears in either table.
_VALID_SEGMENT_KEYS: set[str] = set(SIMPLE_TO_IPA.keys()) | set(PANPHON_DB.keys())
_UNMAPPED_KEYS: set[str] = set(ARPABET_TO_SIMPLE.values()) - _VALID_SEGMENT_KEYS
if _UNMAPPED_KEYS:
    import warnings
    warnings.warn(
        f"panphon_adapter: the following ARPABET output symbols are not "
        f"recognised by the package (neither in SIMPLE_TO_IPA nor in "
        f"PANPHON_DB): {_UNMAPPED_KEYS}.  These phonemes will not be "
        f"processed by the package pipeline.",
        stacklevel=2,
    )


# ============================================================================
# Conversion functions
# ============================================================================

def arpabet_to_simple(phonemes: list[str]) -> str:
    """Convert an ARPAbet phoneme list (with stress markers) to a
    simplified synthSYL string.

    Stress digits (0, 1, 2) are stripped from the end of each ARPAbet
    symbol before lookup.  Unrecognised phonemes are silently ignored.

    Parameters
    ----------
    phonemes : list[str]
        ARPAbet phoneme symbols, possibly with trailing stress digits
        (e.g. ``['AE', 'T', 'IH0', 'N']``).

    Returns
    -------
    str
        Concatenated simplified notation (e.g. ``'ètin'``).
    """
    result: list[str] = []
    for ph in phonemes:
        if not isinstance(ph, str):
            ph = str(ph)
        ph_clean = re.sub(r'[0-2]$', '', ph)
        if ph_clean in IGNORE_PHONEMES:
            continue
        simple = ARPABET_TO_SIMPLE.get(ph_clean)
        if simple is not None:
            result.append(simple)
        # Unknown phonemes are silently ignored (preserving original behaviour)
    return ''.join(result)


# ============================================================================
# Text processing
# ============================================================================

def tokenize_text(text: str) -> list[str]:
    """Tokenize English text into words (punctuation removed).

    Parameters
    ----------
    text : str
        Input English text.

    Returns
    -------
    list[str]
        Lower-cased word tokens.
    """
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.lower().split()


def transform_phrase(phrase: str, cmu_dict: dict,
                     unknown_placeholder: str = '',
                     verbose: bool = False) -> str:
    """Transform an English phrase into a string of simplified phonemes.

    Each word is looked up in the CMU pronouncing dictionary.  The first
    pronunciation variant is used.  Unknown words are silently skipped
    or replaced by *unknown_placeholder* if provided.

    Parameters
    ----------
    phrase : str
        English text to convert.
    cmu_dict : dict
        CMU pronouncing dictionary (from ``nltk.corpus.cmudict.dict()``).
    unknown_placeholder : str
        String to insert for unknown words (default: empty → skip).
    verbose : bool
        If True, print warnings for unknown words and empty conversions.

    Returns
    -------
    str
        Space-separated simplified phoneme strings, one per word.
    """
    words = tokenize_text(phrase)
    transformed_parts: list[str] = []
    for word in words:
        if word in cmu_dict:
            entry = cmu_dict[word]
            # NLTK returns a list of pronunciations; each pronunciation
            # is a list of ARPAbet symbols.  Use the first variant.
            if isinstance(entry, list) and len(entry) > 0:
                if isinstance(entry[0], list):
                    phonemes = entry[0]
                else:
                    phonemes = entry
            else:
                phonemes = entry
            simple = arpabet_to_simple(phonemes)
            if simple:
                transformed_parts.append(simple)
            else:
                if verbose:
                    print(f"WARNING: no symbol for '{word}'", file=sys.stderr)
                if unknown_placeholder:
                    transformed_parts.append(unknown_placeholder)
        else:
            # Unknown word: could attempt simple transliteration (e.g. letters)
            if verbose:
                print(f"WARNING: word '{word}' not found in CMU", file=sys.stderr)
            if unknown_placeholder:
                transformed_parts.append(unknown_placeholder)
            # Otherwise skip the word (preserving original behaviour)
    # Space-separated output (pause mode between words)
    return ' '.join(transformed_parts)


# ============================================================================
# CLI entry point
# ============================================================================

def main() -> None:
    """Command-line entry point for the TIMIT → PanPhon adapter."""
    parser = argparse.ArgumentParser(
        description='TIMIT → PanPhon converter (NLTK CMU version)'
    )
    parser.add_argument('input', help='Input file (one phrase per line)')
    parser.add_argument('output', help='Output file (transformed text)')
    parser.add_argument('--verbose', action='store_true',
                        help='Display more information')
    args = parser.parse_args()

    # Read input phrases
    with open(args.input, 'r', encoding='utf-8') as f:
        phrases = [line.strip() for line in f if line.strip()]
    print(f"{len(phrases)} phrases loaded.", file=sys.stderr)

    # Transform each phrase
    output_lines: list[str] = []
    for idx, phrase in enumerate(phrases, start=1):
        transformed = transform_phrase(phrase, cmu,
                                       unknown_placeholder='',
                                       verbose=args.verbose)
        if not transformed:
            transformed = ' '
        output_lines.append(f"{idx}: {transformed}")
        if args.verbose and idx % 10 == 0:
            print(f"Processing: {idx}/{len(phrases)}", file=sys.stderr)

    # Write output file
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    print(f"Output file generated: {args.output} ({len(output_lines)} lines)",
          file=sys.stderr)


if __name__ == '__main__':
    main()
