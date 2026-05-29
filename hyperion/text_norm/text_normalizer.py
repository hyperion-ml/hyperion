"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import re
import unicodedata

import regex

# non-ASCII letters that are not separated by "NFKD" normalization
ADDITIONAL_DIACRITICS = {
    "œ": "oe",
    "Œ": "OE",
    "ø": "o",
    "Ø": "O",
    "æ": "ae",
    "Æ": "AE",
    "ß": "ss",
    "ẞ": "SS",
    "đ": "d",
    "Đ": "D",
    "ð": "d",
    "Ð": "D",
    "þ": "th",
    "Þ": "th",
    "ł": "l",
    "Ł": "L",
}


class BasicTextNormalizer:
    """
    A configurable text normalization class for Unicode text.

    This class supports:
    - Unicode normalization (NFKC/NFKD)
    - Removal of punctuation, symbols, and diacritics
    - Bracket/parenthesis content stripping
    - Optional grapheme-based character splitting

    Attributes:
        remove_punctuation (bool): Whether to remove punctuation characters.
        remove_symbols (bool): Whether to remove symbol characters.
        remove_diacritics (bool): Whether to remove diacritical marks.
        split_chars (bool): Whether to split characters into grapheme clusters.
        remove_map (str): Internal map of Unicode categories to remove.
    """

    def __init__(
        self,
        remove_punctuation: bool = True,
        remove_symbols: bool = True,
        remove_diacritics: bool = False,
        split_chars: bool = False,
    ) -> None:
        """
        Initialize the BasicTextNormalizer.

        Args:
            remove_punctuation (bool): If True, remove punctuation.
            remove_symbols (bool): If True, remove symbols.
            remove_diacritics (bool): If True, remove diacritics.
            split_chars (bool): If True, split text into Unicode grapheme clusters.
        """
        self.remove_punctuation = remove_punctuation
        self.remove_symbols = remove_symbols
        self.remove_diacritics = remove_diacritics
        self.split_chars = split_chars
        self.remove_map = ""
        if self.remove_diacritics:
            self.remove_map += "M"
        if self.remove_symbols:
            self.remove_map += "S"
        if self.remove_punctuation:
            self.remove_map += "P"

    def _normalize_unicode_text(self, text: str, keep: str) -> str:
        """
        Normalize and clean a Unicode string.

        Applies NFKC (or NFKD when removing diacritics),
        and removes specified categories like marks (M), symbols (S), and punctuation (P).

        Args:
            text (str): Input text to normalize.
            keep (str): Characters to keep unchanged, even if they match removal rules.

        Returns:
            str: Normalized and cleaned text.
        """
        if self.remove_diacritics:
            normalized = unicodedata.normalize("NFKD", text)
            "".join(
                (
                    c
                    if c in keep
                    else (
                        ADDITIONAL_DIACRITICS[c]
                        if c in ADDITIONAL_DIACRITICS
                        else (
                            ""
                            if unicodedata.category(c) == "Mn"
                            else (
                                " "
                                if unicodedata.category(c)[0] in self.remove_map
                                else c
                            )
                        )
                    )
                )
                for c in normalized
            )
        normalized = unicodedata.normalize("NFKC", text)
        return "".join(
            c if c in keep else " " if unicodedata.category(c)[0] in "MSP" else c
            for c in normalized
        )

    def __call__(self, text: str) -> str:
        """
        Apply normalization pipeline to a string.

        This includes:
        - Lowercasing
        - Replacing unicode apostrophes
        - Removing bracketed/parenthesized content
        - Unicode category-based cleaning
        - Whitespace normalization
        - Optional character splitting

        Args:
            text (str): Input string.

        Returns:
            str: Normalized string.
        """
        text = text.lower()
        text = re.sub(r"\u2019", "'", text)
        # remove words between brackets
        text = re.sub(r"[<\[][^>\]]*[>\]]", "", text)
        # remove words between parenthesis
        text = re.sub(r"\(([^)]+?)\)", "", text)
        text = self._normalize_unicode_text(text).lower()
        # remove trailing/leading spaces and tabs
        text = re.sub("^[ \t]+|[ \t]+$", "", text)
        # remove duplicated spaces
        text = re.sub(r"\s+", " ", text)

        if self.split_chars:
            s = " ".join(regex.findall(r"\X", s, regex.U))
        return text
