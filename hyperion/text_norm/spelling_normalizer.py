"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import json
from pathlib import Path


class SpellingNormalizer:
    """
    A rule-based normalizer that replaces words in a string using a predefined JSON mapping.

    This class loads a word-to-word mapping from a JSON file and applies substitutions
    on input strings, replacing each word with its mapped value (if any).

    Example:
        If the mapping file contains {"colour": "color"}, then:
            SpellingNormalizer("british_to_american")("colour is nice") → "color is nice"

    Attributes:
        mapping (dict): A dictionary loaded from the specified JSON file, mapping input words to their normalized forms.
    """

    def __init__(self, mapping_name: str):
        mapping_path = str(Path(__file__).parent / f"{mapping_name}.json")
        self.mapping = json.load(open(mapping_path))

    def __call__(self, s: str):
        """
        Apply the loaded spelling normalization to an input string.

        Each word in the string is replaced by its corresponding value in the mapping
        if a match is found; otherwise, the original word is preserved.

        Args:
            s (str): The input string to normalize.

        Returns:
            str: The normalized string.
        """
        return " ".join(self.mapping.get(word, word) for word in s.split())
