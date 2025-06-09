"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import re

import regex

from .english_number_normalizers import (
    EnglishNumberNormalizer,
    EnglishReverseNumberNormalizer,
)
from .spelling_normalizer import SpellingNormalizer
from .text_normalizer import BasicTextNormalizer


class EnglishTextNormalizer(BasicTextNormalizer):
    """
    This is a modified version of the Whisper text normalizer designed to enhance compatibility
    across various ASRs.

    Key features:

    1. Idempotency: output is unchanged with repeated application.
    2. The original Whisper-tailored number normalization is replaced with one that is compatible with
        other ASR systems, mapping numerals into spelled-out numbers.
        See EnglishReverseNumberNormalizer for details and limitations.
    3. Filler words are removed by default, similar to the original normalizer: ['hmm', 'uh', 'ah', 'eh'].
        This is for compatibility with ASRs trained to ignore these.
    4. Added normalization for some common words: okay -> ok, everyday -> every day etc.

    """

    def __init__(
        self,
        standardize_numbers=False,
        standardize_numbers_rev=True,
        remove_fillers=True,
        remove_punctuation: bool = True,
        remove_symbols: bool = True,
        remove_diacritics: bool = False,
        split_chars: bool = False,
    ):
        super().__init__(
            remove_punctuation=remove_punctuation,
            remove_symbols=remove_symbols,
            remove_diacritics=remove_diacritics,
            split_chars=split_chars,
        )
        self.replacers = {
            # common non verbal sounds are mapped to the similar ones
            r"\u2019": ("'"),
            r"\b(hm+)\b|\b(mhm)\b|\b(mm+)\b|\b(m+h)\b|\b(hm+)\b|\b(um+)\b|\b(uhm+)\b": (  # noqa e501
                "hmm"
            ),
            r"\b(a+h+)\b|\b(ha+)\b": "ah",
            r"[!?.]+(?=$|\s)": "",  # Okay.. --> okay
            r"\b(o+h+)\b|\b(h+o+)\b": "oh",
            r"\b(u+h+)\b|\b(h+u+)\b|\b(h+u+h+)\b": "uh",
            # common contractions
            r"\b(wi\sfi)\b": "wifi",
            r"\b(goin)\b": "going",
            r"\wi-fi\b": "wifi",
            r"\bwon't\b": "will not",
            r"\bcan't\b": "can not",
            r"\blet's\b": "let us",
            r"\bain't\b": "aint",
            r"\by'all\b": "you all",
            r"\bwanna\b": "want to",
            r"\bgotta\b": "got to",
            r"\bgonna\b": "going to",
            r"\bi'ma\b": "i am going to",
            r"\bimma\b": "i am going to",
            r"\bwoulda\b": "would have",
            r"\bcoulda\b": "could have",
            r"\bshoulda\b": "should have",
            r"\bma'am\b": "madam",
            r"\bokay\b": "ok",
            r"\bsetup\b": "set up",
            r"\beveryday\b": "every day",
            # contractions in titles/prefixes
            r"\bmr\b": "mister ",
            r"\bmrs\b": "missus ",
            r"\bst\b": "saint ",
            r"\bdr\b": "doctor ",
            r"\bprof\b": "professor ",
            r"\bcapt\b": "captain ",
            r"\bgov\b": "governor ",
            r"\bald\b": "alderman ",
            r"\bgen\b": "general ",
            r"\bsen\b": "senator ",
            r"\brep\b": "representative ",
            r"\bpres\b": "president ",
            r"\brev\b": "reverend ",
            r"\bhon\b": "honorable ",
            r"\basst\b": "assistant ",
            r"\bassoc\b": "associate ",
            r"\blt\b": "lieutenant ",
            r"\bcol\b": "colonel ",
            r"\bjr\b": "junior ",
            r"\bsr\b": "senior ",
            r"\besq\b": "esquire ",
            r"'d been\b": " had been",
            r"'s been\b": " has been",
            r"'d gone\b": " had gone",
            r"'s gone\b": " has gone",
            r"'d done\b": " had done",
            r"'s got\b": " has got",
            # general contractions
            r"n't\b": " not",
            r"'re\b": " are",
            r"'s\b": " is",
            r"'d\b": " would",
            r"'ll\b": " will",
            r"'t\b": " not",
            r"'ve\b": " have",
            r"'m\b": " am",
            "shan't": "shall not",
            "han't": "has not",
            "ain't": "ain not",
        }

        if standardize_numbers:
            self.standardize_numbers = EnglishNumberNormalizer()
            assert not standardize_numbers_rev
        else:
            self.standardize_numbers = None

        if standardize_numbers_rev:
            self.standardize_numbers_rev = EnglishReverseNumberNormalizer()
        else:
            self.standardize_numbers_rev = None

        self.standardize_spellings = SpellingNormalizer("bre_to_use")

        if remove_fillers:
            self.fillers = [
                "hmm",
                "uh",
                "ah",
                "eh",
            ]  # assumes replacers have been applied
        else:
            self.fillers = None

    def __call__(self, text: str):
        text = text.lower()

        # remove words between brackets
        text = re.sub(r"[<\[][^>\]]*[>\]]", "", text)
        # remove words between parenthesis
        text = re.sub(r"\(([^)]+?)\)", "", text)
        # when there's a space before an apostrophe
        text = re.sub(r"\s+'", "'", text)

        for pattern, replacement in self.replacers.items():
            text = re.sub(pattern, replacement, text)

        # remove commas between digits
        text = re.sub(r"(\d),(\d)", r"\1\2", text)
        # remove periods not followed by numbers
        text = re.sub(r"\.([^0-9]|$)", r" \1", text)

        # keep numeric symbols
        text = self._normalize_unicode_text(text, keep=".%$¢€£")

        if self.standardize_numbers is not None:
            text = self.standardize_numbers(text)

        if self.standardize_numbers_rev is not None:
            text = self.standardize_numbers_rev(text)

        text = self.standardize_spellings(text)
        # now remove prefix/suffix symbols
        # that are not preceded/followed by numbers
        text = re.sub(r"[.$¢€£]([^0-9])", r" \1", text)
        text = re.sub(r"([^0-9])%", r"\1 ", text)

        # remove filler words
        # motivation: these words are very common, yet hold little information in the majority of cases.
        # some ASR systems may ignore them by convention and will be penalized unfairly.
        if self.fillers:
            text = re.sub(r"\b(" + "|".join(self.fillers) + r")\b", "", text)

        # remove trailing/leading spaces and tabs
        text = re.sub("^[ \t]+|[ \t]+$", "", text)
        # replace any successive whitespaces with a space
        text = re.sub(r"\s+", " ", text)

        if self.split_chars:
            text = " ".join(regex.findall(r"\X", text, regex.U))
        return text
