"""
 Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import re
from fractions import Fraction
from typing import Iterator, List, Match, Optional, Union

from more_itertools import windowed

class EnglishNumberNormalizer:
    """
    A robust English number normalization utility that converts spelled-out numbers
    and numeric expressions into standard Arabic numeral forms.

    Features:
        - Converts written numbers (e.g., "twenty five") to digits ("25")
        - Handles suffixes like ordinals ("first", "101st") and decades ("1960s")
        - Converts currency expressions (e.g., "$20 million" to "20000000 dollars")
        - Supports expressions like "one oh one" to "101"
        - Deals with multipliers (hundred, thousand, million, etc.)
        - Supports common filler modifiers like "double" and "triple"
        - Preserves and normalizes special symbols (%, ¢, $, £, €)
        - Configurable for integrating into ASR post-processing or NLP pipelines

    Attributes:
        zeros (set): Set of zero-word variants like 'o' and 'zero'.
        ones (dict): Mapping of one-to-nineteen words to their numeric values.
        ones_plural (dict): Plural forms of one-to-nineteen words.
        ones_ordinal (dict): Ordinal forms (e.g., 'first', 'second', etc.).
        ones_suffixed (dict): Combined plural and ordinal forms.
        tens (dict): Tens words (e.g., 'twenty', 'thirty') mapped to values.
        tens_plural (dict): Plural forms of tens.
        tens_ordinal (dict): Ordinal forms of tens (e.g., 'twentieth').
        tens_suffixed (dict): Combined plural and ordinal tens.
        multipliers (dict): Magnitude multipliers (e.g., 'million', 'billion').
        multipliers_plural (dict): Plural forms of multipliers.
        multipliers_ordinal (dict): Ordinal forms of multipliers.
        multipliers_suffixed (dict): Combined plural and ordinal multipliers.
        preceding_prefixers (dict): Words that act as numeric prefixes (e.g., 'minus').
        following_prefixers (dict): Words that follow numbers (e.g., 'dollars', 'cents').
        prefixes (set): All recognized prefix symbols.
        suffixers (dict): Postfix modifiers such as 'percent'.
        specials (set): Special control words like 'and', 'point', 'double'.
        decimals (set): Words allowed in decimal numbers.
        words (set): All valid words handled by the normalizer.
        literal_words (set): Words like 'one' or 'ones' that may be left unconverted.
    """

    def __init__(self) -> None:
        self._init_digits()
        self._init_tens()
        self._init_multipliers()
        self._init_prefix_suffix()
        self._init_specials()
        self._build_vocab()

    def _init_digits(self) -> None:
        """Initializes mappings for zero, one-to-nineteen, plural, and ordinal forms."""
        self.zeros = {"o", "zero"}
        self.ones = {name: i for i, name in enumerate([
            "one", "two", "three", "four", "five", "six",
            "seven", "eight", "nine", "ten", "eleven", "twelve",
            "thirteen", "fourteen", "fifteen", "sixteen",
            "seventeen", "eighteen", "nineteen"
        ], start=1)}
        self.ones_plural = {
            "sixes" if name == "six" else name + "s": (value, "s")
            for name, value in self.ones.items()
        }
        self.ones_ordinal = {
            "zeroth": (0, "th"),
            "first": (1, "st"),
            "second": (2, "nd"),
            "third": (3, "rd"),
            "fifth": (5, "th"),
            "twelfth": (12, "th"),
            **{
                name + ("h" if name.endswith("t") else "th"): (val, "th")
                for name, val in self.ones.items()
                if val > 3 and val not in {5, 12}
            },
        }
        self.ones_suffixed = {**self.ones_plural, **self.ones_ordinal}

    def _init_tens(self) -> None:
        """Initializes mappings for tens (twenty, thirty, etc.), including plural and ordinal forms."""
        self.tens = {
            "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
            "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
        }
        self.tens_plural = {
            name.replace("y", "ies"): (val, "s") for name, val in self.tens.items()
        }
        self.tens_ordinal = {
            name.replace("y", "ieth"): (val, "th") for name, val in self.tens.items()
        }
        self.tens_suffixed = {**self.tens_plural, **self.tens_ordinal}

    def _init_multipliers(self) -> None:
        """Initializes mappings for large number multipliers like thousand, million, etc., including plural and ordinal forms."""
        self.multipliers = {
            "hundred": 100,
            "thousand": 1_000,
            "million": 1_000_000,
            "billion": 1_000_000_000,
            "trillion": 1_000_000_000_000,
            "quadrillion": 1_000_000_000_000_000,
            "quintillion": 1_000_000_000_000_000_000,
            "sextillion": 1_000_000_000_000_000_000_000,
            "septillion": 1_000_000_000_000_000_000_000_000,
            "octillion": 1_000_000_000_000_000_000_000_000_000,
            "nonillion": 1_000_000_000_000_000_000_000_000_000_000,
            "decillion": 1_000_000_000_000_000_000_000_000_000_000_000,
        }
        self.multipliers_plural = {
            name + "s": (val, "s") for name, val in self.multipliers.items()
        }
        self.multipliers_ordinal = {
            name + "th": (val, "th") for name, val in self.multipliers.items()
        }
        self.multipliers_suffixed = {
            **self.multipliers_plural,
            **self.multipliers_ordinal,
        }

    def _init_prefix_suffix(self) -> None:
        """Initializes mappings for prefix symbols (positive, negative) and suffix terms (currency, percent)."""
        self.preceding_prefixers = {
            "minus": "-", "negative": "-",
            "plus": "+", "positive": "+",
        }
        self.following_prefixers = {
            "pound": "£", "pounds": "£",
            "euro": "€", "euros": "€",
            "dollar": "$", "dollars": "$",
            "cent": "¢", "cents": "¢",
        }
        self.prefixes = set(self.preceding_prefixers.values()) | set(self.following_prefixers.values())
        self.suffixers = {
            "per": {"cent": "%"},
            "percent": "%",
        }

    def _init_specials(self) -> None:
        """Initializes special control words such as 'and', 'double', 'point'."""
        self.specials = {"and", "double", "triple", "point"}
        self.literal_words = {"one", "ones"}

    def _build_vocab(self) -> None:
        """Builds the full set of recognized words from all category dictionaries."""
        self.decimals = {*self.ones, *self.tens, *self.zeros}
        self.words = set().union(
            self.zeros,
            self.ones,
            self.ones_suffixed,
            self.tens,
            self.tens_suffixed,
            self.multipliers,
            self.multipliers_suffixed,
            self.preceding_prefixers,
            self.following_prefixers,
            self.suffixers,
            self.specials,
        )


    def process_words(self, words: List[str]) -> Iterator[str]:
        """
        Process a list of tokenized words and convert recognized number patterns
        (e.g., 'twenty five', 'one oh one', 'ten thousand dollars') into numeric strings.

        Args:
            words (List[str]): A tokenized list of words from a text or transcript.

        Yields:
            Iterator[str]: Converted or unchanged word tokens, as normalized string values.
        """
        prefix: Optional[str] = None
        value: Optional[Union[str, int]] = None
        skip = False

        def to_fraction(s: str) -> Optional[Fraction]:
            try:
                return Fraction(s)
            except ValueError:
                return None

        def output(result: Union[str, int]) -> str:
            nonlocal prefix, value
            result = str(result)
            if prefix is not None:
                result = prefix + result
            value = None
            prefix = None
            return result

        if len(words) == 0:
            return

        for prev, current, next in windowed([None] + words + [None], 3):
            if skip:
                skip = False
                continue

            next_is_numeric = next is not None and re.match(r"^\d+(\.\d+)?$", next)
            has_prefix = current[0] in self.prefixes
            current_without_prefix = current[1:] if has_prefix else current
            if re.match(r"^\d+(\.\d+)?$", current_without_prefix):
                # arabic numbers (potentially with signs and fractions)
                f = to_fraction(current_without_prefix)
                assert f is not None
                if value is not None:
                    if isinstance(value, str) and value.endswith("."):
                        # concatenate decimals / ip address components
                        value = str(value) + str(current)
                        continue
                    else:
                        yield output(value)

                prefix = current[0] if has_prefix else prefix
                if f.denominator == 1:
                    value = f.numerator  # store integers as int
                else:
                    value = current_without_prefix
            elif current not in self.words:
                # non-numeric words
                if value is not None:
                    yield output(value)
                yield output(current)
            elif current in self.zeros:
                value = str(value or "") + "0"
            elif current in self.ones:
                ones = self.ones[current]

                if value is None:
                    value = ones
                elif isinstance(value, str) or prev in self.ones:
                    if (
                        prev in self.tens and ones < 10
                    ):  # replace the last zero with the digit
                        assert value[-1] == "0"
                        value = value[:-1] + str(ones)
                    else:
                        value = str(value) + str(ones)
                elif ones < 10:
                    if value % 10 == 0:
                        value += ones
                    else:
                        value = str(value) + str(ones)
                else:  # eleven to nineteen
                    if value % 100 == 0:
                        value += ones
                    else:
                        value = str(value) + str(ones)
            elif current in self.ones_suffixed:
                # ordinal or cardinal; yield the number right away
                ones, suffix = self.ones_suffixed[current]
                if value is None:
                    yield output(str(ones) + suffix)
                elif isinstance(value, str) or prev in self.ones:
                    if prev in self.tens and ones < 10:
                        assert value[-1] == "0"
                        yield output(value[:-1] + str(ones) + suffix)
                    else:
                        yield output(str(value) + str(ones) + suffix)
                elif ones < 10:
                    if value % 10 == 0:
                        yield output(str(value + ones) + suffix)
                    else:
                        yield output(str(value) + str(ones) + suffix)
                else:  # eleven to nineteen
                    if value % 100 == 0:
                        yield output(str(value + ones) + suffix)
                    else:
                        yield output(str(value) + str(ones) + suffix)
                value = None
            elif current in self.tens:
                tens = self.tens[current]
                if value is None:
                    value = tens
                elif isinstance(value, str):
                    value = str(value) + str(tens)
                else:
                    if value % 100 == 0:
                        value += tens
                    else:
                        value = str(value) + str(tens)
            elif current in self.tens_suffixed:
                # ordinal or cardinal; yield the number right away
                tens, suffix = self.tens_suffixed[current]
                if value is None:
                    yield output(str(tens) + suffix)
                elif isinstance(value, str):
                    yield output(str(value) + str(tens) + suffix)
                else:
                    if value % 100 == 0:
                        yield output(str(value + tens) + suffix)
                    else:
                        yield output(str(value) + str(tens) + suffix)
            elif current in self.multipliers:
                multiplier = self.multipliers[current]
                if value is None:
                    value = multiplier
                elif isinstance(value, str) or value == 0:
                    f = to_fraction(value)
                    p = f * multiplier if f is not None else None
                    if f is not None and p.denominator == 1:
                        value = p.numerator
                    else:
                        yield output(value)
                        value = multiplier
                else:
                    before = value // 1000 * 1000
                    residual = value % 1000
                    value = before + residual * multiplier
            elif current in self.multipliers_suffixed:
                multiplier, suffix = self.multipliers_suffixed[current]
                if value is None:
                    yield output(str(multiplier) + suffix)
                elif isinstance(value, str):
                    f = to_fraction(value)
                    p = f * multiplier if f is not None else None
                    if f is not None and p.denominator == 1:
                        yield output(str(p.numerator) + suffix)
                    else:
                        yield output(value)
                        yield output(str(multiplier) + suffix)
                else:  # int
                    before = value // 1000 * 1000
                    residual = value % 1000
                    value = before + residual * multiplier
                    yield output(str(value) + suffix)
                value = None
            elif current in self.preceding_prefixers:
                # apply prefix (positive, minus, etc.) if it precedes a number
                if value is not None:
                    yield output(value)

                if next in self.words or next_is_numeric:
                    prefix = self.preceding_prefixers[current]
                else:
                    yield output(current)
            elif current in self.following_prefixers:
                # apply prefix (dollars, cents, etc.) only after a number
                if value is not None:
                    prefix = self.following_prefixers[current]
                    yield output(value)
                else:
                    yield output(current)
            elif current in self.suffixers:
                # apply suffix symbols (percent -> '%')
                if value is not None:
                    suffix = self.suffixers[current]
                    if isinstance(suffix, dict):
                        if next in suffix:
                            yield output(str(value) + suffix[next])
                            skip = True
                        else:
                            yield output(value)
                            yield output(current)
                    else:
                        yield output(str(value) + suffix)
                else:
                    yield output(current)
            elif current in self.specials:
                if next not in self.words and not next_is_numeric:
                    # apply special handling
                    # only if the next word can be numeric
                    if value is not None:
                        yield output(value)
                    yield output(current)
                elif current == "and":
                    # ignore "and" after hundreds, thousands, etc.
                    if prev not in self.multipliers:
                        if value is not None:
                            yield output(value)
                        yield output(current)
                elif current == "double" or current == "triple":
                    if next in self.ones or next in self.zeros:
                        repeats = 2 if current == "double" else 3
                        ones = self.ones.get(next, 0)
                        value = str(value or "") + str(ones) * repeats
                        skip = True
                    else:
                        if value is not None:
                            yield output(value)
                        yield output(current)
                elif current == "point":
                    if next in self.decimals or next_is_numeric:
                        value = str(value or "") + "."
                else:
                    # should all have been covered at this point
                    raise ValueError(f"Unexpected token: {current}")
            else:
                # all should have been covered at this point
                raise ValueError(f"Unexpected token: {current}")

        if value is not None:
            yield output(value)

    def preprocess(self, s: str) -> str:
        """
        Preprocess a string to normalize patterns that could affect numeric interpretation.

        This method performs the following:

        - Replaces patterns like "<number> and a half" with "<number> point five"
          if the preceding word is a recognized number or multiplier.
        - Inserts spaces between number-letter boundaries (e.g., "20th" → "20 th")
          to improve tokenization.
        - Removes spaces between numbers and suffixes (e.g., "20 th" → "20th").

        Args:
            s (str): The raw input string.

        Returns:
            str: A preprocessed string suitable for further normalization.
        """
        # replace "<number> and a half" with "<number> point five"
        results = []

        segments = re.split(r"\band\s+a\s+half\b", s)
        for i, segment in enumerate(segments):
            if len(segment.strip()) == 0:
                continue
            if i == len(segments) - 1:
                results.append(segment)
            else:
                results.append(segment)
                last_word = segment.rsplit(maxsplit=2)[-1]
                if last_word in self.decimals or last_word in self.multipliers:
                    results.append("point five")
                else:
                    results.append("and a half")

        s = " ".join(results)

        # put a space at number/letter boundary
        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)

        # but remove spaces which could be a suffix
        s = re.sub(r"([0-9])\s+(st|nd|rd|th|s)\b", r"\1\2", s)

        return s

    def postprocess(self, s: str) -> str:
        """
        Postprocess the normalized string to refine currency and singular word forms.

        This method applies the following transformations:
            - Converts patterns like "$2 and ¢7" into "$2.07" (combining integer and cent parts).
            - Converts "$0.xx" into "¢xx" when appropriate (extracting cents).
            - Replaces standalone "1" or "1s" with "one" or "ones" for improved readability.

        Args:
            s (str): The normalized string to be postprocessed.

        Returns:
            str: A cleaned-up string with currency and linguistic adjustments.
        """
        def combine_cents(m: Match) -> str:
            try:
                currency = m.group(1)
                integer = m.group(2)
                cents = int(m.group(3))
                return f"{currency}{integer}.{cents:02d}"
            except ValueError:
                return m.string

        def extract_cents(m: Match) -> str:
            try:
                return f"¢{int(m.group(1))}"
            except ValueError:
                return m.string

        # apply currency postprocessing; "$2 and ¢7" -> "$2.07"
        s = re.sub(r"([€£$])([0-9]+) (?:and )?¢([0-9]{1,2})\b", combine_cents, s)
        s = re.sub(r"[€£$]0.([0-9]{1,2})\b", extract_cents, s)

        # write "one(s)" instead of "1(s)", just for the readability
        s = re.sub(r"\b1(s?)\b", r"one\1", s)

        return s

    def __call__(self, s: str) -> str:
        """
        Normalize a string containing spelled-out English numbers into numeric form.

        This is the main entry point for using the normalizer. It applies:
            1. Preprocessing (e.g., handling "and a half", spacing around suffixes)
            2. Word-level normalization via `process_words`
            3. Postprocessing (e.g., combining "$2 and ¢7" into "$2.07", replacing "1" with "one")

        Args:
            s (str): The raw input string to normalize.

        Returns:
            str: The fully normalized string with numbers in standard Arabic format.
        """
        s = self.preprocess(s)
        s = " ".join(word for word in self.process_words(s.split()) if word is not None)
        s = self.postprocess(s)

        return s


class EnglishReverseNumberNormalizer(EnglishNumberNormalizer):
    """
    A reverse number normalizer that approximates the inverse of EnglishNumberNormalizer.

    Converts Arabic numerals (e.g., '365') back into spelled-out English forms (e.g., 'three hundred sixty five').
    This is useful for comparing Whisper's output to ASRs that cannot produce numerals directly.

    Motivation:
        Whisper outputs rich numeric expressions like "$20" or "50%", which many ASRs cannot generate.
        This class converts Whisper's output back into spoken-word forms to ensure fair comparison.

    Examples:
        - "365" -> "three hundred sixty five"
        - "$20" -> "twenty dollars"
        - "50%" -> "fifty percent"
        - "12th" -> "twelfth"
        - "12s" -> "twelves"
        - "90th" -> "ninetieth"
        - "90s" -> "nineties"
        - "70 000" -> "seventy thousand" (special case)

    Caveats:
        - Only supports numbers in the range 0–1000
        - Does not handle signs like '+' or '-'
        - Some ambiguity is inherent (e.g., "100" → "one hundred" vs. "a hundred")

    Attributes:
        int_to_ones (dict[int, str]):
            Mapping from integer values (1–19) to their spelled-out word forms.
        int_to_tens (dict[int, str]):
            Mapping from tens values (20, 30, ..., 90) to their word forms.
        str_to_ones_suffixed (dict[str, str]):
            Mapping from numeric strings with suffixes (e.g., '12th', '12s') to their spoken equivalents.
        str_to_tens_suffixed (dict[str, str]):
            Mapping from suffixed tens (e.g., '90s', '90th') to spelled-out versions like 'nineties', 'ninetieth'.
    """
    def __init__(self) -> None:
        super().__init__()
        # Reverse dictionaries
        self.int_to_ones = {v: k for k, v in self.ones.items()}
        self.int_to_tens = {v: k for k, v in self.tens.items()}

        # 11th -> eleventh etc.
        self.str_to_ones_suffixed = {
            str(n) + s: k for k, (n, s) in self.ones_suffixed.items()
        }
        # 20s -> twenties etc.
        self.str_to_tens_suffixed = {
            str(n) + s: k for k, (n, s) in self.tens_suffixed.items()
        }

    def __call__(self, s: str) -> str:
        """
        Converts numeric expressions in a string back to their approximate spelled-out equivalents.

        Rewrites:
            - Currency symbols (e.g., "$20" → "twenty dollars")
            - Percent signs (e.g., "50%" → "fifty percent")
            - Ordinal suffixes (e.g., "12th" → "twelfth")
            - Plurals (e.g., "20s" → "twenties")

        Args:
            s (str): A string containing numeric expressions.

        Returns:
            str: The normalized string with numerals converted back to words.
        """
        # "$x[.y]" -> "x[.y] dollars"
        s = re.sub(r"\$(\d+(\.\d+)?)", r"\1 dollars", s)
        s = re.sub(r"(\d+(\.\d+)?)\$", r"\1 dollars", s)
        # "x[.y]"% -> "x[.y] percent"
        s = re.sub(r"(\d+(\.\d+)?)%", r"\1 percent", s)
        # note this doesn't handle cases such as -x or +x.

        def number_to_words(w: str) -> str:
            if w.isdigit():
                num = int(w)
                if w == "000":
                    return "thousand"  # will work in case of "70 000" -> "seventy thousand"
                if num == 0:
                    return "zero"
                elif num == 100:
                    return "hundred"
                elif 0 < num < 1000:
                    hundreds, remainder = divmod(num, 100)
                    tens, ones = divmod(remainder, 10)
                    h = (
                        [f"{self.int_to_ones[hundreds]} hundred"]
                        if hundreds > 0
                        else []
                    )
                    if 0 < remainder <= 19:
                        t = [self.int_to_ones[remainder]]
                        o = []
                    else:
                        t = [self.int_to_tens[tens * 10]] if tens > 0 else []
                        o = [self.int_to_ones[ones]] if ones > 0 else []
                    return " ".join(h + t + o)
                elif num == 1000:
                    return "thousand"
                else:
                    return w  # case not handled
            else:
                # suffixed numbers
                w = self.str_to_ones_suffixed.get(w, w)
                w = self.str_to_tens_suffixed.get(w, w)
                return w

        return " ".join(number_to_words(w) for w in s.split())
