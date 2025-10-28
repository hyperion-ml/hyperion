"""
Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional

import langcodes
import pycountry

__all__ = [
    "language_to_alpha2",
    "language_to_alpha3",
    "alphax_to_language",
    "dialect_to_alpha3",
    "alpha3_to_dialect",
]


def _standardize(identifier: str) -> Optional[str]:
    """Normalize free-form identifiers (names, codes) into langcodes tags."""
    # Normalize arbitrary language tags (names, alpha codes) to a canonical
    # `langcodes` representation when possible.
    try:
        return langcodes.standardize_tag(identifier)
    except Exception:
        return None


def _pycountry_lookup(identifier: str):
    """Run pycountry lookup with graceful failure handling."""
    # pycountry raises LookupError on misses; convert that into a `None`.
    try:
        return pycountry.languages.lookup(identifier)
    except LookupError:
        return None


def _pycountry_language_for_alpha3(code: str):
    """Return a pycountry record for ISO 639-3 terminology or bibliographic code."""
    # Handle both ISO 639-3 terminology (alpha_3) and bibliographic codes.
    lower = code.lower()
    return pycountry.languages.get(alpha_3=lower) or pycountry.languages.get(
        bibliographic=lower
    )


def _alpha3_from_record(record) -> Optional[str]:
    """Extract an ISO 639-3 code string from a pycountry language record."""
    if record is None:
        return None
    # Return the first available ISO 639-3 code variant present in the record.
    for attr in ("alpha_3", "alpha_3T", "alpha_3B", "terminology", "bibliographic"):
        code = getattr(record, attr, None)
        if code:
            return code.lower()
    return None


def _language_name(
    record, alpha2: Optional[str], alpha3: Optional[str]
) -> Optional[str]:
    """Return the best human-readable language name from available metadata."""
    for code in (alpha2, alpha3):
        if code:
            # Prefer langcodes display names because they are well-formatted.
            try:
                return langcodes.Language.get(code).display_name()
            except (LookupError, ValueError):
                continue

    if record is not None:
        for attr in ("common_name", "name"):
            name = getattr(record, attr, None)
            if name:
                # Fall back to names provided by pycountry record metadata.
                return name

    return None


@lru_cache(maxsize=None)
def language_to_alpha2(identifier: str) -> str:
    """Return the ISO 639-1 alpha-2 code for ``identifier``."""
    if not identifier or not identifier.strip():
        raise ValueError("language identifier must be a non-empty string")

    # Normalize whitespace so length checks and caching behave predictably.
    query = identifier.strip()

    if len(query) == 2 and query.isalpha():
        # Input already looks like an alpha-2 code; normalize casing and exit.
        return query.lower()

    if len(query) == 3 and query.isalpha():
        # Direct alpha-3 match; let pycountry map it to alpha-2.
        record = _pycountry_language_for_alpha3(query)
        if record and getattr(record, "alpha_2", None):
            return record.alpha_2.lower()

    # Try to coerce names or non-canonical tags into something we can parse.
    normalized = _standardize(query)
    if normalized:
        # `langcodes` canonical tag may already contain the alpha-2 code.
        lang_part = normalized.split("-", 1)[0]
        if len(lang_part) == 2 and lang_part.isalpha():
            return lang_part.lower()
        if len(lang_part) == 3 and lang_part.isalpha():
            # Canonical form resolved to alpha-3; map back with pycountry.
            record = _pycountry_language_for_alpha3(lang_part)
            if record and getattr(record, "alpha_2", None):
                return record.alpha_2.lower()

    # If standardization failed, ask langcodes to locate the best-matching language tag by name.
    # Ask langcodes to perform a name-based lookup if normalization was inconclusive.
    try:
        langcodes_result = langcodes.find(query)
    except LookupError:
        langcodes_result = None
    if langcodes_result:
        # `find` returns a tag (usually alpha-2); reuse it if possible.
        lang_part = langcodes_result.split("-", 1)[0]
        if len(lang_part) == 2 and lang_part.isalpha():
            return lang_part.lower()

    # Final attempt: rely on pycountry's fuzzy matching for variant names.
    # Final fallback: consult pycountry's lookup for any remaining matches.
    record = _pycountry_lookup(query)
    if record and getattr(record, "alpha_2", None):
        # Final fallback: pycountry fuzzy lookup matched a language name.
        return record.alpha_2.lower()

    raise ValueError(
        f"language identifier {identifier!r} cannot be converted to ISO 639-1"
    )


@lru_cache(maxsize=None)
def language_to_alpha3(identifier: str) -> str:
    """Return the ISO 639 alpha-3 code for ``identifier``."""
    if not identifier or not identifier.strip():
        raise ValueError("language identifier must be a non-empty string")

    # Strip whitespace so cache lookups share the same key.
    query = identifier.strip()

    if len(query) == 3 and query.isalpha():
        # Try to canonicalize three-letter tags, covering bibliographic aliases.
        normalized = _standardize(query)
        if normalized:
            lang_part = normalized.split("-", 1)[0]
            if len(lang_part) == 3 and lang_part.isalpha():
                return lang_part.lower()
        record = _pycountry_language_for_alpha3(query)
        alpha3 = _alpha3_from_record(record)
        if alpha3:
            return alpha3
        raise ValueError(
            f"language identifier {identifier!r} is not a valid ISO 639 alpha-3 code"
        )

    if len(query) == 2 and query.isalpha():
        # Straight alpha-2 input: first ask langcodes for a mapping.
        try:
            return langcodes.Language.get(query.lower()).to_alpha3().lower()
        except (LookupError, ValueError):
            record = pycountry.languages.get(alpha_2=query.lower())
            alpha3 = _alpha3_from_record(record)
            if alpha3:
                return alpha3

    normalized = _standardize(query)
    if normalized:
        lang_part = normalized.split("-", 1)[0]
        if len(lang_part) == 2 and lang_part.isalpha():
            try:
                return langcodes.Language.get(lang_part).to_alpha3().lower()
            except (LookupError, ValueError):
                # `langcodes` could not map it; fall back to pycountry.
                record = pycountry.languages.get(alpha_2=lang_part.lower())
                alpha3 = _alpha3_from_record(record)
                if alpha3:
                    return alpha3
        elif len(lang_part) == 3 and lang_part.isalpha():
            record = _pycountry_language_for_alpha3(lang_part)
            alpha3 = _alpha3_from_record(record)
            if alpha3:
                return alpha3
            # If no record was found, trust the normalized lang_part itself.
            return lang_part.lower()

    try:
        langcodes_result = langcodes.find(query)
    except LookupError:
        langcodes_result = None
    if langcodes_result:
        lang_part = langcodes_result.split("-", 1)[0]
        try:
            return langcodes.Language.get(lang_part).to_alpha3().lower()
        except (LookupError, ValueError):
            # Last attempt with pycountry if langcodes lacks the mapping.
            record = pycountry.languages.get(alpha_2=lang_part.lower())
            alpha3 = _alpha3_from_record(record)
            if alpha3:
                return alpha3

    record = _pycountry_lookup(query)
    alpha3 = _alpha3_from_record(record)
    if alpha3:
        # pycountry may still know the identifier (different language name).
        return alpha3

    raise ValueError(
        f"language identifier {identifier!r} cannot be converted to ISO 639-3"
    )


@lru_cache(maxsize=None)
def _alphax_to_language_cached(identifier: str) -> str:
    """Return the canonical language name for any language identifier."""
    if not identifier or not identifier.strip():
        raise ValueError("language identifier must be a non-empty string")

    # Strip whitespace so cache lookups share the same key.
    query = identifier.strip()
    record = None
    alpha2: Optional[str] = None
    alpha3: Optional[str] = None

    # Step 1: try the alpha-2 pathway (handles names and two-letter tags).
    try:
        alpha2 = language_to_alpha2(query)
    except ValueError:
        alpha2 = None
    else:
        # Resolve name starting from the alpha-2 code.
        record = pycountry.languages.get(alpha_2=alpha2)
        alpha3 = _alpha3_from_record(record)
        if alpha3 is None:
            try:
                alpha3 = language_to_alpha3(alpha2)
            except ValueError:
                alpha3 = None
        name = _language_name(record, alpha2, alpha3)
        if name:
            return name

    # Step 2: attempt resolution using alpha-3 codes or ISO 639-2 identifiers.
    try:
        alpha3 = language_to_alpha3(query)
    except ValueError:
        alpha3 = None
    else:
        # Resolve name starting from the alpha-3 code.
        record = _pycountry_language_for_alpha3(alpha3)
        alpha2 = getattr(record, "alpha_2", None)
        if alpha2:
            alpha2 = alpha2.lower()
        else:
            try:
                alpha2 = language_to_alpha2(alpha3)
            except ValueError:
                alpha2 = None
        name = _language_name(record, alpha2, alpha3)
        if name:
            return name

    try:
        langcodes_result = langcodes.find(query)
    except LookupError:
        langcodes_result = None
    if langcodes_result:
        # langcodes can match human-readable names (e.g. "French") to a tag.
        try:
            name = langcodes.Language.get(langcodes_result).display_name()
        except (LookupError, ValueError):
            name = None
        if name:
            return name

    record = _pycountry_lookup(query)
    if record:
        # As a final fallback, rely on pycountry's fuzzy search results.
        alpha2 = getattr(record, "alpha_2", None)
        if alpha2:
            alpha2 = alpha2.lower()
        alpha3 = _alpha3_from_record(record)
        name = _language_name(record, alpha2, alpha3)
        if name:
            return name

    # If none of the strategies above worked, surface a descriptive error.
    raise ValueError(
        f"language identifier {identifier!r} cannot be resolved to a language name"
    )


def alphax_to_language(identifier: str) -> str:
    """Return the canonical language name in lowercase for the given identifier."""
    return _alphax_to_language_cached(identifier).lower()


_DIALECT_EXACT_MAP = {
    "a savage texas gentleman": "usa",
    "a variety of texan english with some german influence that has undergone the cot caught merger": "usa",
    "afrikaans english": "zaf",
    "american south east georgia dialect": "usa",
    "australian english": "aus",
    "austrian": "aut",
    "bangladeshi": "bgd",
    "bangladeshi english": "bgd",
    "brazilian": "bra",
    "brazillian accent": "bra",
    "british": "gbr",
    "british accent": "gbr",
    "british english received pronunciation rp": "gbr",
    "bulgarian": "bgr",
    "california": "usa",
    "canadian english": "can",
    "caribbean canadian": "can",
    "central scottish": "gbr",
    "chichester": "gbr",
    "chinese": "chn",
    "chinese english": "chn",
    "colombian accent": "col",
    "cretan accent": "grc",
    "croatian english": "hrv",
    "czech": "cze",
    "czech accent": "cze",
    "east african khoja": "ken",
    "dutch": "nld",
    "dutch english": "nld",
    "east indian": "ind",
    "east london": "gbr",
    "egyptian": "egy",
    "england english": "gbr",
    "england english with a touch of canadian": "gbr",
    "england west county": "gbr",
    "england non native": "gbr",
    "english native greek speaker": "grc",
    "english county durham": "gbr",
    "english as second language russian as first": "rus",
    "english north of england": "gbr",
    "english with swiss german accent": "che",
    "filipino": "phl",
    "filipino english": "phl",
    "finnish": "fin",
    "french": "fra",
    "georgian english": "geo",
    "german english": "deu",
    "german german accent": "deu",
    "greek": "grc",
    "haitian creole": "hti",
    "hong kong english": "hkg",
    "hungarian": "hun",
    "hunglish": "hun",
    "i have a mild brooklyn accent": "usa",
    "indonesian": "idn",
    "indonesian english": "idn",
    "irish english": "irl",
    "israeli": "isr",
    "israeli english": "isr",
    "israeli accent": "isr",
    "italian": "ita",
    "japan english": "jpn",
    "japanese english": "jpn",
    "kazakhstan english": "kaz",
    "kenyan": "ken",
    "kenyan english": "ken",
    "kenyan english accent": "ken",
    "kiwi": "nzl",
    "korean": "kor",
    "latvian": "lva",
    "lebanese accent": "lbn",
    "liverpudlian english": "gbr",
    "malaysian english": "mys",
    "midatlantic": "usa",
    "mild northern england english": "gbr",
    "nepali": "npl",
    "new jerseyan": "usa",
    "new york city": "usa",
    "new zealand english": "nzl",
    "nigerian": "nga",
    "nigerian english": "nga",
    "non native speaker from france": "fra",
    "northern irish": "gbr",
    "northern irish english": "gbr",
    "northumbrian british english": "gbr",
    "norwegian": "nor",
    "polish": "pol",
    "polish accent": "pol",
    "polish english": "pol",
    "rhode island new england accent": "usa",
    "russian": "rus",
    "russian accent": "rus",
    "russian english": "rus",
    "scottish english": "gbr",
    "singaporean english": "sgp",
    "south african english": "zaf",
    "south african english accent": "zaf",
    "south australia": "aus",
    "south indian": "ind",
    "southern drawl": "usa",
    "spanish bilingual": "esp",
    "swedish english": "swe",
    "swedish accent": "swe",
    "swiss english": "che",
    "thai": "tha",
    "thai english": "tha",
    "turkish": "tur",
    "u k english": "gbr",
    "uk southern english": "gbr",
    "ukrainian": "ukr",
    "united states english": "usa",
    "united states english combined with european english": "usa",
    "vietnam": "vnm",
    "welsh english": "gbr",
    "wise canadian english": "can",
    "with heavy cantonese accent": "hkg",
    "yoruba": "nga",
    "american": "usa",
    "french accent": "fra",
    "french english": "fra",
    "light french accent": "fra",
    "mexican accent": "mex",
    "midwestern united states": "usa",
    "minor french accent": "fra",
    "nigeria english": "nga",
    "nigerian accent": "nga",
    "northen united states": "usa",
    "serbian": "srb",
    "strong latvian accent": "lva",
    "slovak outh african english accent": "zaf",
}

_DIALECT_AMBIGUOUS = {
    "a lo",
    "african accent",
    "blurpy",
    "danish british american blend",
    "east asian",
    "eastern european",
    "eastern european english",
    "european",
    "european accent",
    "european english",
    "generic european",
    "hispanic",
    "hispanic latino",
    "l2",
    "latin american accent",
    "latin american accent influenced by american english",
    "latin english",
    "latinamerican",
    "latino",
    "mix of american and british accent",
    "non native",
    "non native english",
    "north european english",
    "personal idiolect",
    "second tongue",
    "slavic",
    "slightly slurred due to age and alcohol consumption",
    "transnational englishes blend",
    "west african",
    "west indian",
    "western europe",
    "not a native speaker",
    "mostly american with some british and australian inflections",
}

_DIALECT_SUBSTRING_MAP = [
    ("texas", "usa"),
    ("brooklyn", "usa"),
    ("georgia", "usa"),
    ("california", "usa"),
    ("midwestern", "usa"),
    ("new york", "usa"),
    ("new jersey", "usa"),
    ("rhode island", "usa"),
    ("southern drawl", "usa"),
    ("united states", "usa"),
    ("canadian", "can"),
    ("northern irish", "gbr"),
    ("scottish", "gbr"),
    ("welsh", "gbr"),
    ("irish", "irl"),
    ("greek", "grc"),
    ("greece", "grc"),
    ("british", "gbr"),
    ("england", "gbr"),
    ("london", "gbr"),
    ("durham", "gbr"),
    ("northumbrian", "gbr"),
    ("kiwi", "nzl"),
    ("new zealand", "nzl"),
    ("australian", "aus"),
    ("australia", "aus"),
    ("cantonese", "hkg"),
    ("hong kong", "hkg"),
    ("latvian", "lva"),
    ("french", "fra"),
    ("mexican", "mex"),
    ("polish", "pol"),
    ("russian", "rus"),
    ("italian", "ita"),
    ("hungarian", "hun"),
    ("hunglish", "hun"),
    ("swiss", "che"),
    ("swedish", "swe"),
    ("thai", "tha"),
    ("vietnamese", "vnm"),
    ("vietnam", "vnm"),
    ("turkish", "tur"),
    ("filipino", "phl"),
    ("philippine", "phl"),
    ("georgian", "geo"),
    ("german", "deu"),
    ("hindi", "ind"),
    ("indian", "ind"),
    ("kenyan", "ken"),
    ("nigerian", "nga"),
    ("yoruba", "nga"),
    ("brazil", "bra"),
    ("bangladesh", "bgd"),
    ("bangladeshi", "bgd"),
    ("chinese", "chn"),
    ("croatian", "hrv"),
    ("czech", "cze"),
    ("dutch", "nld"),
    ("egypt", "egy"),
    ("finnish", "fin"),
    ("haitian", "hti"),
    ("indonesian", "idn"),
    ("israeli", "isr"),
    ("kazakh", "kaz"),
    ("korean", "kor"),
    ("lebanese", "lbn"),
    ("malaysian", "mys"),
    ("nepali", "npl"),
    ("norwegian", "nor"),
    ("singaporean", "sgp"),
    ("south african", "zaf"),
    ("spanish", "esp"),
    ("ukrainian", "ukr"),
    ("japan", "jpn"),
    ("japanese", "jpn"),
    ("colombian", "col"),
    ("bulgarian", "bgr"),
    ("estonian", "est"),
]


def _normalize_text(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())


def _dialect_region_to_alpha3(dialect_name: str) -> str:
    if not dialect_name or not dialect_name.strip():
        raise ValueError("dialect name must be a non-empty string")

    normalized = _normalize_text(dialect_name)

    if normalized in _DIALECT_AMBIGUOUS:
        raise ValueError(f"dialect description {dialect_name!r} is too ambiguous")

    if normalized in _DIALECT_EXACT_MAP:
        code = _DIALECT_EXACT_MAP[normalized]
        if code == "wld":
            raise ValueError(
                f"dialect description {dialect_name!r} is not tied to a specific ISO region"
            )
        return code

    for pattern, code in _DIALECT_SUBSTRING_MAP:
        if pattern in normalized:
            return code

    for candidate in (
        dialect_name,
        normalized,
        normalized.replace(" ", "_"),
        normalized.replace(" ", "-"),
    ):
        try:
            country = pycountry.countries.lookup(candidate)
        except LookupError:
            continue
        if country:
            return country.alpha_3.lower()

    tokens = normalized.split()
    for token in reversed(tokens):
        if len(token) < 3:
            continue
        try:
            country = pycountry.countries.lookup(token)
        except LookupError:
            continue
        else:
            return country.alpha_3.lower()

    raise ValueError(
        f"dialect description {dialect_name!r} cannot be matched to an ISO 3166 alpha-3 region code"
    )


def dialect_to_alpha3(language_identifier: str, dialect_name: str) -> str:
    """
    Convert a language identifier and dialect description into a compound dialect code.

    The resulting identifier uses the pattern ``<language_alpha3>-<region_alpha3>``, matching
    the convention used by datasets such as VCTK (e.g., ``eng-gbr`` for British English).

    Args:
        language_identifier: Language name or code (alpha-2, alpha-3, or full name).
        dialect_name: Free-form dialect or accent description.

    Returns:
        str: The composite dialect code.

    Raises:
        ValueError: If either the language or dialect cannot be resolved to ISO codes.
    """

    language_alpha3 = language_to_alpha3(language_identifier)
    region_alpha3 = _dialect_region_to_alpha3(dialect_name)
    return f"{language_alpha3}-{region_alpha3}"


def alpha3_to_dialect(code: str) -> tuple[str, str]:
    """
    Decode a composite dialect code (``<language_alpha3>-<region_alpha3>``) into human-readable names.

    Args:
        code: Composite dialect identifier such as ``eng-aus``.

    Returns:
        tuple[str, str]: A pair ``(language_name, dialect_name)`` in lowercase.

    Raises:
        ValueError: If the code is malformed or cannot be resolved.
    """

    if not code or not isinstance(code, str):
        raise ValueError("dialect code must be a non-empty string")

    parts = code.strip().split("-", 1)
    if len(parts) != 2:
        raise ValueError(
            f"dialect code {code!r} must follow the '<language>-<dialect>' pattern"
        )

    lang_part, region_part = parts
    if len(lang_part) != 3 or not lang_part.isalpha():
        raise ValueError(
            f"language component {lang_part!r} in {code!r} is not a valid ISO 639-3 code"
        )
    if len(region_part) != 3 or not region_part.isalpha():
        raise ValueError(
            f"dialect component {region_part!r} in {code!r} is not a valid ISO 3166-1 alpha-3 code"
        )

    language_name = alphax_to_language(lang_part)

    country = pycountry.countries.get(alpha_3=region_part.upper())
    if country is None:
        # Try historic/legacy entries if present.
        try:
            country = pycountry.countries.lookup(region_part)
        except LookupError:
            country = None
    if country is None:
        raise ValueError(
            f"dialect component {region_part!r} in {code!r} cannot be resolved to an ISO 3166-1 region"
        )

    dialect_name = getattr(country, "common_name", None) or getattr(
        country, "name", None
    )
    if not dialect_name:
        raise ValueError(
            f"dialect component {region_part!r} in {code!r} lacks an associated region name"
        )

    return language_name, dialect_name.lower()
