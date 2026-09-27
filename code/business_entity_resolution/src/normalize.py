"""Country-agnostic text normalization for business names and addresses.

All scripts (Devanagari, Tamil, accents, ...) are transliterated to ASCII with
anyascii. Indic transliteration drops inherent vowels ("shivam" -> "sivm"), so
comparisons that must survive that use `skeleton()`: a consonant key with
digraphs unified, vowels removed and doubled letters collapsed.
"""
import logging
import re
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from itertools import chain

import pandas as pd
from anyascii import anyascii

from data import ADDR, COUNTRY, ID, NAME
from logger import fmt_int, fmt_pct, get_logger

log = get_logger(__name__)

NORM_VERSION = "3.0"

# Legal/entity-type suffixes across US, India and France (and generic ones).
LEGAL_TOKENS = {
    "inc", "incorporated", "llc", "llp", "lp", "ltd", "limited", "corp",
    "corporation", "co", "company", "plc", "pvt", "private", "pllc", "pc",
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "gmbh", "ag", "bv",
    "prvt", "pvtltd", "opc",
}

NAME_ABBREV = {
    "intl": "international", "natl": "national", "mfg": "manufacturing",
    "svcs": "services", "svc": "service", "tech": "technology",
    "techs": "technologies", "mgmt": "management", "assoc": "associates",
    "bros": "brothers", "ent": "enterprises", "grp": "group",
    "sys": "systems", "dept": "department", "univ": "university",
    "ste": "sainte", "st": "saint",
}

ADDR_ABBREV = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue",
    "av": "avenue", "blvd": "boulevard", "bd": "boulevard", "dr": "drive",
    "ln": "lane", "ct": "court", "pl": "place", "sq": "square",
    "hwy": "highway", "pkwy": "parkway", "cir": "circle", "ter": "terrace",
    "apt": "apartment", "ste": "suite", "fl": "floor", "flr": "floor",
    "bldg": "building", "blk": "block", "nr": "near", "opp": "opposite",
    "mkt": "market", "ngr": "nagar", "clny": "colony", "sec": "sector",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    "no": "number", "chs": "chemin", "rte": "route", "imp": "impasse",
}

_PUNCT = re.compile(r"[^a-z0-9\s]")
_NUM = re.compile(r"\d+")

# --- consonant skeleton ------------------------------------------------------
_DIGRAPHS = [("tion", "sn"), ("sion", "sn"), ("sh", "s"), ("ch", "c"), ("ph", "f"),
             ("th", "t"), ("bh", "b"), ("dh", "d"), ("kh", "k"), ("gh", "g"),
             ("jh", "j"), ("ck", "k"), ("q", "k"), ("x", "ks"), ("w", "v"), ("z", "s")]
_C_SOFT = re.compile(r"c(?=[eiy])")
_NASAL = re.compile(r"m(?=[cdgjkstz])")
_VOWELS = re.compile(r"[aeiouy]")
_DOUBLE = re.compile(r"(.)\1+")


@lru_cache(maxsize=1 << 20)
def skeleton(tok: str) -> str:
    """Phonetic consonant key: 'shivam'->'svm', 'sivm'->'svm', 'private'->'prvt'."""
    if tok.isdigit():
        return tok
    s = tok
    for a, b in _DIGRAPHS:
        s = s.replace(a, b)
    s = _C_SOFT.sub("s", s).replace("c", "k")
    s = _NASAL.sub("n", s)
    s = _DOUBLE.sub(r"\1", _VOWELS.sub("", s))
    return s or tok[:1]


def name_skeleton(name: str) -> str:
    return " ".join(skeleton(t) for t in name.split())


LEGAL_SKEL = {skeleton(w) for w in LEGAL_TOKENS if len(w) >= 4}
# Short forms mapped to one canonical spelling before skeletonizing
LEGAL_CANON = {"pvt": "private", "prvt": "private", "ltd": "limited", "inc": "incorporated",
               "corp": "corporation", "co": "company", "pvtltd": "private limited"}


def _join_letters(toks: list[str]) -> list[str]:
    """Join runs of single letters: 'l l c' -> 'llc', 'f k' -> 'fk'."""
    out, run = [], []
    for t in toks:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            out.append("".join(run))
            run = []
        out.append(t)
    if run:
        out.append("".join(run))
    return out


def normalize_text(s: str) -> str:
    if not s.isascii():
        s = anyascii(s)
    s = _PUNCT.sub(" ", s.lower().replace("&", " and "))
    return " ".join(_join_letters(s.split()))


def _expand(tokens: list[str], mapping: dict) -> list[str]:
    return [mapping.get(t, t) for t in tokens]


def _is_legal(t: str) -> bool:
    return t in LEGAL_TOKENS or (len(t) >= 4 and skeleton(t) in LEGAL_SKEL)


def normalize_name(s: str) -> tuple[str, str]:
    """Return (core_name, legal_suffixes). Legal tokens are split off so that
    'Acme Pvt Ltd', 'ACME Private Limited' and transliterated 'praivet limited'
    share the same core. Legal tokens are stored as skeletons."""
    toks = _expand(normalize_text(s).split(), NAME_ABBREV)
    core = [t for t in toks if not _is_legal(t)]
    legal = [skeleton(w) for t in toks if _is_legal(t) for w in LEGAL_CANON.get(t, t).split()]
    if not core:  # name made only of legal tokens: keep it whole
        core = toks
    return " ".join(core), " ".join(sorted(set(legal)))


def normalize_address(s: str) -> str:
    return " ".join(_expand(normalize_text(s).split(), ADDR_ABBREV))


def extract_numbers(s: str) -> str:
    """Space-joined digit groups (house numbers, postcodes, PIN codes), in order."""
    return " ".join(_NUM.findall(anyascii(s) if not s.isascii() else s))


NORM_COLS = ["name_core", "legal", "addr_norm", "addr_nums", "country_norm"]
KEEP_COLS = [ID, COUNTRY] + NORM_COLS


def _normalize_chunk(task):
    """Worker: normalize one chunk of (names, addresses, countries)."""
    names, addrs, countries = task
    cols = ([], [], [], [], [])
    for n, a, c in zip(names, addrs, countries):
        core, legal = normalize_name(n)
        cols[0].append(core)
        cols[1].append(legal)
        cols[2].append(normalize_address(a))
        cols[3].append(extract_numbers(a))
        cols[4].append(normalize_text(c))
    return cols


def prepare(df: pd.DataFrame, label: str = "", n_jobs: int = 1, chunk: int = 200_000) -> pd.DataFrame:
    """Return a compact frame with id, country and normalized columns.
    Work is split into chunks and spread over `n_jobs` worker processes."""
    t0 = time.perf_counter()
    names, addrs, countries = df[NAME].tolist(), df[ADDR].tolist(), df[COUNTRY].tolist()
    tasks = [(names[i:i + chunk], addrs[i:i + chunk], countries[i:i + chunk])
             for i in range(0, len(df), chunk)]
    del names, addrs, countries
    if n_jobs > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=min(n_jobs, len(tasks))) as ex:
            results = list(ex.map(_normalize_chunk, tasks))
    else:
        results = [_normalize_chunk(t) for t in tasks]
    out = df[[ID, COUNTRY]].copy()
    for j, col in enumerate(NORM_COLS):
        out[col] = pd.Series(list(chain.from_iterable(r[j] for r in results)),
                             index=out.index, dtype="str")
    log.info("%s: normalized %s records with %d workers in %.1fs", label, fmt_int(len(out)),
             min(n_jobs, len(tasks)), time.perf_counter() - t0)
    _log_prepared(df, out, label)
    return out


def _log_prepared(raw: pd.DataFrame, out: pd.DataFrame, label: str) -> None:
    has_legal = (out["legal"] != "").mean()
    has_nums = (out["addr_nums"] != "").mean()
    log.info("%s normalized: %s have a legal suffix, %s have address numbers",
             label, fmt_pct(has_legal), fmt_pct(has_nums))
    empty_core = int((out["name_core"] == "").sum())
    if empty_core:
        log.warning("%s: %s records have an empty normalized name", label, fmt_int(empty_core))
    if log.isEnabledFor(logging.DEBUG) and len(out):
        smp = out.sample(min(200_000, len(out)), random_state=0)   # stats on a sample
        legal = smp["legal"].str.split().explode().value_counts().head(10)
        log.debug("%s top legal tokens (sample): %s", label, legal.to_dict())
        vocab = smp["name_core"].str.split().explode()
        log.debug("%s name vocab (sample): %s tokens, %s unique", label,
                  fmt_int(len(vocab)), fmt_int(vocab.nunique()))
        log.debug("%s avg tokens (sample): name %.2f, address %.2f", label,
                  smp["name_core"].str.split().str.len().mean(),
                  smp["addr_norm"].str.split().str.len().mean())
        for i in smp.index[:3]:
            log.debug("%s sample | %r -> name=%r legal=%r | %r -> addr=%r nums=%r", label,
                      raw.at[i, NAME], out.at[i, "name_core"], out.at[i, "legal"],
                      raw.at[i, ADDR], out.at[i, "addr_norm"], out.at[i, "addr_nums"])
