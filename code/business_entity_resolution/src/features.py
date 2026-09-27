"""Pairwise feature engineering, parallelized over worker processes.

String features run in a process pool (each worker holds IDF tables fitted on
the split); block and context features are added in the parent. Output is a
float32 matrix whose column order is `FeatureEngine.columns`.
"""
import logging
import math
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

from blocking import BLOCK_COLS
from data import ID
from logger import fmt_int, fmt_pct, get_logger
from normalize import name_skeleton, skeleton

log = get_logger(__name__)

FEATURE_VERSION = "3.0"
TEXT_COLS = ["name_core", "legal", "addr_norm", "addr_nums"]
PAIR_COLS = [
    "name_jw", "name_lev", "name_tset", "name_tsort", "name_partial", "name_jacc",
    "name_exact", "name_first_tok", "name_acronym", "name_len_diff", "legal_match",
    "skel_tset", "skel_squash_jw", "skel_squash_eq", "skel_contains",
    "name_idf_jacc", "skel_idf_jacc",
    "addr_tset", "addr_partial", "addr_jw", "addr_jacc", "addr_missing", "addr_idf_jacc",
    "num_jacc", "num_shared", "postcode_conflict", "house_eq",
]
CONTEXT_COLS = ["is_s3", "key_sim", "gap_to_best_s1", "rank_in_s1", "bscore_gap"]

# ---------------------------------------------------------------------------
# Worker side
# ---------------------------------------------------------------------------
_IDF: dict = {}


def _init_worker(idf: dict):
    _IDF.clear()
    _IDF.update(idf)


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return np.nan
    return len(sa & sb) / len(sa | sb)


def _wjaccard(a: str, b: str, table: str) -> float:
    """IDF-weighted Jaccard: rare shared tokens count more than common ones."""
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return np.nan
    idf, dflt = _IDF[table], _IDF[table + "_default"]
    inter = sum(idf.get(t, dflt) for t in sa & sb)
    return inter / sum(idf.get(t, dflt) for t in sa | sb)


def _acronym(a: str, b: str) -> float:
    ta, tb = a.split(), b.split()
    if not ta or not tb:
        return 0.0
    ia, ib = "".join(t[0] for t in ta), "".join(t[0] for t in tb)
    return float((len(ta) == 1 and len(ta[0]) > 1 and ta[0] == ib)
                 or (len(tb) == 1 and len(tb[0]) > 1 and tb[0] == ia))


def _house(nums: str):
    return next((x.lstrip("0") or "0" for x in nums.split() if len(x) <= 5), None)


def _num_feats(a: str, b: str) -> tuple[float, float, float, float]:
    ha, hb = _house(a), _house(b)
    house_eq = np.nan if ha is None or hb is None else float(ha == hb)
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return np.nan, 0.0, 0.0, house_eq
    shared = sa & sb
    long_a = {n for n in sa if len(n) >= 5}
    long_b = {n for n in sb if len(n) >= 5}
    return (len(shared) / len(sa | sb), float(len(shared)),
            float(bool(long_a and long_b and not (long_a & long_b))), house_eq)


def _pair_features(task) -> np.ndarray:
    """Worker: string features for one slice of pairs -> float32 (n, len(PAIR_COLS))."""
    ln, rn, ll, rl, la, ra, lnum, rnum = task
    n = len(ln)
    out = np.empty((n, len(PAIR_COLS)), np.float32)
    col = {c: i for i, c in enumerate(PAIR_COLS)}

    def cpd(q, c, scorer, scale=1.0):
        return process.cpdist(q, c, scorer=scorer, workers=1, dtype=np.float32) / scale

    lsk = [name_skeleton(x) for x in ln]
    rsk = [name_skeleton(x) for x in rn]
    lsq = [x.replace(" ", "") for x in lsk]
    rsq = [x.replace(" ", "") for x in rsk]

    out[:, col["name_jw"]] = cpd(ln, rn, JaroWinkler.normalized_similarity)
    out[:, col["name_lev"]] = cpd(ln, rn, Levenshtein.normalized_similarity)
    out[:, col["name_tset"]] = cpd(ln, rn, fuzz.token_set_ratio, 100)
    out[:, col["name_tsort"]] = cpd(ln, rn, fuzz.token_sort_ratio, 100)
    out[:, col["name_partial"]] = cpd(ln, rn, fuzz.partial_ratio, 100)
    out[:, col["skel_tset"]] = cpd(lsk, rsk, fuzz.token_set_ratio, 100)
    out[:, col["skel_squash_jw"]] = cpd(lsq, rsq, JaroWinkler.normalized_similarity)
    out[:, col["addr_tset"]] = cpd(la, ra, fuzz.token_set_ratio, 100)
    out[:, col["addr_partial"]] = cpd(la, ra, fuzz.partial_ratio, 100)
    out[:, col["addr_jw"]] = cpd(la, ra, JaroWinkler.normalized_similarity)

    out[:, col["name_jacc"]] = [_jaccard(a, b) for a, b in zip(ln, rn)]
    out[:, col["name_exact"]] = [float(a == b) for a, b in zip(ln, rn)]
    out[:, col["name_first_tok"]] = [float(a.split()[:1] == b.split()[:1]) for a, b in zip(ln, rn)]
    out[:, col["name_acronym"]] = [_acronym(a, b) for a, b in zip(ln, rn)]
    out[:, col["name_len_diff"]] = [abs(len(a) - len(b)) / max(len(a), len(b), 1) for a, b in zip(ln, rn)]
    out[:, col["legal_match"]] = [np.nan if not (a and b) else float(bool(set(a.split()) & set(b.split())))
                                  for a, b in zip(ll, rl)]
    out[:, col["skel_squash_eq"]] = [float(bool(a) and a == b) for a, b in zip(lsq, rsq)]
    out[:, col["skel_contains"]] = [float(bool(a) and bool(b) and (a in b or b in a)) for a, b in zip(lsq, rsq)]
    out[:, col["name_idf_jacc"]] = [_wjaccard(a, b, "name") for a, b in zip(ln, rn)]
    out[:, col["skel_idf_jacc"]] = [_wjaccard(a, b, "skel") for a, b in zip(lsk, rsk)]
    out[:, col["addr_jacc"]] = [_jaccard(a, b) for a, b in zip(la, ra)]
    out[:, col["addr_missing"]] = [float(not a) + float(not b) for a, b in zip(la, ra)]
    out[:, col["addr_idf_jacc"]] = [_wjaccard(a, b, "addr") for a, b in zip(la, ra)]
    nums = np.array([_num_feats(a, b) for a, b in zip(lnum, rnum)], dtype=np.float32).reshape(-1, 4)
    (out[:, col["num_jacc"]], out[:, col["num_shared"]], out[:, col["postcode_conflict"]],
     out[:, col["house_eq"]]) = nums.T
    return out


# ---------------------------------------------------------------------------
# Parent side
# ---------------------------------------------------------------------------
def fit_vectorizers(s1: pd.DataFrame, targets: pd.DataFrame, sample: int, seed: int) -> tuple[dict]:
    """IDF tables (name tokens, name skeletons, address tokens) from a sample of the split.
    Returned as a 1-tuple so it can be passed as worker initargs."""
    t0 = time.perf_counter()
    both = pd.concat([s1[["name_core", "addr_norm"]], targets[["name_core", "addr_norm"]]],
                     ignore_index=True)
    if len(both) > sample:
        both = both.sample(sample, random_state=seed)
    n = len(both)
    counts = {"name": Counter(), "skel": Counter(), "addr": Counter()}
    for nm, ad in zip(both["name_core"].tolist(), both["addr_norm"].tolist()):
        toks = set(nm.split())
        counts["name"].update(toks)
        counts["skel"].update({skeleton(t) for t in toks})
        counts["addr"].update(set(ad.split()))
    idf = {}
    for k, cnt in counts.items():
        idf[k] = {t: float(np.log((n + 1) / (c + 1)) + 1) for t, c in cnt.items() if c >= 2}
        idf[k + "_default"] = float(np.log(n + 1) + 1)   # unseen / singleton tokens: rarest
    log.info("IDF tables fitted on %s records: %s name, %s skeleton, %s address tokens (%.1fs)",
             fmt_int(n), fmt_int(len(idf["name"])), fmt_int(len(idf["skel"])), fmt_int(len(idf["addr"])),
             time.perf_counter() - t0)
    return (idf,)


class FeatureEngine:
    def __init__(self, s1: pd.DataFrame, targets: pd.DataFrame, vectorizers, n_jobs: int,
                 embedder=None):
        self.L = {c: s1[c] for c in TEXT_COLS}
        self.R = {c: targets[c] for c in TEXT_COLS}
        self.is_s3 = targets[ID].str.startswith("S3-").to_numpy(dtype=bool)
        self.embedder = embedder
        self.n_jobs = n_jobs
        self.pool = None
        if n_jobs > 1:
            self.pool = ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_worker,
                                            initargs=vectorizers)
            log.info("Feature pool started with %d worker processes", n_jobs)
        else:
            _init_worker(*vectorizers)

    @property
    def columns(self) -> list[str]:
        return PAIR_COLS + BLOCK_COLS + CONTEXT_COLS + (["embed_cos"] if self.embedder else [])

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown()
            self.pool = None

    def _texts(self, li: np.ndarray, ri: np.ndarray):
        take = lambda s, idx: s.take(idx).tolist()
        return (take(self.L["name_core"], li), take(self.R["name_core"], ri),
                take(self.L["legal"], li), take(self.R["legal"], ri),
                take(self.L["addr_norm"], li), take(self.R["addr_norm"], ri),
                take(self.L["addr_nums"], li), take(self.R["addr_nums"], ri))

    def build(self, c: dict) -> np.ndarray:
        """Feature matrix for one candidate chunk (all pairs of its S1 records)."""
        li, ri = c["s1_row"], c["t_row"]
        n = len(li)
        if n == 0:
            return np.empty((0, len(self.columns)), np.float32)
        t0 = time.perf_counter()
        size = max(5_000, math.ceil(n / (2 * max(self.n_jobs, 1))))
        tasks = [self._texts(li[i:i + size], ri[i:i + size]) for i in range(0, n, size)]
        if self.pool is not None:
            parts = list(self.pool.map(_pair_features, tasks))
        else:
            parts = [_pair_features(t) for t in tasks]
        P = np.vstack(parts)
        t_str = time.perf_counter() - t0

        block = np.column_stack([c[k].astype(np.float32) for k in BLOCK_COLS])
        ctx = self._context(li, P, c["bscore"], ri)
        cols = [P, block, ctx]
        if self.embedder is not None:
            cols.append(self._embed_cos(li, ri)[:, None])
        X = np.hstack(cols).astype(np.float32, copy=False)
        log.debug("  features for %s pairs: strings %.1fs, total %.1fs", fmt_int(n), t_str,
                  time.perf_counter() - t0)
        return X

    def _context(self, li, P, bscore, ri) -> np.ndarray:
        """How each pair compares with the other candidates of the same S1."""
        pc = {c: i for i, c in enumerate(PAIR_COLS)}
        key = (P[:, pc["name_tset"]] + P[:, pc["addr_tset"]]
               + np.nan_to_num(P[:, pc["name_idf_jacc"]]) + np.nan_to_num(P[:, pc["skel_idf_jacc"]]))
        _, g = np.unique(li, return_inverse=True)
        best = np.full(g.max() + 1, -np.inf, np.float32)
        np.maximum.at(best, g, key)
        bbest = np.full(g.max() + 1, -np.inf, np.float32)
        np.maximum.at(bbest, g, bscore)
        order = np.lexsort((-key, g))
        gs = g[order]
        starts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
        rank = np.empty(len(g), np.float32)
        rank[order] = np.arange(len(g)) - np.repeat(starts, np.diff(np.r_[starts, len(g)])) + 1
        return np.column_stack([self.is_s3[ri].astype(np.float32), key, key - best[g], rank,
                                bscore - bbest[g]]).astype(np.float32)

    def _embed_cos(self, li, ri) -> np.ndarray:
        ul, il = np.unique(li, return_inverse=True)
        ur, ir = np.unique(ri, return_inverse=True)
        full = lambda side, idx: (side["name_core"].take(idx) + " " + side["addr_norm"].take(idx)).tolist()
        A = self.embedder.encode(full(self.L, ul))
        B = self.embedder.encode(full(self.R, ur))
        return (A[il] * B[ir]).sum(axis=1)


def describe_features(X: np.ndarray, columns: list[str], label: str, sample: int = 500_000) -> None:
    """Log shape, missing values and degenerate columns (on a row sample)."""
    log.info("%s feature matrix: %s x %d (%.0f MB)", label, fmt_int(len(X)), X.shape[1], X.nbytes / 2**20)
    if len(X) == 0:
        return
    rng = np.random.default_rng(0)
    f = pd.DataFrame(X[rng.choice(len(X), min(sample, len(X)), replace=False)], columns=columns)
    nan_rate = f.isna().mean()
    for col, r in nan_rate[nan_rate > 0].sort_values(ascending=False).items():
        (log.warning if r > 0.5 else log.debug)("  feature %-20s NaN %s", col, fmt_pct(r))
    const = [c for c in f.columns if f[c].nunique(dropna=True) <= 1]
    if const:
        log.warning("Constant features (no signal): %s", const)
    if log.isEnabledFor(logging.DEBUG):
        desc = f.describe().T[["mean", "std", "min", "max"]]
        log.debug("Feature summary (sample):\n%s", desc.to_string(float_format=lambda x: f"{x:.3f}"))


def log_feature_separation(X: np.ndarray, y: np.ndarray, columns: list[str], top: int = 15,
                           sample: int = 1_000_000) -> None:
    """Mean of each feature on matches vs non-matches (on a row sample)."""
    if len(X) > sample:
        idx = np.random.default_rng(0).choice(len(X), sample, replace=False)
        X, y = X[idx], y[idx]
    f = pd.DataFrame(X, columns=columns)
    pos, neg = f[y == 1].mean(), f[y == 0].mean()
    std = f.std().replace(0, np.nan)
    sep = ((pos - neg) / std).abs().sort_values(ascending=False)
    lines = [f"  {c:<22} match={pos[c]:8.3f}  non-match={neg[c]:8.3f}  |d|={sep[c]:.2f}"
             for c in sep.index[:top]]
    log.info("Most separating features (standardized mean difference):\n%s", "\n".join(lines))
    weak = sep[sep < 0.05].index.tolist()
    if weak:
        log.debug("Weak features (|d| < 0.05): %s", weak)
