"""Scalable candidate generation: inverted indexes over hashed blocking keys.

Every record emits keys of several types, each prefixed with its country:

  postcode    5+ digit number                      "411001"
  token       raw name token                       "nexari"
  squash      skeleton of the whole name (per DBA segment, stop words removed)
  addr        skeleton of the longest address words
  pair        two name-token skeletons             "blk nksr"
  tok_num     house number + name-token skeleton   "543 kk"
  num_street  house number + address-word skeleton "543 krk"
  post_tok    postcode + name-token skeleton

Single tokens are too common at 10M records; the compound types stay rare and
identify a business. Skeletons (normalize.skeleton) make keys survive typos in
vowels and transliterated Indic names. Keys shared by more targets than the
per-type cap are dropped; each shared key adds an IDF-style weight and the top
`max_cands` targets per S1 by summed weight become candidates.

One sorted sub-index per key type keeps peak memory bounded. Key extraction
runs in worker processes; chunk queries run in threads.
"""
import re
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from itertools import combinations

import numpy as np
import pandas as pd

from config import Config
from logger import fmt_int, fmt_pct, get_logger
from normalize import ADDR_ABBREV, skeleton

log = get_logger(__name__)

BLOCK_VERSION = "3.0"
KEY_TYPES = ("postcode", "token", "squash", "addr", "pair", "tok_num", "num_street", "post_tok")
T = {name: j for j, name in enumerate(KEY_TYPES)}
HIT_COLS = [f"hit_{t}" for t in KEY_TYPES]
BLOCK_COLS = ["bscore", "n_keys", *HIT_COLS, "brank"]
WILDCARD = "*"

# Generic address words: never distinctive enough to be keys
_ADDR_STOP = set(ADDR_ABBREV.values()) | {
    "near", "opposite", "behind", "beside", "next", "main", "cross", "city",
    "district", "state", "india", "usa", "france", "united", "states", "null",
}
# Name words ignored in keys (connectors, honorifics, DBA markers)
_KEY_STOP = {"and", "the", "of", "for", "in", "at", "to", "by", "on", "de", "la", "le", "et",
             "smt", "shri", "sri", "shree", "mr", "mrs", "ms", "dr", "mss"}
_DBA = {"formerly", "dba", "aka", "fka"}
MAX_PAIR_TOKENS = 5


def _segments(tokens: list[str]) -> list[list[str]]:
    """Split a name on DBA markers: 'x formerly y z' -> [['x'], ['y', 'z']]."""
    segs, cur = [], []
    for t in tokens:
        if t in _DBA:
            if cur:
                segs.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        segs.append(cur)
    return segs


def record_keys(n: str, a: str, nm: str, max_addr: int) -> set[tuple[int, str]]:
    """All (key_type, value) pairs of one normalized record."""
    rec = set()
    numl = nm.split()
    posts = {x for x in numl if len(x) >= 5}
    house = next((x.lstrip("0") or "0" for x in numl if len(x) <= 5), None)
    raw = n.split()
    toks = [t for t in raw if t not in _KEY_STOP and t not in _DBA]
    for t in toks:
        if len(t) >= 2:
            rec.add((T["token"], t))
    for seg in _segments(raw):
        sq = "".join(skeleton(t) for t in seg if t not in _KEY_STOP)
        if len(sq) >= 3:
            rec.add((T["squash"], sq))
    sk = list(dict.fromkeys(s for s in (skeleton(t) for t in toks) if len(s) >= 2))
    for x, y in combinations(sorted(sk[:MAX_PAIR_TOKENS]), 2):
        rec.add((T["pair"], f"{x} {y}"))
    awords = sorted({skeleton(t) for t in a.split()
                     if len(t) >= 4 and not t.isdigit() and t not in _ADDR_STOP},
                    key=lambda t: (-len(t), t))
    for t in [w for w in awords if len(w) >= 3][:max_addr]:
        rec.add((T["addr"], t))
    for p in posts:
        rec.add((T["postcode"], p))
        for s in sk[:3]:
            rec.add((T["post_tok"], f"{p} {s}"))
    if house is not None:
        for s in sk[:3]:
            rec.add((T["tok_num"], f"{house} {s}"))
        for t in awords[:3]:
            rec.add((T["num_street"], f"{house} {t}"))
    return rec


def _keys_chunk(task):
    """Worker: emit (hash, row, key_type) for a chunk of records."""
    names, addrs, nums, ckeys, row0, also_wild, max_addr = task
    keys, rows, kts = [], [], []
    for i, (n, a, nm, c) in enumerate(zip(names, addrs, nums, ckeys)):
        rec = record_keys(n, a, nm, max_addr)
        for cc in ((c, WILDCARD) if also_wild and c != WILDCARD else (c,)):
            for kt, v in rec:
                keys.append(f"{kt}|{cc}|{v}")
                rows.append(row0 + i)
                kts.append(kt)
    if not keys:
        return np.empty(0, np.uint64), np.empty(0, np.int32), np.empty(0, np.int8)
    h = pd.util.hash_array(np.array(keys, dtype=object), categorize=False)
    return h, np.asarray(rows, np.int32), np.asarray(kts, np.int8)


def country_keys(df: pd.DataFrame, cfg: Config, target_countries: set | None = None) -> pd.Series:
    """Country part of each key. S1 countries absent from targets use a wildcard."""
    if not cfg.block_within_country:
        return pd.Series("", index=df.index, dtype="str")
    c = df["country_norm"]
    if target_countries is None:
        return c
    return c.where(c.isin(list(target_countries)), WILDCARD)


def compute_keys(df: pd.DataFrame, ckeys: pd.Series, cfg: Config, also_wild: bool,
                 label: str, chunk: int = 100_000, max_workers: int = 6):
    """Return (hash uint64, row int32, key_type int8) sorted by row.

    Memory-bounded: each task's strings are sliced from the frame only when it
    is submitted, and at most 2 tasks per worker are in flight at a time.
    """
    t0 = time.perf_counter()
    n_tasks = -(-len(df) // chunk)
    workers = max(1, min(cfg.n_jobs, max_workers, n_tasks))

    def task(k):
        i = k * chunk
        sl = slice(i, i + chunk)
        return (df["name_core"].iloc[sl].tolist(), df["addr_norm"].iloc[sl].tolist(),
                df["addr_nums"].iloc[sl].tolist(), ckeys.iloc[sl].tolist(),
                i, also_wild, cfg.max_addr_keys)

    parts = [None] * n_tasks
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            inflight, nxt = {}, 0
            while nxt < n_tasks or inflight:
                while nxt < n_tasks and len(inflight) < 2 * workers:
                    inflight[ex.submit(_keys_chunk, task(nxt))] = nxt
                    nxt += 1
                done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                for f in done:
                    parts[inflight.pop(f)] = f.result()
    else:
        parts = [_keys_chunk(task(k)) for k in range(n_tasks)]
    h = np.concatenate([p[0] for p in parts]) if parts else np.empty(0, np.uint64)
    rows = np.concatenate([p[1] for p in parts]) if parts else np.empty(0, np.int32)
    kt = np.concatenate([p[2] for p in parts]) if parts else np.empty(0, np.int8)
    del parts
    by_type = np.bincount(kt, minlength=len(KEY_TYPES))
    log.info("%s: %s keys (%.1f per record) with %d workers in %.1fs", label, fmt_int(len(h)),
             len(h) / max(len(df), 1), workers, time.perf_counter() - t0)
    log.debug("%s keys by type: %s", label, dict(zip(KEY_TYPES, map(fmt_int, by_type))))
    return h, rows, kt


class KeyIndex:
    """One sorted sub-index per key type: hash -> (target row, weight)."""

    def __init__(self, parts: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]], n_targets: int):
        self.parts = parts
        self.n_t = int(n_targets)

    @classmethod
    def build(cls, h, rows, kt, n_targets: int, cfg: Config) -> "KeyIndex":
        t0 = time.perf_counter()
        parts = {}
        for j, name in enumerate(KEY_TYPES):
            sel = np.flatnonzero(kt == j)
            if len(sel) == 0:
                continue
            hj, rj = h[sel], rows[sel]
            del sel
            order = np.argsort(hj, kind="stable")
            hj, rj = hj[order], rj[order]
            del order
            starts = np.flatnonzero(np.r_[True, hj[1:] != hj[:-1]])
            sizes = np.diff(np.r_[starts, len(hj)]).astype(np.int32)
            cap = cfg.max_block_size.get(name, cfg.max_block_size["default"])
            keep = np.repeat(sizes <= cap, sizes)
            df = np.repeat(sizes, sizes)[keep]
            w = (cfg.key_weights[name] * np.log1p(n_targets / df)).astype(np.float32)
            parts[j] = (hj[keep], rj[keep], w)
            log.info("  index %-10s %s entries, %s keys | dropped %s keys > %d targets (%s of entries)",
                     name, fmt_int(len(hj)), fmt_int(len(starts)), fmt_int((sizes > cap).sum()), cap,
                     fmt_pct(1 - keep.mean()))
            del hj, rj, keep, df
        idx = cls(parts, n_targets)
        log.info("Index built: %s entries (%.0f MB) in %.1fs", fmt_int(idx.n_entries),
                 idx.nbytes / 2**20, time.perf_counter() - t0)
        return idx

    @property
    def n_entries(self) -> int:
        return sum(len(p[0]) for p in self.parts.values())

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for p in self.parts.values() for a in p)

    def to_arrays(self) -> dict:
        out = {"n_t": np.array([self.n_t])}
        for j, (h, r, w) in self.parts.items():
            out[f"h{j}"], out[f"r{j}"], out[f"w{j}"] = h, r, w
        return out

    @classmethod
    def from_arrays(cls, a: dict) -> "KeyIndex":
        parts = {j: (a[f"h{j}"], a[f"r{j}"], a[f"w{j}"]) for j in range(len(KEY_TYPES)) if f"h{j}" in a}
        return cls(parts, int(a["n_t"][0]))

    def query(self, qh: np.ndarray, qrow: np.ndarray, qkt: np.ndarray, max_cands: int) -> dict:
        """Score all (S1, target) pairs sharing keys; keep top `max_cands` per S1."""
        S, R, W, K = [], [], [], []
        for j, (h, rows, w) in self.parts.items():
            m = qkt == j
            if not m.any() or len(h) == 0:
                continue
            qhj, qrj = qh[m], qrow[m]
            lo = np.searchsorted(h, qhj, "left")
            hi = np.searchsorted(h, qhj, "right")
            cnt = hi - lo
            tot = int(cnt.sum())
            if tot == 0:
                continue
            base = lo - (np.cumsum(cnt) - cnt)
            pos = np.repeat(base, cnt) + np.arange(tot)
            S.append(np.repeat(qrj, cnt))
            R.append(rows[pos])
            W.append(w[pos])
            K.append(np.full(tot, j, np.int8))
        if not S:
            return {c: np.empty(0, np.int32) for c in ["s1_row", "t_row", *BLOCK_COLS]}
        s1 = np.concatenate(S).astype(np.int64)
        code = s1 * self.n_t + np.concatenate(R)
        del s1, S, R
        wt, kt = np.concatenate(W), np.concatenate(K)
        uniq, inv = np.unique(code, return_inverse=True)
        del code
        score = np.bincount(inv, weights=wt).astype(np.float32)
        n_keys = np.bincount(inv).astype(np.int16)
        hits = {f"hit_{t}": (np.bincount(inv, weights=(kt == j), minlength=len(uniq)) > 0).astype(np.int8)
                for j, t in enumerate(KEY_TYPES)}
        s1u = (uniq // self.n_t).astype(np.int32)
        tu = (uniq % self.n_t).astype(np.int32)
        order = np.lexsort((-score, s1u))
        s1s = s1u[order]
        starts = np.flatnonzero(np.r_[True, s1s[1:] != s1s[:-1]])
        rank = np.arange(len(s1s)) - np.repeat(starts, np.diff(np.r_[starts, len(s1s)]))
        top = rank < max_cands
        sel = order[top]
        out = {"s1_row": s1u[sel], "t_row": tu[sel], "bscore": score[sel], "n_keys": n_keys[sel],
               "brank": (rank[top] + 1).astype(np.int16)}
        out.update({k: v[sel] for k, v in hits.items()})
        return out


def select_rows(krow_sorted: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Positions in a row-sorted key array belonging to `rows`."""
    lo = np.searchsorted(krow_sorted, rows, "left")
    hi = np.searchsorted(krow_sorted, rows, "right")
    cnt = hi - lo
    tot = int(cnt.sum())
    base = lo - (np.cumsum(cnt) - cnt)
    return np.repeat(base, cnt) + np.arange(tot)


def log_candidates(c: dict, n_s1: int, label: str) -> None:
    """Summary of a candidate set (dict of arrays)."""
    n = len(c["s1_row"])
    per = np.bincount(np.unique(c["s1_row"], return_inverse=True)[1]) if n else np.array([0])
    zero = n_s1 - len(np.unique(c["s1_row"]))
    log.info("%s: %s pairs | per S1 mean %.1f, max %d | S1 with no candidates: %s (%s)",
             label, fmt_int(n), n / max(n_s1, 1), int(per.max()), fmt_int(zero),
             fmt_pct(zero / max(n_s1, 1)))
    if zero / max(n_s1, 1) > 0.05:
        log.warning("%s: many S1 records have no candidates; consider raising max_block_size", label)
    if n:
        hit_rates = {k.removeprefix("hit_"): fmt_pct(c[k].mean()) for k in HIT_COLS}
        log.debug("%s: pairs sharing each key type: %s", label, hit_rates)
