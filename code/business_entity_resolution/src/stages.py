"""Checkpointed data-preparation stages shared by train and predict.

    normalize -> key index -> candidates (per S1 chunk) -> features (per chunk)

Every stage output lives under artifacts/cache/<stage>_<split>-<fingerprint>/.
Chunked stages save each chunk as soon as it is done, so an interrupted run
resumes from the last finished chunk.
"""
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from blocking import (BLOCK_VERSION, WILDCARD, KeyIndex, compute_keys, country_keys,
                      log_candidates, select_rows)
from checkpoint import Checkpointer, file_signature, fingerprint
from config import Config
from data import ID, load_split
from features import FEATURE_VERSION, FeatureEngine, fit_vectorizers
from logger import fmt_int, get_logger
from normalize import ADDR_ABBREV, LEGAL_TOKENS, NAME_ABBREV, NORM_VERSION, prepare

log = get_logger(__name__)


def normalize_stage(cfg: Config, ck: Checkpointer, split: str):
    """Return (s1, targets, fp) with normalized columns (cached as Parquet)."""
    d = Path(cfg.data_dir) / split
    files = [d / f"{split}_source{i}.tsv" for i in (1, 2, 3)]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        log.error("Missing input files: %s", missing)
        raise FileNotFoundError(missing[0])
    fp = fingerprint("norm", NORM_VERSION, file_signature(files),
                     sorted(LEGAL_TOKENS), NAME_ABBREV, ADDR_ABBREV)
    sd = ck.dir(f"norm_{split}", fp, {"files": [str(f.resolve()) for f in files]})
    p1, pt = sd / "s1.parquet", sd / "targets.parquet"
    if ck.has(p1) and ck.has(pt):
        t0 = time.perf_counter()
        s1, targets = ck.load_parquet(p1), ck.load_parquet(pt)
        log.info("[%s] loaded normalized data from checkpoint: S1=%s targets=%s (%.1fs)", split,
                 fmt_int(len(s1)), fmt_int(len(targets)), time.perf_counter() - t0)
        return s1, targets, fp
    raw_s1, raw_t = load_split(cfg.data_dir, split)
    s1 = prepare(raw_s1, f"{split}/S1", cfg.n_jobs)
    del raw_s1
    targets = prepare(raw_t, f"{split}/S2+S3", cfg.n_jobs)
    del raw_t
    ck.save_parquet(p1, s1)
    ck.save_parquet(pt, targets)
    log.info("[%s] normalized data checkpointed to %s", split, sd)
    return s1, targets, fp


def index_stage(cfg: Config, ck: Checkpointer, split: str, s1, targets, fp_norm: str):
    """Return (KeyIndex over targets, S1 keys sorted by row, fp)."""
    params = {"v": BLOCK_VERSION, "within_country": cfg.block_within_country,
              "max_block_size": cfg.max_block_size, "max_addr_keys": cfg.max_addr_keys,
              "key_weights": cfg.key_weights}
    fp = fingerprint(fp_norm, params)
    sd = ck.dir(f"index_{split}", fp, params)
    pi, ps = sd / "index.npz", sd / "s1_keys.npz"
    if ck.has(pi) and ck.has(ps):
        t0 = time.perf_counter()
        index = KeyIndex.from_arrays(ck.load_npz(pi))
        s1_keys = ck.load_npz(ps)
        log.info("[%s] loaded key index from checkpoint: %s entries, %s S1 keys (%.1fs)", split,
                 fmt_int(index.n_entries), fmt_int(len(s1_keys["h"])), time.perf_counter() - t0)
        return index, s1_keys, fp

    t_countries = set(targets["country_norm"].unique())
    ck_t = country_keys(targets, cfg)
    ck_s = country_keys(s1, cfg, t_countries)
    also_wild = bool((ck_s == WILDCARD).any())
    if also_wild:
        unseen = sorted(set(s1["country_norm"].unique()) - t_countries)
        log.warning("S1 countries with no S2/S3 records %s are matched against all countries", unseen)
    h, rows, kt = compute_keys(targets, ck_t, cfg, also_wild, f"{split}/targets")
    index = KeyIndex.build(h, rows, kt, len(targets), cfg)
    del h, rows, kt
    ck.save_npz(pi, **index.to_arrays())
    sh, srow, skt = compute_keys(s1, ck_s, cfg, False, f"{split}/S1")
    s1_keys = {"h": sh, "row": srow, "kt": skt}   # already sorted by row
    ck.save_npz(ps, **s1_keys)
    return index, s1_keys, fp


def chunk_rows(rows: np.ndarray, size: int) -> list[np.ndarray]:
    return [rows[i:i + size] for i in range(0, len(rows), size)]


def candidates_stage(cfg: Config, ck: Checkpointer, split: str, index: KeyIndex, s1_keys: dict,
                     sets: dict[str, np.ndarray], fp_index: str, sets_sig) -> tuple[Path, str]:
    """Block every chunk of every S1 set (threaded); returns (dir, fp)."""
    params = {"max_cands": cfg.max_cands, "chunk_s1": cfg.chunk_s1, "sets": sets_sig}
    fp = fingerprint(fp_index, params)
    cd = ck.dir(f"cands_{split}", fp, params)
    jobs = [(name, i, rows) for name, all_rows in sets.items()
            for i, rows in enumerate(chunk_rows(all_rows, cfg.chunk_s1))]
    todo = [j for j in jobs if not ck.has(cd / f"{j[0]}_{j[1]:04d}.npz")]
    log.info("[%s] blocking: %d chunks total, %d cached, %d to compute (%d threads)", split,
             len(jobs), len(jobs) - len(todo), len(todo), cfg.block_threads)
    if not todo:
        return cd, fp

    def run(job):
        name, i, rows = job
        t0 = time.perf_counter()
        pos = select_rows(s1_keys["row"], np.sort(rows))
        c = index.query(s1_keys["h"][pos], s1_keys["row"][pos], s1_keys["kt"][pos], cfg.max_cands)
        ck.save_npz(cd / f"{name}_{i:04d}.npz", **c)
        return name, i, len(c["s1_row"]), time.perf_counter() - t0

    t0, done, step = time.perf_counter(), 0, max(1, len(todo) // 20)
    with ThreadPoolExecutor(max_workers=cfg.block_threads) as ex:
        for fut in as_completed([ex.submit(run, j) for j in todo]):
            name, i, n, dt = fut.result()
            done += 1
            log.debug("  blocked %s chunk %d: %s pairs in %.1fs", name, i, fmt_int(n), dt)
            if done % step == 0 or done == len(todo):
                el = time.perf_counter() - t0
                log.info("  blocking progress %d/%d chunks | %.0fs elapsed, ~%.0fs left", done,
                         len(todo), el, el / done * (len(todo) - done))
    return cd, fp


def load_cands(cd: Path, name: str, n_rows: int, chunk: int) -> dict:
    n_chunks = math.ceil(n_rows / chunk)
    parts = [Checkpointer.load_npz(cd / f"{name}_{i:04d}.npz") for i in range(n_chunks)]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]} if parts else {}


def vectorizer_stage(cfg: Config, ck: Checkpointer, sd: Path, s1, targets):
    pv = sd / "vectorizers.joblib"
    if ck.has(pv):
        return ck.load_joblib(pv)
    vec = fit_vectorizers(s1, targets, cfg.tfidf_fit_sample, cfg.seed)
    ck.save_joblib(pv, vec)
    return vec


def features_stage(cfg: Config, ck: Checkpointer, split: str, cd: Path, fp_c: str,
                   sets: dict[str, np.ndarray], s1, targets, embedder) -> tuple[Path, list[str]]:
    """Compute and cache the feature matrix of every candidate chunk."""
    params = {"v": FEATURE_VERSION, "tfidf_fit_sample": cfg.tfidf_fit_sample, "seed": cfg.seed,
              "embeddings": cfg.embed_model if cfg.use_embeddings else None}
    fp = fingerprint(fp_c, params)
    fd = ck.dir(f"feats_{split}", fp, params)
    jobs = [(name, i) for name, rows in sets.items()
            for i in range(math.ceil(len(rows) / cfg.chunk_s1))]
    todo = [j for j in jobs if not ck.has(fd / f"{j[0]}_{j[1]:04d}.npy")]
    log.info("[%s] features: %d chunks total, %d cached, %d to compute", split, len(jobs),
             len(jobs) - len(todo), len(todo))
    vec = vectorizer_stage(cfg, ck, fd, s1, targets)
    engine = FeatureEngine(s1, targets, vec, cfg.n_jobs if todo else 1, embedder)
    columns = engine.columns
    try:
        t0 = time.perf_counter()
        step = max(1, len(todo) // 20)
        for k, (name, i) in enumerate(todo, 1):
            c = Checkpointer.load_npz(cd / f"{name}_{i:04d}.npz")
            ck.save_npy(fd / f"{name}_{i:04d}.npy", engine.build(c))
            if k % step == 0 or k == len(todo):
                el = time.perf_counter() - t0
                log.info("  feature progress %d/%d chunks | %.0fs elapsed, ~%.0fs left", k,
                         len(todo), el, el / k * (len(todo) - k))
    finally:
        engine.close()
    ck.save_json(fd / "columns.json", columns)
    return fd, columns


def load_features(fd: Path, name: str, n_rows: int, chunk: int) -> np.ndarray:
    n_chunks = math.ceil(n_rows / chunk)
    parts = [Checkpointer.load_npy(fd / f"{name}_{i:04d}.npy") for i in range(n_chunks)]
    return np.vstack(parts) if parts else np.empty((0, 0), np.float32)


def make_train_sets(n_s1: int, cfg: Config) -> dict[str, np.ndarray]:
    """Disjoint S1 subsets from one seeded permutation: val, base, pool1..pool{N-1}."""
    perm = np.random.default_rng(cfg.seed).permutation(n_s1).astype(np.int32)
    sets, off = {}, 0

    def take(name, k):
        nonlocal off
        if off >= n_s1:
            log.warning("Not enough S1 records for set %r; skipped", name)
            return
        if off + k > n_s1:
            log.warning("Set %r truncated to %s S1 records", name, fmt_int(n_s1 - off))
        sets[name] = perm[off:off + k]
        off += len(sets[name])

    take("val", cfg.val_s1)
    base_n = cfg.train_sample_s1 or (n_s1 - cfg.val_s1 - cfg.mine_s1 * (cfg.iterations - 1))
    take("base", max(base_n, 1))
    for k in range(1, cfg.iterations):
        take(f"pool{k}", cfg.mine_s1)
    log.info("S1 sets: %s", {k: fmt_int(len(v)) for k, v in sets.items()})
    return sets


def sets_signature(cfg: Config) -> dict:
    return {"seed": cfg.seed, "val": cfg.val_s1, "base": cfg.train_sample_s1, "mine": cfg.mine_s1,
            "iterations_for_base": cfg.iterations if not cfg.train_sample_s1 else None}


def s1_countries(s1: pd.DataFrame) -> np.ndarray:
    return s1["country"].to_numpy(dtype=object)


def ids(df: pd.DataFrame) -> np.ndarray:
    return df[ID].to_numpy(dtype=object)
