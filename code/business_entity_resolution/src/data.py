"""Loading source files and ground truth (all tab-separated)."""
import csv
from pathlib import Path

import pandas as pd

from logger import fmt_int, fmt_pct, get_logger

log = get_logger(__name__)

ID, NAME, ADDR, COUNTRY = "entity_id", "business_name", "business_address", "country"
REQUIRED = [ID, NAME, ADDR, COUNTRY]


def read_tsv(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        log.error("File not found: %s", path.resolve())
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                     quoting=csv.QUOTE_NONE, encoding="utf-8")
    log.debug("Read %s: %s rows x %d cols %s", path.name, fmt_int(len(df)), df.shape[1], list(df.columns))
    if df.shape[1] == 1:
        log.warning("%s parsed into a single column; is it really tab-separated?", path.name)
    return df.fillna("")


def describe_source(df: pd.DataFrame, label: str) -> None:
    """Log size, missing fields, duplicates and country mix of a source table."""
    missing_cols = [c for c in REQUIRED if c not in df.columns]
    if missing_cols:
        log.error("%s is missing columns %s", label, missing_cols)
        raise ValueError(f"{label} missing columns {missing_cols}")
    n = len(df)
    log.info("%s: %s records", label, fmt_int(n))
    for col in (NAME, ADDR, COUNTRY):
        empty = int((df[col].str.strip() == "").sum())
        if empty:
            lvl = log.warning if empty / max(n, 1) > 0.05 else log.debug
            lvl("%s: %s empty %s (%s)", label, fmt_int(empty), col, fmt_pct(empty / max(n, 1)))
    dup = int(df[ID].duplicated().sum())
    if dup:
        log.warning("%s: %s duplicate entity_ids", label, fmt_int(dup))
    prefixes = df[ID].str[:3].value_counts().to_dict()
    log.debug("%s: id prefixes %s", label, prefixes)
    countries = df[COUNTRY].value_counts()
    log.info("%s: countries %s", label, countries.to_dict())
    log.debug("%s: avg name len %.1f, avg address len %.1f", label,
              df[NAME].str.len().mean(), df[ADDR].str.len().mean())


def load_split(data_dir: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (source1, targets) where targets = source2 + source3."""
    d = Path(data_dir) / split
    log.info("Loading %s split from %s", split, d.resolve())
    s1 = read_tsv(d / f"{split}_source1.tsv")
    s2 = read_tsv(d / f"{split}_source2.tsv")
    s3 = read_tsv(d / f"{split}_source3.tsv")
    for df, label in ((s1, "S1"), (s2, "S2"), (s3, "S3")):
        describe_source(df, f"{split}/{label}")
    targets = pd.concat([s2, s3], ignore_index=True)
    s1_c, t_c = set(s1[COUNTRY]), set(targets[COUNTRY])
    if s1_c - t_c:
        log.warning("Countries in S1 but not in S2/S3: %s (blocking falls back to all targets)",
                    sorted(s1_c - t_c))
    if t_c - s1_c:
        log.debug("Countries in S2/S3 but not in S1: %s", sorted(t_c - s1_c))
    return s1, targets


def load_ground_truth(data_dir: Path, s1_ids=None, target_ids=None) -> pd.DataFrame:
    """Explode the ground truth into one (s1_id, cand_id) row per true match."""
    gt = read_tsv(Path(data_dir) / "train" / "train_ground_truth.tsv")
    rows = [
        (s1, m.strip())
        for s1, ms in zip(gt["source1_entity_id"], gt["matched_entity_ids"])
        for m in ms.split(",") if m.strip()
    ]
    pairs = pd.DataFrame(rows, columns=["s1_id", "cand_id"])

    n_s1 = len(gt)
    n_single = int((gt["matched_entity_ids"].str.strip() == "").sum())
    sizes = pairs.groupby("s1_id").size()
    log.info("Ground truth: %s S1 rows, %s true pairs, %s singletons (%s)",
             fmt_int(n_s1), fmt_int(len(pairs)), fmt_int(n_single), fmt_pct(n_single / max(n_s1, 1)))
    if len(sizes):
        log.info("Matches per non-singleton S1: mean %.2f, median %d, max %d",
                 sizes.mean(), int(sizes.median()), int(sizes.max()))
        log.debug("Match-count distribution: %s", sizes.value_counts().sort_index().head(10).to_dict())
    src = pairs["cand_id"].str[:2].value_counts().to_dict()
    log.info("True matches by source: %s", src)

    multi = pairs["cand_id"].duplicated(keep=False)
    if multi.any():
        log.warning("%s target ids are matched to more than one S1 (one-to-one assumption violated)",
                    fmt_int(pairs.loc[multi, "cand_id"].nunique()))
    if s1_ids is not None:
        unknown = set(gt["source1_entity_id"]) - set(s1_ids)
        if unknown:
            log.warning("%s ground-truth S1 ids not found in train_source1", fmt_int(len(unknown)))
    if target_ids is not None:
        unknown = set(pairs["cand_id"]) - set(target_ids)
        if unknown:
            log.warning("%s ground-truth target ids not found in S2/S3", fmt_int(len(unknown)))
    return pairs
