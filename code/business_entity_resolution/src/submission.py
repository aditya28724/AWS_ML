"""Writers for matching_results.tsv and candidate_pairs.tsv from flat arrays."""
from pathlib import Path

import numpy as np
import pandas as pd

from logger import fmt_int, get_logger

log = get_logger(__name__)


def write_grouped(path: Path, header: tuple[str, str], s1_ids: np.ndarray,
                  s1_row: np.ndarray, t_row: np.ndarray, t_ids: np.ndarray) -> None:
    """One row per S1 id (source order) with its unique target ids comma-joined.
    Grouping works on integer rows; ids are looked up only at the end."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    code = np.unique(s1_row.astype(np.int64) * len(t_ids) + t_row)    # sorted + deduplicated
    r = (code // len(t_ids)).astype(np.int64)
    t = (code % len(t_ids)).astype(np.int64)
    used = np.unique(t)
    prefix_ok = pd.Series(t_ids[used]).str.match(r"^S[23]-").to_numpy(dtype=bool)
    if not prefix_ok.all():
        bad = np.isin(t, used[~prefix_ok])
        log.warning("%s: dropping %s ids without an S2-/S3- prefix", path.name, fmt_int(bad.sum()))
        r, t = r[~bad], t[~bad]
    joined = pd.Series(t_ids[t]).groupby(r).agg(",".join)
    col = np.full(len(s1_ids), "", dtype=object)
    col[joined.index.to_numpy()] = joined.to_numpy()
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\t".join(header) + "\n")
        fh.writelines(f"{i}\t{v}\n" for i, v in zip(s1_ids, col))
    tmp.replace(path)
    log.info("Wrote %s: %s rows, %s ids, %s empty rows", path.resolve(), fmt_int(len(s1_ids)),
             fmt_int(len(t)), fmt_int(len(s1_ids) - len(joined)))


def write_matches(path: Path, s1_ids, s1_row, t_row, t_ids) -> None:
    write_grouped(path, ("source1_entity_id", "matched_entity_ids"), s1_ids, s1_row, t_row, t_ids)


def write_candidates(path: Path, s1_ids, s1_row, t_row, t_ids) -> None:
    write_grouped(path, ("source1_entity_id", "candidate_entity_ids"), s1_ids, s1_row, t_row, t_ids)
