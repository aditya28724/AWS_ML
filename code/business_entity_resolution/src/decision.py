"""Precision-first decision layer on flat arrays of (s1_row, t_row, p)."""
import numpy as np

from logger import fmt_int, get_logger

log = get_logger(__name__)


def one_to_one_mask(t_row: np.ndarray, p: np.ndarray) -> np.ndarray:
    """True for the highest-probability S1 of each target (S1 is deduplicated)."""
    mask = np.zeros(len(p), bool)
    if len(p) == 0:
        return mask
    order = np.lexsort((-p, t_row))
    ts = t_row[order]
    first = np.r_[True, ts[1:] != ts[:-1]]
    mask[order[first]] = True
    return mask


def decide(s1_row: np.ndarray, t_row: np.ndarray, p: np.ndarray, threshold: float, margin: float,
           one_to_one: bool, o2o: np.ndarray | None = None, verbose: bool = False) -> np.ndarray:
    """Boolean mask of pairs kept as matches.

    1. one_to_one: each target keeps only its best S1 (pass a precomputed `o2o`
       mask to skip recomputing it during grid search).
    2. p >= threshold.
    3. p >= best p of that S1 - margin.
    """
    n0 = len(p)
    m = (o2o if o2o is not None else one_to_one_mask(t_row, p)) if one_to_one else np.ones(n0, bool)
    n1 = int(m.sum())
    m = m & (p >= threshold)
    n2 = int(m.sum())
    if n2:
        best = np.full(int(s1_row.max()) + 1, -np.inf, np.float32)
        np.maximum.at(best, s1_row[m], p[m])
        m &= p >= best[s1_row] - margin
    n3 = int(m.sum())
    if verbose:
        log.info("Decision (threshold=%.3f, margin=%.2f, one_to_one=%s):", threshold, margin, one_to_one)
        log.info("  %s candidate pairs", fmt_int(n0))
        log.info("  -%s by one-to-one  -> %s", fmt_int(n0 - n1), fmt_int(n1))
        log.info("  -%s by threshold   -> %s", fmt_int(n1 - n2), fmt_int(n2))
        log.info("  -%s by margin      -> %s matched pairs across %s S1 entities",
                 fmt_int(n2 - n3), fmt_int(n3), fmt_int(len(np.unique(s1_row[m]))))
    return m
