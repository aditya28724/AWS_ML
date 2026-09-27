"""Vectorized macro F0.5 (per S1, singletons included), blocking and error reports."""
import itertools
import time

import numpy as np
import pandas as pd

from blocking import HIT_COLS
from decision import decide, one_to_one_mask
from logger import fmt_int, fmt_pct, get_logger, metric

log = get_logger(__name__)


class SetEvaluator:
    """Scores predictions for a fixed set of S1 rows.

    n_true_global[r] = number of true matches of S1 row r (including matches
    blocking missed, so recall losses from blocking are counted).
    """

    def __init__(self, s1_rows: np.ndarray, n_true_global: np.ndarray, n_s1_total: int):
        self.rows = np.asarray(s1_rows)
        self.pos = np.full(n_s1_total, -1, np.int64)
        self.pos[self.rows] = np.arange(len(self.rows))
        self.n_true = n_true_global[self.rows].astype(np.float64)

    def per_entity(self, s1_row: np.ndarray, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
        pos = self.pos[s1_row[mask]]
        ym = y[mask].astype(np.float64)
        ok = pos >= 0                       # ignore rows outside this set
        pos, ym = pos[ok], ym[ok]
        n = len(self.rows)
        npred = np.bincount(pos, minlength=n).astype(np.float64)
        tp = np.bincount(pos, weights=ym, minlength=n)
        with np.errstate(divide="ignore", invalid="ignore"):
            prec = np.where(npred > 0, tp / npred, 0.0)
            rec = np.where(self.n_true > 0, tp / self.n_true, 0.0)
            f = np.where(tp > 0, 1.25 * prec * rec / (0.25 * prec + rec), 0.0)
        return np.where(self.n_true == 0, (npred == 0).astype(np.float64), f)

    def macro_f05(self, s1_row, y, mask) -> float:
        return float(self.per_entity(s1_row, y, mask).mean()) if len(self.rows) else 0.0


def tune_decision(ev: SetEvaluator, s1_row, t_row, p, y, one_to_one: bool) -> tuple[dict, float]:
    """Grid-search threshold and margin; the one-to-one mask is computed once."""
    thresholds = np.round(np.arange(0.20, 0.96, 0.025), 3)
    margins = [1.0, 0.5, 0.3, 0.2, 0.1]
    t0 = time.perf_counter()
    o2o = one_to_one_mask(t_row, p) if one_to_one else None
    res = []
    for t, m in itertools.product(thresholds, margins):
        mask = decide(s1_row, t_row, p, float(t), m, one_to_one, o2o=o2o)
        res.append((float(t), float(m), ev.macro_f05(s1_row, y, mask)))
    res = pd.DataFrame(res, columns=["threshold", "margin", "f05"]).sort_values("f05", ascending=False)
    best = {"threshold": float(res.iloc[0]["threshold"]), "margin": float(res.iloc[0]["margin"])}
    log.info("Decision tuning (%d settings, %.1fs). Top 5:\n%s", len(res), time.perf_counter() - t0,
             res.head(5).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    log.debug("F0.5 vs threshold at margin=%.2f:\n%s", best["margin"],
              res[res["margin"] == best["margin"]].sort_values("threshold")
              .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if best["threshold"] in (thresholds[0], thresholds[-1]):
        log.warning("Best threshold %.3f is at the edge of the grid", best["threshold"])
    return best, float(res.iloc[0]["f05"])


def blocking_report(c: dict, gt_s1: np.ndarray, gt_t: np.ndarray, set_rows: np.ndarray,
                    n_t: int, s1_country: np.ndarray, label: str) -> dict:
    """Recall ceiling of the candidates for one S1 set, with breakdowns."""
    in_set = np.isin(gt_s1, set_rows)
    g_s1, g_t = gt_s1[in_set], gt_t[in_set]
    g_code = g_s1.astype(np.int64) * n_t + g_t
    c_code = c["s1_row"].astype(np.int64) * n_t + c["t_row"]
    order = np.argsort(c_code)
    c_sorted = c_code[order]
    pos = np.searchsorted(c_sorted, g_code)
    pos = np.clip(pos, 0, max(len(c_sorted) - 1, 0))
    found = (c_sorted[pos] == g_code) if len(c_sorted) else np.zeros(len(g_code), bool)
    recall = float(found.mean()) if len(g_code) else 1.0
    rep = {"pairs": int(len(c_code)), "pairs_per_s1": len(c_code) / max(len(set_rows), 1),
           "pair_recall": recall, "true_pairs": int(len(g_code))}
    metric(log, "%s blocking recall %s (%s / %s true pairs) | %.1f pairs per S1", label,
           fmt_pct(recall), fmt_int(found.sum()), fmt_int(len(g_code)), rep["pairs_per_s1"])
    if recall < 0.95:
        log.warning("Blocking recall below 95 percent: raise max_cands or max_block_size in config.py")
    if found.any():
        idx = order[pos[found]]
        for col in HIT_COLS:
            log.info("  true pairs sharing a %-8s key: %s", col.removeprefix("hit_"),
                     fmt_pct(c[col][idx].mean()))
        ranks = c["brank"][idx]
        log.debug("  block rank of found true pairs: p50=%d p90=%d p99=%d max=%d",
                  *np.percentile(ranks, [50, 90, 99]).astype(int), ranks.max())
        at_k = {k: fmt_pct((ranks <= k).sum() / max(len(g_code), 1)) for k in (5, 10, 20, 30, 50)
                if k <= ranks.max() or k == 5}
        log.info("  recall@K (top-K candidates per S1): %s", at_k)
        per_s1_found = np.bincount(g_s1[found], minlength=int(g_s1.max()) + 1)
        per_s1_true = np.bincount(g_s1, minlength=int(g_s1.max()) + 1)
        has = per_s1_true > 0
        log.info("  S1 with all matches found %s | with none found %s",
                 fmt_pct((per_s1_found[has] == per_s1_true[has]).mean()),
                 fmt_pct((per_s1_found[has] == 0).mean()))
    ctry = pd.Series(s1_country[g_s1])
    for k, r in pd.Series(found).groupby(ctry.values).agg(["mean", "size"]).iterrows():
        log.info("  recall %-10s %s (n=%s)", k, fmt_pct(r["mean"]), fmt_int(r["size"]))
    return rep


def error_report(ev: SetEvaluator, s1_row, t_row, y, mask, s1_country: np.ndarray,
                 is_s3_t: np.ndarray, label: str) -> float:
    """Break macro F0.5 down by singletons, matched entities, source and country."""
    f = ev.per_entity(s1_row, y, mask)
    score = float(f.mean())
    metric(log, "%s macro F0.5 = %.4f over %s S1 entities", label, score, fmt_int(len(f)))
    single = ev.n_true == 0
    pos = ev.pos[s1_row[mask]]
    npred = np.bincount(pos[pos >= 0], minlength=len(f))
    if single.any():
        log.info("  singletons: %s | correctly empty %s | falsely merged %s", fmt_int(single.sum()),
                 fmt_pct((npred[single] == 0).mean()), fmt_int((npred[single] > 0).sum()))
    if (~single).any():
        tp = y[mask].sum()
        fp = mask.sum() - tp
        fn = ev.n_true[~single].sum() - tp
        log.info("  matched S1: %s | mean F0.5 %.4f | micro P %s R %s", fmt_int((~single).sum()),
                 f[~single].mean(), fmt_pct(tp / max(tp + fp, 1)), fmt_pct(tp / max(tp + fn, 1)))
    fp_mask = mask & (y == 0)
    log.info("  false positives: S2=%s S3=%s", fmt_int((fp_mask & ~is_s3_t[t_row]).sum()),
             fmt_int((fp_mask & is_s3_t[t_row]).sum()))
    ctry = s1_country[ev.rows]
    for k, g in pd.Series(f).groupby(ctry).agg(["mean", "size"]).iterrows():
        log.info("  country %-10s F0.5 %.4f (n=%s)", k, g["mean"], fmt_int(g["size"]))
    return score
