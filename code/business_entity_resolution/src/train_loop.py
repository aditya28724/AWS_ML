"""Iterative training with hard-negative mining.

Iteration k:
  1. fit LightGBM on base set + everything mined so far (early stopping on val/tune)
  2. tune threshold/margin on val/tune
  3. COST: macro F0.5 (and AUC/AP/logloss) on val/eval, never used for fitting
  4. if k < N: score fresh pool k; add its positives and plausible negatives
     (p >= mine_min_p), up-weighting false merges and near-miss matches
Each finished iteration is checkpointed (model, metrics, mined examples), so a
restart continues with the next iteration. The best iteration by val/eval
F0.5 becomes artifacts/model.joblib.
"""
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from checkpoint import Checkpointer, fingerprint
from config import Config
from decision import decide
from evaluate import SetEvaluator, error_report, tune_decision
from logger import fmt_int, fmt_pct, get_logger, log_step, metric
from model import MatchModel, pair_scores

log = get_logger(__name__)


class PairSet:
    """Flat arrays for the candidate pairs of one S1 set."""

    def __init__(self, s1_row, t_row, X, y):
        self.s1_row, self.t_row, self.X, self.y = s1_row, t_row, X, y.astype(np.int8)

    def subset(self, mask) -> "PairSet":
        return PairSet(self.s1_row[mask], self.t_row[mask], self.X[mask], self.y[mask])

    def __len__(self):
        return len(self.y)


def run_iterations(cfg: Config, ck: Checkpointer, fp_feats: str, columns: list[str],
                   base: PairSet, tune: PairSet, evl: PairSet, pools: dict[int, PairSet],
                   ev_tune: SetEvaluator, ev_eval: SetEvaluator,
                   s1_country: np.ndarray, is_s3_t: np.ndarray) -> dict:
    params = {"lgbm": cfg.lgbm_params, "es": cfg.early_stopping_rounds, "seed": cfg.seed,
              "mine": [cfg.mine_min_p, cfg.hard_neg_weight, cfg.near_miss_weight],
              "one_to_one": cfg.one_to_one}
    it_root = ck.dir("iters", fingerprint(fp_feats, params), params)
    mined: list[tuple[PairSet, np.ndarray]] = []
    history: list[dict] = []

    for k in range(1, cfg.iterations + 1):
        kdir = it_root / f"iter_{k:02d}"
        if ck.has(kdir / "metrics.json"):
            m = ck.load_json(kdir / "metrics.json")
            history.append(m)
            log.info("Iteration %d/%d restored from checkpoint: eval F0.5 %.4f", k, cfg.iterations,
                     m["eval_f05"])
            if k < cfg.iterations and ck.has(kdir / "mined.npz"):
                z = ck.load_npz(kdir / "mined.npz")
                mined.append((pools[k].subset(z["idx"]), z["w"]))
            continue

        with log_step(log, f"Iteration {k}/{cfg.iterations}"):
            t0 = time.perf_counter()
            parts = [base] + [p for p, _ in mined]
            X = np.vstack([p.X for p in parts])
            y = np.concatenate([p.y for p in parts])
            w = np.concatenate([np.ones(len(base), np.float32)] + [wt for _, wt in mined])
            log.info("Training set: base %s pairs + %d mined batches (%s pairs) = %s", fmt_int(len(base)),
                     len(mined), fmt_int(len(y) - len(base)), fmt_int(len(y)))

            mm = MatchModel(cfg.lgbm_params, cfg.seed, cfg.lgbm_threads)
            mm.fit(X, y, w, columns, tune.X, tune.y, cfg.early_stopping_rounds)
            del X, y, w

            p_tune = mm.predict_proba(tune.X)
            best, tune_f05 = tune_decision(ev_tune, tune.s1_row, tune.t_row, p_tune, tune.y, cfg.one_to_one)

            p_eval = mm.predict_proba(evl.X)
            s_eval = pair_scores(evl.y, p_eval)
            mask = decide(evl.s1_row, evl.t_row, p_eval, best["threshold"], best["margin"],
                          cfg.one_to_one, verbose=True)
            eval_f05 = error_report(ev_eval, evl.s1_row, evl.t_row, evl.y, mask, s1_country,
                                    is_s3_t, f"Iteration {k} val/eval")
            metric(log, "Iteration %d COST | eval F0.5 %.4f | tune F0.5 %.4f | eval AUC %.4f AP %.4f "
                   "logloss %.4f", k, eval_f05, tune_f05, s_eval["auc"], s_eval["ap"], s_eval["logloss"])
            mm.log_importance()

            m = {"iteration": k, "eval_f05": eval_f05, "tune_f05": tune_f05, **best,
                 **{f"eval_{a}": v for a, v in s_eval.items()}, "best_round": mm.best_iteration,
                 "train_pairs": int(len(base) + sum(len(p) for p, _ in mined)), "mined_pairs": 0}

            if k < cfg.iterations and k not in pools:
                log.warning("No S1 left for pool%d; iteration %d will retrain without new mined data",
                            k, k + 1)
            elif k < cfg.iterations:
                idx, wts = mine_pool(mm, pools[k], best["threshold"], cfg, k)
                ck.save_npz(kdir / "mined.npz", idx=idx, w=wts)
                mined.append((pools[k].subset(idx), wts))
                m["mined_pairs"] = int(len(idx))

            m["seconds"] = round(time.perf_counter() - t0, 1)
            mm.save(kdir / "model.joblib", extra={"decision": best, "one_to_one": cfg.one_to_one,
                                                   "use_embeddings": cfg.use_embeddings, "metrics": m})
            ck.save_json(kdir / "metrics.json", m)      # written last: marks iteration done
            history.append(m)

    return finish(cfg, it_root, history)


def mine_pool(mm: MatchModel, pool: PairSet, threshold: float, cfg: Config, k: int):
    """Select positives + plausible negatives from an unseen pool, with weights."""
    p = mm.predict_proba(pool.X)
    y = pool.y
    keep = (y == 1) | (p >= cfg.mine_min_p)
    w = np.ones(len(y), np.float32)
    false_merge = (y == 0) & (p >= threshold)
    near_miss = (y == 1) & (p < threshold)
    w[false_merge] = cfg.hard_neg_weight
    w[near_miss] = cfg.near_miss_weight
    idx = np.flatnonzero(keep)
    log.info("Mined pool%d (%s pairs): kept %s | positives %s | false merges %s (w=%.1f) | "
             "near misses %s (w=%.1f) | other plausible negatives %s", k, fmt_int(len(y)),
             fmt_int(len(idx)), fmt_int((y == 1).sum()), fmt_int(false_merge.sum()), cfg.hard_neg_weight,
             fmt_int(near_miss.sum()), cfg.near_miss_weight,
             fmt_int((keep & (y == 0) & ~false_merge).sum()))
    return idx.astype(np.int64), w[idx]


def finish(cfg: Config, it_root: Path, history: list[dict]) -> dict:
    hist = pd.DataFrame(history)
    cols = ["iteration", "eval_f05", "tune_f05", "eval_auc", "eval_ap", "eval_logloss",
            "threshold", "margin", "best_round", "train_pairs", "mined_pairs", "seconds"]
    log.info("Iteration summary:\n%s", hist[[c for c in cols if c in hist]].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))
    best = hist.loc[hist["eval_f05"].idxmax()]
    k = int(best["iteration"])
    if len(hist) > 1:
        delta = hist["eval_f05"].iloc[-1] - hist["eval_f05"].iloc[0]
        log.info("Eval F0.5 change from iteration 1 to %d: %+.4f", len(hist), delta)
    dst = Path(cfg.artifact_dir) / "model.joblib"
    shutil.copyfile(it_root / f"iter_{k:02d}" / "model.joblib", dst)
    hist.to_csv(Path(cfg.artifact_dir) / "iterations.csv", index=False)
    metric(log, "Best iteration %d (eval F0.5 %.4f) copied to %s", k, best["eval_f05"], dst.resolve())
    return best.to_dict()


def split_val(val: PairSet, val_rows: np.ndarray) -> tuple[PairSet, PairSet, np.ndarray, np.ndarray]:
    """First half of the val S1 rows tunes, second half scores (cost)."""
    half = len(val_rows) // 2
    tune_rows, eval_rows = val_rows[:half], val_rows[half:]
    in_tune = np.isin(val.s1_row, tune_rows)
    return val.subset(in_tune), val.subset(~in_tune), tune_rows, eval_rows


def label_pairs(s1_row: np.ndarray, t_row: np.ndarray, gt_codes_sorted: np.ndarray, n_t: int) -> np.ndarray:
    code = s1_row.astype(np.int64) * n_t + t_row
    pos = np.clip(np.searchsorted(gt_codes_sorted, code), 0, max(len(gt_codes_sorted) - 1, 0))
    return (gt_codes_sorted[pos] == code).astype(np.int8) if len(gt_codes_sorted) else np.zeros(len(code), np.int8)


def log_label_stats(name: str, ps: PairSet) -> None:
    log.info("  %-6s %s pairs | positives %s (%s) | S1 %s", name, fmt_int(len(ps)), fmt_int(ps.y.sum()),
             fmt_pct(ps.y.mean() if len(ps) else 0), fmt_int(len(np.unique(ps.s1_row))))
