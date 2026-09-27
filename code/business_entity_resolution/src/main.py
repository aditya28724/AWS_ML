"""End-to-end CLI (run from the repo root):

    python code/business_entity_resolution/src/main.py train   --data-dir dataset -i 3
    python code/business_entity_resolution/src/main.py predict --data-dir dataset --out-dir output
"""
import argparse
import dataclasses
import gc
import math
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from checkpoint import Checkpointer, file_md5, fingerprint
from config import Config
from data import ID, load_ground_truth
from decision import decide
from evaluate import SetEvaluator, blocking_report
from features import FeatureEngine, describe_features, log_feature_separation
from logger import fmt_int, fmt_pct, get_logger, log_step, metric, setup_logging
from model import MatchModel
from stages import (candidates_stage, chunk_rows, features_stage, ids, index_stage, load_cands,
                    load_features, make_train_sets, normalize_stage, s1_countries, sets_signature,
                    vectorizer_stage)
from submission import write_candidates, write_matches
from train_loop import PairSet, label_pairs, log_label_stats, run_iterations, split_val

log = get_logger("main")


def _log_environment(cfg: Config, mode: str) -> None:
    import lightgbm, pandas, rapidfuzz, sklearn
    log.info("Mode=%s | Python %s | %s | %d CPUs", mode, platform.python_version(),
             platform.platform(), os.cpu_count() or 0)
    log.info("pandas %s | numpy %s | scikit-learn %s | lightgbm %s | rapidfuzz %s",
             pandas.__version__, np.__version__, sklearn.__version__, lightgbm.__version__,
             rapidfuzz.__version__)
    log.info("Command: %s", " ".join(sys.argv))
    log.info("Parallelism: %d worker processes, %d blocking threads, %d LightGBM threads",
             cfg.n_jobs, cfg.block_threads, cfg.lgbm_threads)
    for f in dataclasses.fields(cfg):
        log.debug("config.%-22s = %s", f.name, getattr(cfg, f.name))


def _embedder(cfg: Config):
    if not cfg.use_embeddings:
        log.info("Embeddings disabled (pass --embeddings to enable)")
        return None
    from embeddings import Embedder
    return Embedder(cfg.embed_model, cfg.embed_batch_size)


# ---------------------------------------------------------------------------
# TRAIN
# ---------------------------------------------------------------------------
def train(cfg: Config, ck: Checkpointer) -> None:
    t_start = time.perf_counter()
    embedder = _embedder(cfg)

    with log_step(log, "[train] Normalize"):
        s1, targets, fp_norm = normalize_stage(cfg, ck, "train")
    with log_step(log, "[train] Key index"):
        index, s1_keys, fp_index = index_stage(cfg, ck, "train", s1, targets, fp_norm)
    sets = make_train_sets(len(s1), cfg)
    with log_step(log, "[train] Blocking"):
        cd, fp_c = candidates_stage(cfg, ck, "train", index, s1_keys, sets, fp_index,
                                    sets_signature(cfg))
    n_t = index.n_t
    del index, s1_keys
    gc.collect()
    with log_step(log, "[train] Features"):
        fd, columns = features_stage(cfg, ck, "train", cd, fp_c, sets, s1, targets, embedder)

    with log_step(log, "[train] Labels & blocking quality"):
        gt = load_ground_truth(cfg.data_dir)
        gt_s1 = pd.Index(s1[ID]).get_indexer(gt["s1_id"])
        gt_t = pd.Index(targets[ID]).get_indexer(gt["cand_id"])
        bad_s1, bad_t = int((gt_s1 < 0).sum()), int((gt_t < 0).sum())
        if bad_s1 or bad_t:
            log.warning("Ground truth ids not found in sources: %s S1, %s targets (ignored)",
                        fmt_int(bad_s1), fmt_int(bad_t))
        ok = (gt_s1 >= 0) & (gt_t >= 0)
        gt_s1, gt_t = gt_s1[ok].astype(np.int64), gt_t[ok].astype(np.int64)
        gt_codes = np.unique(gt_s1 * n_t + gt_t)
        n_true = np.bincount(gt_s1, minlength=len(s1))
        country = s1_countries(s1)
        is_s3_t = targets[ID].str.startswith("S3-").to_numpy(dtype=bool)
        del gt

        data = {}
        for name, rows in sets.items():
            c = load_cands(cd, name, len(rows), cfg.chunk_s1)
            if name in ("val", "base"):
                blocking_report(c, gt_s1, gt_t, rows, n_t, country, f"[{name}]")
            X = load_features(fd, name, len(rows), cfg.chunk_s1)
            data[name] = PairSet(c["s1_row"], c["t_row"], X, label_pairs(c["s1_row"], c["t_row"], gt_codes, n_t))
            log_label_stats(name, data[name])
        del s1, targets
        gc.collect()
        describe_features(data["base"].X, columns, "[base]")
        log_feature_separation(data["base"].X, data["base"].y, columns)

    tune, evl, tune_rows, eval_rows = split_val(data.pop("val"), sets["val"])
    log_label_stats("tune", tune)
    log_label_stats("eval", evl)
    ev_tune = SetEvaluator(tune_rows, n_true, len(n_true))
    ev_eval = SetEvaluator(eval_rows, n_true, len(n_true))
    pools = {int(k[4:]): v for k, v in data.items() if k.startswith("pool")}

    fp_feats = fingerprint(fd.name, sets_signature(cfg))
    best = run_iterations(cfg, ck, fp_feats, columns, data["base"], tune, evl, pools,
                          ev_tune, ev_eval, country, is_s3_t)
    metric(log, "TRAIN COMPLETE in %.1f min | best iteration %d | eval F0.5 %.4f",
           (time.perf_counter() - t_start) / 60, int(best["iteration"]), best["eval_f05"])


# ---------------------------------------------------------------------------
# PREDICT
# ---------------------------------------------------------------------------
def predict(cfg: Config, ck: Checkpointer) -> None:
    t_start = time.perf_counter()
    model_path = Path(cfg.artifact_dir) / "model.joblib"
    with log_step(log, "[test] Load model"):
        mm, meta = MatchModel.load(model_path, cfg.lgbm_threads)
        d = meta["decision"]
        log.info("Decision %s | one_to_one=%s | trained metrics: %s", d, meta["one_to_one"],
                 {k: v for k, v in meta.get("metrics", {}).items() if k.startswith("eval_")})
        if meta.get("use_embeddings", False) != cfg.use_embeddings:
            log.warning("Using embeddings=%s to match the trained model", meta.get("use_embeddings"))
            cfg.use_embeddings = bool(meta.get("use_embeddings", False))
        model_fp = file_md5(model_path)
    embedder = _embedder(cfg)

    with log_step(log, "[test] Normalize"):
        s1, targets, fp_norm = normalize_stage(cfg, ck, "test")
    with log_step(log, "[test] Key index"):
        index, s1_keys, fp_index = index_stage(cfg, ck, "test", s1, targets, fp_norm)
    all_rows = np.arange(len(s1), dtype=np.int32)
    with log_step(log, "[test] Blocking"):
        cd, fp_c = candidates_stage(cfg, ck, "test", index, s1_keys, {"all": all_rows}, fp_index, "all")
    del index, s1_keys
    gc.collect()

    with log_step(log, "[test] Features & scoring"):
        pd_dir = ck.dir("probs_test", fingerprint(fp_c, model_fp), {"model": str(model_path.resolve())})
        n_chunks = math.ceil(len(all_rows) / cfg.chunk_s1)
        todo = [i for i in range(n_chunks) if not ck.has(pd_dir / f"all_{i:04d}.npy")]
        log.info("Scoring %d chunks (%d cached)", n_chunks, n_chunks - len(todo))
        if todo:
            vec = vectorizer_stage(cfg, ck, ck.dir("vec_test", fp_norm), s1, targets)
            engine = FeatureEngine(s1, targets, vec, cfg.n_jobs, embedder)
            if engine.columns != mm.features:
                engine.close()
                log.error("Feature columns differ from the trained model; retrain first")
                raise ValueError("feature mismatch")
            cascade = (cfg.cascade_trees, cfg.cascade_cutoff) if cfg.cascade_trees else None
            log.info("Cascade scoring: %s", f"first {cascade[0]} trees, full model for p >= {cascade[1]}"
                     if cascade else "off")
            try:
                t0, step = time.perf_counter(), max(1, len(todo) // 20)
                for k, i in enumerate(todo, 1):
                    c = ck.load_npz(cd / f"all_{i:04d}.npz")
                    # probability chunks are keyed by the model file only: cascade scores differ
                    # from full scores only far below any decision threshold
                    ck.save_npy(pd_dir / f"all_{i:04d}.npy", mm.predict_proba(engine.build(c), cascade=cascade))
                    if k % step == 0 or k == len(todo):
                        el = time.perf_counter() - t0
                        log.info("  scoring progress %d/%d | %.0fs elapsed, ~%.0fs left", k, len(todo),
                                 el, el / k * (len(todo) - k))
            finally:
                engine.close()
        c = load_cands(cd, "all", len(all_rows), cfg.chunk_s1)
        p = np.concatenate([ck.load_npy(pd_dir / f"all_{i:04d}.npy") for i in range(n_chunks)])
        log.info("Scored %s pairs | mean p %.3f | p>=0.5: %s", fmt_int(len(p)), p.mean() if len(p) else 0,
                 fmt_int((p >= 0.5).sum()))

    with log_step(log, "[test] Decide & write"):
        mask = decide(c["s1_row"], c["t_row"], p, d["threshold"], d["margin"], meta["one_to_one"],
                      verbose=True)
        s1_ids, t_ids = ids(s1), ids(targets)
        write_candidates(Path(cfg.out_dir) / "candidate_pairs.tsv", s1_ids, c["s1_row"], c["t_row"], t_ids)
        write_matches(Path(cfg.out_dir) / "matching_results.tsv", s1_ids, c["s1_row"][mask],
                      c["t_row"][mask], t_ids)
        matched = np.zeros(len(s1), bool)
        matched[c["s1_row"][mask]] = True
        country = s1_countries(s1)
        for ctry in sorted(set(country)):
            sel = country == ctry
            log.info("  %-10s %s / %s S1 matched (%s)", ctry, fmt_int(matched[sel].sum()),
                     fmt_int(sel.sum()), fmt_pct(matched[sel].mean()))
    metric(log, "PREDICT COMPLETE in %.1f min | %s / %s S1 entities matched",
           (time.perf_counter() - t_start) / 60, fmt_int(matched.sum()), fmt_int(len(s1)))


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Business entity resolution")
    ap.add_argument("mode", choices=["train", "predict", "all"])
    ap.add_argument("--data-dir", type=Path, default=Config.data_dir)
    ap.add_argument("--out-dir", type=Path, default=Config.out_dir)
    ap.add_argument("--artifact-dir", type=Path, default=Config.artifact_dir)
    ap.add_argument("-i", "--iterations", type=int, default=1,
                    help="train/evaluate/mine cycles (default 1); best iteration is kept")
    ap.add_argument("--sample", type=int, default=Config.train_sample_s1,
                    help="S1 records in the base training set (0 = all remaining)")
    ap.add_argument("--jobs", type=int, default=Config.n_jobs, help="worker processes")
    ap.add_argument("--chunk-size", type=int, default=Config.chunk_s1, help="S1 records per chunk")
    ap.add_argument("--fresh", action="store_true", help="ignore existing checkpoints and recompute")
    ap.add_argument("--embeddings", action="store_true", help="enable multilingual embeddings")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    help="console verbosity (the log file always records DEBUG)")
    ap.add_argument("--log-file", type=Path, default=None,
                    help="log file path (default: ARTIFACT_DIR/logs/<mode>_<timestamp>.log)")
    ap.add_argument("--no-log-file", action="store_true", help="do not write a log file")
    ap.add_argument("--no-color", action="store_true", help="disable colored console output")
    args = ap.parse_args()
    if args.iterations < 1:
        ap.error("--iterations must be >= 1")

    log_file = None
    if not args.no_log_file:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = args.log_file or args.artifact_dir / "logs" / f"{args.mode}_{stamp}.log"
    setup_logging(args.log_level, log_file, color=not args.no_color)

    cfg = Config(data_dir=args.data_dir, out_dir=args.out_dir, artifact_dir=args.artifact_dir,
                 use_embeddings=args.embeddings, iterations=args.iterations,
                 train_sample_s1=args.sample, n_jobs=max(1, args.jobs), chunk_s1=args.chunk_size,
                 fresh=args.fresh)
    _log_environment(cfg, args.mode)
    if log_file:
        log.info("Full DEBUG log: %s", Path(log_file).resolve())
    ck = Checkpointer(Path(cfg.artifact_dir) / "cache", fresh=cfg.fresh)
    try:
        if args.mode in ("train", "all"):
            train(cfg, ck)
        if args.mode in ("predict", "all"):
            predict(cfg, ck)
    except KeyboardInterrupt:
        log.warning("Interrupted; finished stages/chunks are checkpointed, rerun the same command to resume")
        sys.exit(130)
    except Exception as e:
        log.critical("Pipeline aborted; rerun the same command to resume from checkpoints%s",
                     f" (details in {log_file})" if log_file else "",
                     exc_info=not getattr(e, "_ber_logged", False))
        sys.exit(1)


if __name__ == "__main__":
    main()
