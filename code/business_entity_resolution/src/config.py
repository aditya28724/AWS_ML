"""Central configuration for the entity resolution pipeline."""
import os
from dataclasses import dataclass, field
from pathlib import Path

_CPUS = os.cpu_count() or 2


@dataclass
class Config:
    data_dir: Path = Path("dataset")
    out_dir: Path = Path("output")
    artifact_dir: Path = Path("artifacts")

    # --- Runtime ------------------------------------------------------------
    n_jobs: int = max(1, _CPUS - 2)        # worker processes (normalize, keys, features)
    block_threads: int = min(3, _CPUS)     # threads for blocking queries (memory-bound)
    iterations: int = 1                    # train -> evaluate -> mine hard negatives -> retrain
    fresh: bool = False                    # ignore existing checkpoints (overwrites them)

    # --- Sampling of Source 1 for training (disjoint subsets) --------------
    seed: int = 42
    val_s1: int = 100_000                  # holdout: first half tunes, second half scores
    train_sample_s1: int = 300_000         # base training set; 0 = all remaining S1
    mine_s1: int = 100_000                 # fresh S1 pool scored per extra iteration

    # --- Blocking (inverted index on keys) ----------------------------------
    block_within_country: bool = True      # keys include the country label
    max_block_size: dict = field(default_factory=lambda: {
        # keys shared by more targets than this are dropped (too common)
        "default": 500, "postcode": 500, "token": 500, "addr": 500,
        "squash": 2000, "pair": 2000, "tok_num": 2000, "num_street": 2000, "post_tok": 2000,
    })
    max_cands: int = 50                    # top candidates kept per S1 (by key score)
    max_addr_keys: int = 2                 # longest distinctive address words used as keys
    chunk_s1: int = 20_000                 # S1 records per blocking/feature chunk
    key_weights: dict = field(default_factory=lambda: {
        "postcode": 1.0, "token": 1.0, "squash": 3.0, "addr": 0.5,
        "pair": 1.5, "tok_num": 2.0, "num_street": 1.5, "post_tok": 2.0,
    })

    # --- Features -------------------------------------------------------------
    tfidf_fit_sample: int = 1_000_000      # texts sampled to fit TF-IDF vocab/IDF
    use_embeddings: bool = False
    embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embed_batch_size: int = 512

    # --- Matcher ------------------------------------------------------------
    lgbm_params: dict = field(default_factory=lambda: {
        "n_estimators": 1500,              # upper bound; early stopping picks the best
        "learning_rate": 0.1,
        "num_leaves": 63,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "max_bin": 255,
        "force_col_wise": True,
        "verbose": -1,
    })
    lgbm_threads: int = _CPUS
    # Cascade scoring at predict time: score with the first N trees, give the full
    # model only to pairs at or above the cutoff (validated on val: no decision changes)
    cascade_trees: int = 100               # 0 disables the cascade
    cascade_cutoff: float = 0.003
    early_stopping_rounds: int = 50

    # --- Hard-negative mining (iterations > 1) ------------------------------
    mine_min_p: float = 0.10               # pool negatives with p >= this are added
    hard_neg_weight: float = 3.0           # weight for false merges (neg with p >= threshold)
    near_miss_weight: float = 2.0          # weight for missed matches (pos with p < threshold)

    # --- Decision layer (tuned per iteration) --------------------------------
    one_to_one: bool = True                # each S2/S3 record joins at most one S1
