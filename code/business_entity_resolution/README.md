# Business Entity Resolution v1.1

For every Source 1 record, the pipeline finds the Source 2 and Source 3 records that refer to the same business. It is built to handle millions of records on a 16 GB laptop.

**Pipeline:** parallel normalization → inverted-index key blocking → multiprocess pair features → LightGBM (early stopping) → precision-first decision layer (one-to-one assignment, tuned threshold and margin).

Training can run several **iterations with hard-negative mining**. Every stage is **checkpointed**, so an interrupted run resumes where it stopped.

## Layout

```
src/
  main.py        CLI: train / predict / all
  config.py      all hyperparameters
  logger.py      color-coded console + DEBUG log file
  checkpoint.py  fingerprinted, atomic stage checkpoints
  stages.py      normalize -> key index -> candidates -> features (checkpointed, chunked)
  data.py        TSV loading, ground truth
  normalize.py   name/address normalization (multiprocess)
  blocking.py    hashed-key inverted index, top-K candidates per S1 (threaded)
  features.py    pair features (process pool, rapidfuzz cpdist, TF-IDF cosine)
  embeddings.py  optional sentence-transformers wrapper
  model.py       LightGBM with early stopping and sample weights
  train_loop.py  iterations: train -> cost -> mine hard negatives -> retrain
  decision.py    probabilities -> matches (vectorized)
  evaluate.py    vectorized macro F0.5, blocking recall, error breakdown
  submission.py  writes matching_results.tsv / candidate_pairs.tsv
```

## Setup (Windows cmd, from the repo root)

```
python -m venv .venv
.venv\Scripts\activate
pip install -r code\business_entity_resolution\requirements.txt
```

## Run (from the repo root)

```
python code\business_entity_resolution\src\main.py train --data-dir dataset -i 3
python code\business_entity_resolution\src\main.py predict --data-dir dataset --out-dir output
```

`all` runs `train` and then `predict`. `predict` always uses the best iteration's model at `artifacts\model.joblib`.

| Flag | Default | Meaning |
|---|---|---|
| `-i`, `--iterations N` | 1 | number of train → cost → mine → retrain cycles; the best one is kept |
| `--sample N` | 300000 | S1 records in the base training set (0 = all remaining) |
| `--jobs N` | CPUs − 2 | worker processes for normalization, keys and features |
| `--chunk-size N` | 20000 | S1 records per blocking/feature chunk (lower it if memory is tight) |
| `--fresh` | off | ignore existing checkpoints and recompute everything |
| `--embeddings` | off | multilingual embedding feature (needs `sentence-transformers`, ideally a GPU) |
| `--data-dir` / `--out-dir` / `--artifact-dir` | `dataset` / `output` / `artifacts` | paths |
| `--log-level`, `--log-file`, `--no-log-file`, `--no-color` | | logging |

## How training iterations work

Disjoint S1 subsets are drawn from one seeded permutation:

| Set | Size | Role |
|---|---|---|
| `val` | 100k | first half tunes early stopping and the threshold; second half is the **cost** set, never used for fitting |
| `base` | `--sample` | base training set |
| `pool1`..`poolN-1` | 100k each | fresh, unseen S1 records mined in iterations 1..N-1 |

In iteration *k*:
1. Fit on the base set plus everything mined so far.
2. Tune the threshold and margin.
3. Log the **cost**: macro F0.5, AUC, AP and log loss on val/eval.
4. Score pool *k* and add its positives and plausible negatives (p ≥ 0.1). False merges get 3× weight and near-miss matches get 2×.

At the end, `artifacts\iterations.csv` summarizes every iteration, and the best one by eval F0.5 is copied to `artifacts\model.joblib`.

## Checkpoints and resume

Stage outputs live in `artifacts\cache\<stage>-<fingerprint>\`:
- normalized Parquet files
- the key index
- candidate and feature chunks
- per-iteration models and metrics
- test probability chunks

The fingerprint covers the input files and the settings each stage depends on. Changing a setting only recomputes the stages after it. After a crash or Ctrl+C, **rerun the same command** to continue from the last finished chunk or iteration. Delete `artifacts\cache` to reclaim disk space.

## Logging

Console output is color-coded:

| Level | Color | Used for |
|---|---|---|
| DEBUG | cyan | detail |
| INFO | green | progress |
| STEP | magenta | step banners with timings |
| METRIC | blue | cost and headline numbers |
| WARNING | yellow | warnings |
| ERROR / CRITICAL | red | errors |

Every run writes a full DEBUG log to `ARTIFACT_DIR\logs\<mode>_<timestamp>.log`.

## Validate the output

```
python utils\validate_submission.py --matching output\matching_results.tsv --candidate output\candidate_pairs.tsv --test-dir dataset\test
```

Run this from the folder that contains the organizers' `utils\`.
