"""Stage checkpoints: atomic save/load keyed by a fingerprint of their inputs.

Each stage writes into  <root>/<stage>-<fingerprint>/ . The fingerprint hashes
everything the stage depends on (input file signatures, upstream fingerprints,
relevant config), so changing a setting only invalidates downstream stages.
Files are written to *.tmp and renamed, so a crash never leaves a partial
file that would be mistaken for a finished one.
"""
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from logger import get_logger

log = get_logger(__name__)


def fingerprint(*parts) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:10]


def file_signature(paths) -> list:
    sig = []
    for p in paths:
        st = Path(p).stat()
        sig.append((Path(p).name, st.st_size, st.st_mtime_ns))
    return sig


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:10]


class Checkpointer:
    def __init__(self, root: Path, enabled: bool = True, fresh: bool = False):
        self.root = Path(root)
        self.enabled = enabled
        self.fresh = fresh
        self._written: set[Path] = set()
        if enabled:
            self.root.mkdir(parents=True, exist_ok=True)
            log.info("Checkpoints: %s%s", self.root.resolve(),
                     " (fresh: existing checkpoints ignored)" if fresh else "")
        else:
            log.info("Checkpoints disabled")

    def dir(self, stage: str, fp: str, meta: dict | None = None) -> Path:
        d = self.root / f"{stage}-{fp}"
        if self.enabled:
            d.mkdir(parents=True, exist_ok=True)
            if meta is not None and not (d / "meta.json").exists():
                self._atomic(d / "meta.json", lambda fh: fh.write(
                    json.dumps(meta, indent=2, default=str).encode()))
        return d

    def has(self, path: Path) -> bool:
        path = Path(path)
        if not self.enabled or not path.exists():
            return False
        return (not self.fresh) or path in self._written

    # --- writers -----------------------------------------------------------
    def _atomic(self, path: Path, write) -> None:
        if not self.enabled:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as fh:
            write(fh)
        os.replace(tmp, path)
        self._written.add(path)

    def save_npz(self, path: Path, **arrays) -> None:
        self._atomic(path, lambda fh: np.savez(fh, **arrays))

    def save_npy(self, path: Path, arr: np.ndarray) -> None:
        self._atomic(path, lambda fh: np.save(fh, arr))

    def save_parquet(self, path: Path, df: pd.DataFrame) -> None:
        self._atomic(path, lambda fh: df.to_parquet(fh, index=False))

    def save_json(self, path: Path, obj) -> None:
        self._atomic(path, lambda fh: fh.write(json.dumps(obj, indent=2, default=str).encode()))

    def save_joblib(self, path: Path, obj) -> None:
        self._atomic(path, lambda fh: joblib.dump(obj, fh))

    # --- readers -----------------------------------------------------------
    @staticmethod
    def load_npz(path: Path) -> dict:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    @staticmethod
    def load_npy(path: Path) -> np.ndarray:
        return np.load(path, allow_pickle=False)

    @staticmethod
    def load_parquet(path: Path) -> pd.DataFrame:
        return pd.read_parquet(path)

    @staticmethod
    def load_json(path: Path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    @staticmethod
    def load_joblib(path: Path):
        return joblib.load(path)
