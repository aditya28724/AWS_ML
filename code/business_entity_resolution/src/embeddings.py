"""Optional multilingual sentence embeddings (uses the GPU when available)."""
import time

import numpy as np

from logger import fmt_int, get_logger

log = get_logger(__name__)


class Embedder:
    def __init__(self, model_name: str, batch_size: int = 512):
        try:
            from sentence_transformers import SentenceTransformer  # lazy import
        except ImportError:
            log.error("--embeddings needs `pip install sentence-transformers`")
            raise
        t0 = time.perf_counter()
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        log.info("Loaded embedding model %s on %s (%.1fs)", model_name, self.model.device,
                 time.perf_counter() - t0)
        if str(self.model.device) == "cpu":
            log.warning("Embeddings on CPU will be slow for millions of records; install CUDA torch")

    def encode(self, texts: list[str]) -> np.ndarray:
        """No global cache: callers pass unique texts per chunk to bound memory."""
        t0 = time.perf_counter()
        vecs = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 show_progress_bar=False, convert_to_numpy=True)
        log.debug("Encoded %s texts in %.1fs", fmt_int(len(texts)), time.perf_counter() - t0)
        return vecs.astype(np.float32, copy=False)
