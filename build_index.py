"""Precompute listing embeddings so the app starts in seconds instead of minutes.

Usage: python build_index.py   (writes embeddings.npy next to the dataset)
"""

import os
import time

import pandas as pd
from sentence_transformers import SentenceTransformer

from rag import DATA_PATH, EMBED_MODEL_NAME, EMBEDDINGS_PATH, load_embeddings

if __name__ == "__main__":
    if os.path.exists(EMBEDDINGS_PATH):
        os.remove(EMBEDDINGS_PATH)
    started = time.perf_counter()
    df = pd.read_csv(DATA_PATH)
    embeddings = load_embeddings(df, SentenceTransformer(EMBED_MODEL_NAME))
    print(f"Embedded {embeddings.shape[0]:,} listings ({embeddings.shape[1]} dims) in {time.perf_counter() - started:.1f}s -> {EMBEDDINGS_PATH}")
