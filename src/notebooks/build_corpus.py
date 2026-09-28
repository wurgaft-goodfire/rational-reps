"""Build a corpus's co-occurrence statistics: count the exact 50k-word vocabulary over its shards
(40 processes), encode it once into an id stream (same pool, Rust word tokenizer), and accumulate the
full vocabulary x vocabulary window co-occurrence matrix on the GPU. Writes ``$CACHE_DIR/corpus/{tag}/``
with ``ids.bin`` (uint16), ``doc_lengths.npy``, ``vocab.json``, ``freq.npy`` and
``C_w{window}_{weighting}.npy``. Run: ``uv run python -m src.notebooks.build_corpus wikipedia [weighting]``."""
import json
import os
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

from src.code.corpus import (at_least, cooccurrence_matrix, encode_corpus, hub_shards, joined, parquet_texts, vocabulary,
                             word_tokenizer)

VOCAB, WINDOW, WORKERS = 50000, 16, 40
CORPORA = {  # repo, shard prefix, text column, article filter (min words), lines joined per passage
    "wikipedia": {"repo": "wikimedia/wikipedia", "prefix": "20231101.en/", "column": "text", "min_words": 200,
                  "join": 1, "tag": "wikipedia_20231101_en"},   # English Wikipedia 2023-11, articles of >= 200 words
    "babylm": {"repo": "deven367/babylm-100M", "prefix": "data/train-", "column": "text", "min_words": 0,
               "join": 100, "tag": "babylm_100M"},              # BabyLM 100M: one utterance per row, joined into passages
    "gutenberg": {"repo": "sedthh/gutenberg_english", "prefix": "data/train-", "column": "TEXT", "min_words": 200,
                  "join": 1, "tag": "gutenberg_en"},            # Project Gutenberg English books, one book per row
    "fineweb": {"repo": "HuggingFaceFW/fineweb-edu", "prefix": "sample/10BT/", "column": "text", "min_words": 200,
                "join": 1, "tag": "fineweb_edu_10bt"},          # FineWeb-Edu 10B-token sample, one web page per row
}


def shard_texts(corpus, path):
    """The documents of one parquet shard: passages of ``join`` rows, or single rows of >= ``min_words`` words."""
    texts = parquet_texts([path], column=corpus["column"])
    return joined(texts, corpus["join"]) if corpus["join"] > 1 else at_least(texts, corpus["min_words"])


if __name__ == "__main__":
    corpus, weighting = CORPORA[sys.argv[1]], (sys.argv[2] if len(sys.argv) > 2 else "1/d")
    out = Path(os.environ.get("CACHE_DIR", "outputs")) / "corpus" / corpus["tag"]; out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    if not (out / "ids.bin").exists():                            # the id stream is built once per corpus
        shards, read = hub_shards(corpus["repo"], corpus["prefix"]), partial(shard_texts, corpus)
        words, freq = vocabulary(shards, read, VOCAB, WORKERS)
        print(f"vocabulary of {len(words):,} words from {len(shards)} shards (rarest: {str(words[-1])!r} x {freq[-1]:,}): {time.time() - t:.0f}s", flush=True)
        ids, lengths = encode_corpus(shards, read, word_tokenizer(words), WORKERS)
        assert np.array_equal(np.bincount(ids, minlength=len(words)), freq)   # the Rust tokenizer reproduces the Python counter
        ids.tofile(out / "ids.bin")
        np.save(out / "doc_lengths.npy", lengths)
        (out / "vocab.json").write_text(json.dumps(words.tolist()))
        np.save(out / "freq.npy", freq)
        print(f"encoded {len(ids):,} tokens in {len(lengths):,} documents: {time.time() - t:.0f}s", flush=True)
    ids = np.fromfile(out / "ids.bin", dtype=np.uint16)
    docs = np.repeat(np.arange(len(lengths := np.load(out / "doc_lengths.npy")), dtype=np.int32), lengths)
    V = len(json.loads((out / "vocab.json").read_text()))
    C = cooccurrence_matrix(ids, docs, V, window=WINDOW, weighting=weighting)
    np.save(out / f"C_w{WINDOW}_{weighting.replace('/', '')}.npy", C)
    print(f"matrix {C.shape}, weighting {weighting}: {time.time() - t:.0f}s")
