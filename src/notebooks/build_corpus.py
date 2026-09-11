"""Build a corpus's co-occurrence statistics: train a 25k-word tokenizer on it, encode it once into an
id stream, and accumulate the full vocabulary x vocabulary window co-occurrence matrix on the GPU.
Writes ``$CACHE_DIR/corpus/{tag}/`` with ``ids.bin`` (uint16), ``doc_lengths.npy``, ``vocab.json``,
``freq.npy`` and ``C_w{window}_{weighting}.npy``. Run: ``uv run python -m src.notebooks.build_corpus wikipedia [weighting]``."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from src.code.corpus import at_least, cooccurrence_matrix, encode_corpus, hub_texts, joined, train_tokenizer

VOCAB, WINDOW = 25000, 16
CORPORA = {  # repo, shard prefix, text column, article filter (min words), lines joined per passage
    "wikipedia": {"repo": "wikimedia/wikipedia", "prefix": "20231101.en/", "column": "text", "min_words": 200,
                  "join": 1, "tag": "wikipedia_20231101_en"},   # English Wikipedia 2023-11, articles of >= 200 words
    "babylm": {"repo": "deven367/babylm-100M", "prefix": "data/train-", "column": "text", "min_words": 0,
               "join": 100, "tag": "babylm_100M"},              # BabyLM 100M: one utterance per row, joined into passages
    "gutenberg": {"repo": "sedthh/gutenberg_english", "prefix": "data/train-", "column": "TEXT", "min_words": 200,
                  "join": 1, "tag": "gutenberg_en"},            # Project Gutenberg English books, one book per row
}


def texts_of(corpus):
    texts = hub_texts(corpus["repo"], corpus["prefix"], column=corpus["column"])
    return joined(texts, corpus["join"]) if corpus["join"] > 1 else at_least(texts, corpus["min_words"])


if __name__ == "__main__":
    corpus, weighting = CORPORA[sys.argv[1]], (sys.argv[2] if len(sys.argv) > 2 else "1/d")
    out = Path(os.environ.get("CACHE_DIR", "outputs")) / "corpus" / corpus["tag"]; out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    if not (out / "ids.bin").exists():                            # the id stream is built once per corpus
        tok = train_tokenizer(texts_of(corpus), VOCAB)
        print(f"tokenizer trained: {time.time() - t:.0f}s", flush=True)
        ids, docs, vocab, freq = encode_corpus(texts_of(corpus), tok)
        ids.tofile(out / "ids.bin")
        np.save(out / "doc_lengths.npy", np.bincount(docs).astype(np.int64))
        (out / "vocab.json").write_text(json.dumps(vocab.tolist()))
        np.save(out / "freq.npy", freq)
        print(f"encoded {len(ids):,} tokens, {len(vocab):,} words, {docs[-1] + 1:,} documents: {time.time() - t:.0f}s", flush=True)
    ids = np.fromfile(out / "ids.bin", dtype=np.uint16)
    docs = np.repeat(np.arange(len(lengths := np.load(out / "doc_lengths.npy")), dtype=np.int32), lengths)
    V = len(json.loads((out / "vocab.json").read_text()))
    C = cooccurrence_matrix(ids, docs, V, window=WINDOW, weighting=weighting)
    np.save(out / f"C_w{WINDOW}_{weighting.replace('/', '')}.npy", C)
    print(f"matrix {C.shape}, weighting {weighting}: {time.time() - t:.0f}s")
