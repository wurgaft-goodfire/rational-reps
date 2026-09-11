"""One counting pass over a corpus: windowed co-occurrence of the month, color and number concepts
with the 25k most frequent tokens, and the concept x concept block of each domain. Writes one table
per domain to ``$CACHE_DIR/corpus``. Run: ``uv run python -m src.notebooks.count_corpus wikipedia``."""
import os
import sys
import time
from pathlib import Path

import numpy as np

from src.code.corpus import at_least, cooccurrence_table, hub_texts, joined, window_counts

WINDOW, K = 16, 25000
CORPORA = {  # repo, shard prefix, text column, article filter (min words), lines joined per passage
    "wikipedia": {"repo": "wikimedia/wikipedia", "prefix": "20231101.en/", "column": "text", "min_words": 200,
                  "join": 1, "tag": "wikipedia_20231101_en"},   # English Wikipedia 2023-11, articles of >= 200 words
    "babylm": {"repo": "deven367/babylm-100M", "prefix": "data/train-", "column": "text", "min_words": 0,
               "join": 100, "tag": "babylm_100M"},              # BabyLM 100M: one utterance per row, joined into passages
    "gutenberg": {"repo": "sedthh/gutenberg_english", "prefix": "data/train-", "column": "TEXT", "min_words": 200,
                  "join": 1, "tag": "gutenberg_en"},            # Project Gutenberg English books, one book per row
}
DOMAINS = {"months": ["january", "february", "march", "april", "may", "june", "july", "august",
                      "september", "october", "november", "december"],
           # hue order, then the achromatic ones; words with at least 20k occurrences in Wikipedia
           "colors": ["red", "scarlet", "crimson", "orange", "yellow", "lime", "olive", "green", "blue", "violet",
                      "purple", "pink", "brown", "black", "white", "gray"],
           "numbers": [str(n) for n in range(1, 501)]}

if __name__ == "__main__":
    corpus = CORPORA[sys.argv[1]]
    out = Path(os.environ.get("CACHE_DIR", "outputs")) / "corpus"; out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    texts = hub_texts(corpus["repo"], corpus["prefix"], column=corpus["column"])
    texts = joined(texts, corpus["join"]) if corpus["join"] > 1 else at_least(texts, corpus["min_words"])
    counts, freq = window_counts(texts, [w for ws in DOMAINS.values() for w in ws], window=WINDOW, workers=40, chunk=20000)
    for name, words in DOMAINS.items():
        C, vocab = cooccurrence_table(counts, freq, words, K)
        block = np.array([[counts[c].get(w, 0) for w in words] for c in words], dtype=np.float64)   # concept x concept
        np.savez(out / f"{corpus['tag']}_{name}_w{WINDOW}.npz", C=C.astype(np.float32), block=block, vocab=np.array(vocab),
                 concepts=np.array(words), concept_counts=np.array([freq[w] for w in words]),
                 word_counts=np.array([freq[w] for w in vocab]), total=sum(freq.values()))
        print(name, C.shape, "occurrences", {w: freq[w] for w in words[:12]})
    print(f"total tokens {sum(freq.values()):,}; {time.time() - t:.0f}s")
