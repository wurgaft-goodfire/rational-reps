"""One counting pass over English Wikipedia (2023-11, articles of at least 200 words): windowed
co-occurrence of the month, color and number concepts with the 25k most frequent tokens.
Writes one table per domain to ``$CACHE_DIR/corpus``. Run: ``uv run python -m src.notebooks.count_wikipedia``."""
import os
import time
from pathlib import Path

import numpy as np

from src.code.corpus import at_least, cooccurrence_table, hub_texts, window_counts

CORPUS, MIN_WORDS, WINDOW, K = ("wikimedia/wikipedia", "20231101.en/"), 200, 16, 25000
TAG = "wikipedia_20231101_en"
DOMAINS = {"months": ["january", "february", "march", "april", "may", "june", "july", "august",
                      "september", "october", "november", "december"],
           "colors": ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white", "gray"],
           "numbers": [str(n) for n in range(1, 101)]}

if __name__ == "__main__":
    out = Path(os.environ.get("CACHE_DIR", "outputs")) / "corpus"; out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    counts, freq = window_counts(at_least(hub_texts(*CORPUS), MIN_WORDS), [w for ws in DOMAINS.values() for w in ws],
                                 window=WINDOW, workers=40, chunk=20000)
    for name, words in DOMAINS.items():
        C, vocab = cooccurrence_table(counts, freq, words, K)
        np.savez(out / f"{TAG}_{name}_w{WINDOW}.npz", C=C.astype(np.float32), vocab=np.array(vocab), concepts=np.array(words),
                 concept_counts=np.array([freq[w] for w in words]), word_counts=np.array([freq[w] for w in vocab]),
                 total=sum(freq.values()))
        print(name, C.shape, "occurrences", {w: freq[w] for w in words[:12]})
    print(f"total tokens {sum(freq.values()):,}; {time.time() - t:.0f}s")
