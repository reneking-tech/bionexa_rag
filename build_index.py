# build_index.py

import os
import pickle
import pandas as pd
import numpy as np
import openai
from dotenv import load_dotenv
from sklearn.neighbors import NearestNeighbors

# ─── Config ─────────────────────────────────────────────────────────────────────
CSV_PATH     = "bionexa_tests_full_varied.csv"
OUTPUT_PATH  = "kb.pkl"
EMBED_MODEL  = "text-embedding-3-small"
BATCH_SIZE   = 100
# ────────────────────────────────────────────────────────────────────────────────

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embeddings(texts):
    all_embs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        try:
            resp = openai.Embedding.create(model=EMBED_MODEL, input=batch)
            embs = [d["embedding"] for d in resp["data"]]
            all_embs.extend(embs)
            print(f"✅ Embedded batch {i}–{i+len(batch)}")
        except Exception as e:
            print(f"❌ Error embedding batch {i}–{i+len(batch)}: {e}")
    return np.array(all_embs, dtype="float32")

def main():
    # 1) load your CSV  
    df = pd.read_csv(CSV_PATH)
    records = df.to_dict("records")

    # 2) turn each row into a context string
    docs = [
        f"Test: {r.get('test_name','')} | Price: R{r.get('price_ZAR','')} | "
        f"Turnaround: {r.get('turnaround_days','')} days | Prep: {r.get('sample_prep','')} | "
        f"Notes: {r.get('notes','')}"
        for r in records
    ]

    # 3) get embeddings
    embs = get_embeddings(docs)

    # 4) fit a lightweight nearest-neighbors index
    nn = NearestNeighbors(n_neighbors=5, metric="cosine")
    nn.fit(embs)

    # 5) serialize everything
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump((embs, records, nn), f)

    print(f"✅ Saved {len(records)} records to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
