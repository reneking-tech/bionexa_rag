# build_index.py

import pandas as pd
import numpy as np
import faiss
import pickle
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables (for OpenAI key)
load_dotenv()
client = OpenAI()

# Load your CSV file
df = pd.read_csv("bionexa_tests_full_varied.csv")

# 🧠 Format each row into a clearly labeled text string
docs = df.apply(
    lambda row: (
        f"Test: {row.get('test_name', '')}, "
        f"Price: R{row.get('price_ZAR', 'N/A')}, "
        f"Turnaround time: {row.get('turnaround_days', 'N/A')} days, "
        f"Sample preparation: {row.get('sample_prep', '')}, "
        f"Notes: {row.get('notes', '')}"
    ),
    axis=1
).tolist()

# 🧠 Create embeddings using OpenAI's embedding model
embeddings = [
    client.embeddings.create(
        model="text-embedding-3-small",
        input=doc
    ).data[0].embedding for doc in docs
]

# ⚙️ Set up FAISS index (L2 distance)
dim = len(embeddings[0])
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings).astype("float32"))

# 💾 Save index, documents, and original records
with open("kb.pkl", "wb") as f:
    pickle.dump((index, docs, df.to_dict("records")), f)

print(f"✅ Vector store built and saved with {len(docs)} records.")
