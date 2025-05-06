import pandas as pd
import numpy as np
import pickle
import time
import os
import openai
from dotenv import load_dotenv

# Load .env variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load your CSV file
df = pd.read_csv("bionexa_tests_full_varied.csv")

# Format each row into a clearly labeled string
docs = df.apply(
    lambda row: (
        f"Test: {row.get('test_name', '')}, "
        f"Price: R{row.get('price_ZAR', 'N/A')}, "
        f"Turnaround: {row.get('turnaround_days', 'N/A')} days, "
        f"Sample prep: {row.get('sample_prep', '')}, "
        f"Notes: {row.get('notes', '')}"
    ),
    axis=1
).tolist()

# Get embeddings in batches
embeddings = []
batch_size = 100

for i in range(0, len(docs), batch_size):
    batch = docs[i:i+batch_size]
    print(f"Embedding batch {i} to {i+len(batch)}...")
    try:
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        embeddings.extend([r.embedding for r in response.data])
    except Exception as e:
        print(f"❌ Error embedding batch {i}-{i+len(batch)}: {e}")
        break
    time.sleep(1)  # Respect OpenAI rate limits

# Save embeddings and metadata
with open("kb.pkl", "wb") as f:
    pickle.dump((embeddings, docs, df.to_dict("records")), f)

print(f"✅ Saved {len(embeddings)} embeddings.")
