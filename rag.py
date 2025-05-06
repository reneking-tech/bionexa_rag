# rag.py

import os
import pickle
import numpy as np
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# load the index, embeddings array, and raw records
with open("kb.pkl", "rb") as f:
    EMBS, RECORDS, NN = pickle.load(f)

def retrieve_top_k(query, k=5):
    resp = openai.Embedding.create(model="text-embedding-3-small", input=query)
    q_emb = np.array(resp["data"][0]["embedding"], dtype="float32")
    dists, idxs = NN.kneighbors([q_emb], n_neighbors=k)
    return [RECORDS[i] for i in idxs[0]]

def generate_answer(query: str) -> str:
    ql = query.lower()
    # simple rule‐based fallback
    if "where" in ql and "drop" in ql:
        return (
            "📍 Samples can be dropped at:\n"
            "**Modderfontein Industrial Complex**, Standerton Avenue, via Nobel Gate, Modderfontein, Gauteng, South Africa, 1645.\n"
            "Please go to the **permit office on the corner of Nobel Avenue and Standerton Road**.\n"
            "Ensure all bottles are clearly labeled before delivery."
        )

    # retrieve context
    rows = retrieve_top_k(query, k=5)
    context = "\n".join(
        f"• Test: {r.get('test_name','')} | Price: R{r.get('price_ZAR','')} | "
        f"Turnaround: {r.get('turnaround_days','')} days | Prep: {r.get('sample_prep','')} | "
        f"Notes: {r.get('notes','')}"
        for r in rows
    )

    system = (
        "You are a helpful assistant for Bionexa Lab. Use the context below to answer customer questions "
        "about test pricing, turnaround times, sample preparation, or drop-off procedures. "
        "If the context includes a physical address or instructions for sample delivery, include those clearly. "
        "If you can’t find an answer, say “I’m not 100% sure – let me arrange a call with our support team.”"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": f"Question: {query}\n\nContext:\n{context}"}
    ]

    resp = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    return resp.choices[0].message.content
