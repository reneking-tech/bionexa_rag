# rag.py

import os
import pickle
import numpy as np
import openai
from dotenv import load_dotenv

# Load key (locally) or rely on Streamlit’s st.secrets
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Pull in your prebuilt embeddings + sklearn index + raw records
with open("kb.pkl", "rb") as f:
    EMBS, RECORDS, NN = pickle.load(f)

def retrieve_top_k(query: str, k: int = 3):
    """Embed the query & return the k nearest records."""
    resp = openai.Embedding.create(
        model="text-embedding-3-small",
        input=query
    )
    q_emb = np.array(resp["data"][0]["embedding"], dtype="float32")
    dists, idxs = NN.kneighbors([q_emb], n_neighbors=k, return_distance=True)
    return [RECORDS[i] for i in idxs[0]]

def generate_answer(query: str) -> str:
    # simple rule‐based fallback
    ql = query.lower()
    if "where" in ql and "drop" in ql:
        return (
            "📍 Samples can be dropped at:\n"
            "**Modderfontein Industrial Complex**, Standerton Avenue, via Nobel Gate, Modderfontein, Gauteng, South Africa, 1645.\n"
            "Please go to the **permit office on the corner of Nobel Avenue and Standerton Road**.\n"
            "Ensure all bottles are clearly labeled before delivery."
        )

    # pull top-3
    top3 = retrieve_top_k(query, k=3)
    context = "\n".join(
        f"• **{r['test_name']}** – Price R{r['price_ZAR']}, "
        f"{r['turnaround_days']}-day turnaround"
        for r in top3
    )

    system_prompt = (
        "You are a helpful assistant for Bionexa Lab. Use the context below to answer customer questions. "
        "If more than one test is relevant, list the three most relevant options by name, price, "
        "and turnaround in bullet points. "
        "If you truly can’t find an answer, say: “I’m not 100% sure – let me arrange a call with our support team.”"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}"}
    ]
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return resp.choices[0].message.content
