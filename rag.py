# rag.py

import os
import pickle
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Set up OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load the FAISS index and records
with open("kb.pkl", "rb") as f:
    index, docs, records = pickle.load(f)

def retrieve_top_k(query, k=5):
    """
    Embed the query and return top-k matching records using FAISS.
    """
    embedding_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    )
    q_embed = embedding_response.data[0].embedding

    D, I = index.search(np.array([q_embed]).astype("float32"), k)
    return [records[i] for i in I[0]]

def generate_answer(query):
    """
    Generate a natural language answer to a user query based on retrieved context.
    """
    if "where" in query.lower() and "drop" in query.lower():
        return (
            "📍 Samples can be dropped at:\n"
            "**Modderfontein Industrial Complex**, Standerton Avenue, via Nobel Gate, Modderfontein, Gauteng, South Africa, 1645.\n"
            "Please go to the **permit office on the corner of Nobel Avenue and Standerton Road**.\n"
            "Ensure all bottles are clearly labeled before delivery."
        )

    context_rows = retrieve_top_k(query, k=5)

    context = "\n".join(
        f"• Test: {r.get('test_name', '')} | Price: R{r.get('price_ZAR', 'N/A')} | "
        f"Turnaround: {r.get('turnaround_days', 'N/A')} days | "
        f"Sample prep: {str(r.get('sample_prep') or '')} | "
        f"Notes: {str(r.get('notes') or '')}"
        for r in context_rows
    )

    messages = [
        {"role": "system", "content": (
            "You are a helpful assistant for Bionexa Lab. Use the context below to answer customer questions "
            "about test pricing, turnaround times, sample preparation, or drop-off procedures. Only say 'I’m not 100% sure – let me arrange a call with our support team.' "
            "if there is truly no relevant information.")},
        {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}"}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    return response.choices[0].message.content
