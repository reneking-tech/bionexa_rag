# rag.py

import os
import numpy as np
import pickle
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# Load .env if running locally; otherwise use Streamlit secrets on Cloud
if not st.secrets.get("OPENAI_API_KEY"):
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
else:
    openai_api_key = st.secrets["OPENAI_API_KEY"]

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

# Load saved FAISS index, raw documents, and full record rows
with open("kb.pkl", "rb") as f:
    index, docs, records = pickle.load(f)

def retrieve_top_k(query, k=5):
    """
    Embed the query and return top-k matching records using FAISS.
    """
    q_embed = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    D, I = index.search(np.array([q_embed]).astype("float32"), k)
    return [records[i] for i in I[0]]  # Return full record dicts

def generate_answer(query):
    query_lower = query.lower()

    # OPTIONAL: fallback hardcoded rule (can be removed if CSV entry is solid)
    if "where" in query_lower and "drop" in query_lower:
        return (
            "📍 Samples can be dropped at:\n"
            "**Modderfontein Industrial Complex**, Standerton Avenue, via Nobel Gate, Modderfontein, Gauteng, South Africa, 1645.\n"
            "Please go to the **permit office on the corner of Nobel Avenue and Standerton Road**.\n"
            "Ensure all bottles are clearly labeled before delivery."
        )

    # 🔍 Top-k retrieval
    context_rows = retrieve_top_k(query, k=5)

    # Format context nicely
    context = "\n".join(
        f"• Test: {r.get('test_name', '')} | Price: R{r.get('price_ZAR', 'N/A')} | "
        f"Turnaround: {r.get('turnaround_days', 'N/A')} days | "
        f"Sample prep: {str(r.get('sample_prep') or '')} | "
        f"Notes: {str(r.get('notes') or '')}"
        for r in context_rows
    )

    # 🧠 System prompt tailored to include drop-off instructions
    system_prompt = (
        "You are a helpful assistant for Bionexa Lab. Use the context below to answer customer questions "
        "about test pricing, turnaround times, sample preparation, or drop-off procedures. "
        "If the context includes a physical address or instructions for sample delivery, include those clearly in your reply. "
        "Do your best to infer answers from the examples. Only say 'I’m not 100% sure – let me arrange a call with our support team.' "
        "if there is truly no relevant information."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}"}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",  # or "gpt-3.5-turbo"
        messages=messages
    )

    return response.choices[0].message.content.strip()
