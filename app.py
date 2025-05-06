# app.py

import streamlit as st
from rag import generate_answer

st.set_page_config(page_title="Bionexa RAG Assistant")

st.title("🧬 Bionexa Lab Assistant")
st.markdown("Ask about our tests, turnaround time, sample drop-off, or preparation.")

query = st.text_input("Your question:")

if query:
    try:
        response = generate_answer(query)
        st.success(response)
    except Exception as e:
        st.error("❌ An error occurred while processing your question.")
        st.exception(e)
