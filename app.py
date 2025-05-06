# app.py

import streamlit as st
from rag import generate_answer
import openai

# Set OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Bionexa Assistant", page_icon="🧪")

st.title("🧪 Bionexa Lab Assistant")
st.markdown("Ask about our tests, turnaround time, sample drop-off, or preparation.")

query = st.text_input("Your question:")

if query:
    st.chat_message("user").markdown(query)
    response = generate_answer(query)
    st.chat_message("assistant").markdown(response)

    if response.startswith("I’m not 100% sure"):
        st.link_button("📞 Schedule a call", "https://calendly.com/your-support-team")
