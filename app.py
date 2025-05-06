# app.py

import streamlit as st
import openai
from rag import generate_answer

st.set_page_config(page_title="🧪 Bionexa Lab Assistant", page_icon="🧪")
st.title("🧪 Bionexa Lab Assistant")
st.markdown("Ask about our tests, turnaround time, sample drop-off, or preparation.")

# wire up the secret
if "OPENAI_API_KEY" not in st.secrets:
    st.error("🔥 Missing OPENAI_API_KEY in Streamlit secrets!")
    st.stop()
openai.api_key = st.secrets["OPENAI_API_KEY"]

query = st.text_input("Your question:")
if query:
    st.chat_message("user").markdown(query)
    try:
        answer = generate_answer(query)
        st.chat_message("assistant").markdown(answer)
        if answer.startswith("I’m not 100% sure"):
            st.markdown("[📞 Schedule a call](https://calendly.com/your-support-team)")
    except Exception as e:
        st.error(f"Error: {e}")
