'''
Use streamlit to create a UI for assitant created.
'''

import streamlit as st
from assistant import create_assistant


assistant = create_assistant()

st.title("LLM Zoomcamp Course Assistant")

user_input = st.text_input("Enter your question:")

if st.button("Ask"):
    with st.spinner("Processing..."):
        answer = assistant.rag(user_input)
        st.success("Completed!")
        st.write(answer)
