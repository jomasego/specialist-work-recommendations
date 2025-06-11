import streamlit as st

st.set_page_config(layout="wide")

st.title("Specialist Work Recommendations AI")

st.write("Welcome to the platform!")

# Placeholder for chat interface
user_query = st.text_input("Ask a question or describe your talent needs:")

if user_query:
    st.write(f"You asked: {user_query}")
    # Backend call would go here
    st.info("Response from AI will appear here.")

# Placeholder for recommendations
st.sidebar.title("Recommended for you")
st.sidebar.info("Personalized recommendations will appear here.")
