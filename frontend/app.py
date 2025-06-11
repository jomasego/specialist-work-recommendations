import streamlit as st
import requests
import os
import json

# --- Page Configuration ---
# Use a relative path for the favicon, assuming it's in the same 'frontend' directory
# Streamlit will serve it correctly.
st.set_page_config(
    page_title="Makers AI Assistant",
    page_icon="favicon.ico",
    layout="wide"
)

# --- Custom CSS for Makers Vibe ---
# Inspired by makers.works: modern, clean, with specific brand colors.
st.markdown("""
<style>
    /* Main container and text */
    .stApp {
        background-color: #f9f9f9; /* Light grey background */
    }
    .st-emotion-cache-16txtl3, .st-emotion-cache-10trblm, h1, h2, h3, .st-emotion-cache-163ttbj h3 {
        color: #1a1a1a; /* Dark grey for text */
    }
    /* Buttons */
    .stButton>button {
        background-color: #0052cc; /* Professional blue */
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #0041a3;
    }
    /* Text Input */
    .stTextInput>div>div>input {
        background-color: #ffffff;
        border: 1px solid #dcdcdc;
        border-radius: 8px;
    }
    /* Sidebar */
    .st-emotion-cache-163ttbj {
        background-color: #f0f2f5; /* Lighter grey for sidebar */
    }
    /* Recommendation Card Styling */
    .freelancer-card {
        border: 1px solid #e6e6e6;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        background-color: #ffffff;
    }
    .freelancer-card h4 {
        margin-bottom: 2px;
        color: #1a1a1a;
        font-size: 1.1em;
    }
    .freelancer-card .title {
        font-size: 0.9em;
        color: #555;
        margin-bottom: 12px;
    }
    .freelancer-card .details {
        font-size: 0.9em;
        color: #333;
        margin-bottom: 8px;
    }
    .freelancer-card .matched-on {
        font-size: 0.85em;
        color: #0052cc; /* Blue accent for matched skills */
    }
</style>
""", unsafe_allow_html=True)

# --- Backend API Configuration ---
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:5001/query")
RECOMMENDATIONS_URL = os.getenv("RECOMMENDATIONS_URL", "http://localhost:5001/recommendations")

# --- Session State Initialization ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None # To store fetched recommendations
if 'user_id' not in st.session_state: # Example user_id, can be dynamic
    st.session_state.user_id = "user_123"

# --- API Helper Functions ---
def query_backend(query):
    """Sends a query to the Flask backend and returns the response."""
    try:
        response = requests.post(BACKEND_URL, json={"query": query}, timeout=120)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to connect to the backend. Please ensure it's running. Details: {e}"}

def get_recommendations_from_backend(query_history, current_query):
    """Calls the backend to get recommendations."""
    payload = {
        "query_history": [msg['content'] for msg in query_history if msg['role'] == 'user'],
        "current_query": current_query
    }
    print(f"Frontend: Sending to /recommendations: {payload}")
    try:
        response = requests.post(RECOMMENDATIONS_URL, json=payload, timeout=20)
        response.raise_for_status() # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Frontend: Error calling recommendations API: {e}")
        return {"error": f"Failed to get recommendations. Details: {e}"}

# --- UI Layout ---

# Display the logo
# Construct path relative to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
logo_filename = "MAKERS-logo.svg"
logo_path = os.path.join(script_dir, logo_filename)

try:
    st.image(logo_path, width=200) # Adjust width as needed
except Exception as e:
    st.warning(f"Could not load logo from {logo_path}. Please ensure the file exists. Error: {e}")

st.title("Makers AI Assistant 🤝")
st.write("Ask me anything about the Makers platform. I can help with questions about payments, finding talent, and best practices.")

# Main query input
user_query = st.text_input(
    "Your question",
    placeholder="e.g., How do I get paid for a fixed-price project?",
    label_visibility="collapsed"
)

if st.button("Ask", type="primary") and user_query: # This block runs when the user enters a query
    # Add user query to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Get AI response from backend
    with st.spinner("Makers AI is thinking..."):
        backend_response = query_backend(user_query)
        
        if "error" in backend_response:
            ai_response_content = f"Sorry, I encountered an error: {backend_response['error']}"
            sources = []
        else:
            ai_response_content = backend_response.get("answer", "Sorry, I couldn't find an answer.")
            sources = backend_response.get("sources", [])

    # Display AI response
    with st.chat_message("assistant"):
        st.markdown(ai_response_content)
        if sources:
            with st.expander("View Sources"):
                for source in sources:
                    st.caption(f"- {source}")
    
    # Add AI response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": ai_response_content, "sources": sources})
    
    # Fetch and store recommendations
    with st.spinner("Fetching recommendations..."):
        recommendations_data = get_recommendations_from_backend(st.session_state.chat_history, user_query)
        if "error" in recommendations_data:
            st.session_state.recommendations = {"error": recommendations_data['error']}
        else:
            st.session_state.recommendations = recommendations_data
    
    # Rerun to update the sidebar with new recommendations immediately
    st.rerun()

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("View Sources"):
                for source in message["sources"]:
                    st.caption(f"- {source}")

# --- Sidebar ---
with st.sidebar:
    st.header("About Makers AI")
    st.markdown("""
    This AI assistant is designed to help you navigate the Makers platform, 
    answer your questions, and provide recommendations.
    
    **Powered by Gemini & RAG.**
    """)
    st.divider()
    st.header("Recommendations Panel")

    if st.session_state.recommendations:
        if "error" in st.session_state.recommendations:
            st.error(f"Could not load recommendations: {st.session_state.recommendations['error']}")
        else:
            freelancers = st.session_state.recommendations.get("freelancers", [])
            articles = st.session_state.recommendations.get("articles", [])

            if not freelancers and not articles:
                st.info("No specific recommendations at the moment. Keep chatting to get personalized suggestions!")
            
            if freelancers:
                st.subheader("Suggested Freelancers")
                for rec in freelancers:
                    freelancer = rec.get('freelancer', {})
                    match_details = rec.get('match_details', {})
                    matched_skills = match_details.get('skills', [])
                    
                    st.markdown(f"""
                    <div class="freelancer-card">
                        <h4>{freelancer.get('name', 'N/A')}</h4>
                        <div class="title">{freelancer.get('title', 'N/A')}</div>
                        <div class="details">
                            <strong>Rate:</strong> ${freelancer.get('hourly_rate_usd', 'N/A')}/hr<br>
                            <strong>Availability:</strong> {freelancer.get('availability', 'N/A')}
                        </div>
                        {'<div class="matched-on"><b>Matched on:</b> ' + ', '.join(matched_skills) + '</div>' if matched_skills else ''}
                    </div>
                    """, unsafe_allow_html=True)
            
            if articles:
                st.subheader("Relevant Articles")
                for rec in articles:
                    article_name = rec.get('article_name', 'N/A')
                    # Clean up article name for display
                    display_article_name = article_name.replace('_', ' ').replace('.md', '')
                    st.markdown(f"📄 **{display_article_name.title()}**") # Title case for better readability
                    if rec.get('matched_keywords'):
                        st.caption(f"_Keywords: {', '.join(rec.get('matched_keywords'))}_ ")
                    # Future: Add a button or link to view the article content if routes are set up
                    # st.button(f"Read {display_article_name.title()}", key=f"article_{article_name}")
                    st.markdown("---") # Visually separates article entries
    else:
        st.info("Personalized recommendations will appear here based on your conversation.")
