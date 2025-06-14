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
        background-color: #ffffff;
        border: 1px solid #e1e4e8;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: box-shadow 0.2s ease-in-out;
    }
    .freelancer-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .freelancer-card-chat {
        background-color: #f9f9f9;
        border: 1px solid #e1e4e8;
        border-radius: 8px;
        padding: 12px;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .freelancer-card-sidebar {
        border: 1px solid #e6e6e6;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 10px;
        background-color: #fafafa;
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
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:5002")

# --- Session State Initialization ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None # To store fetched recommendations
if 'user_id' not in st.session_state: # Example user_id, can be dynamic
    st.session_state.user_id = "user_123"
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = 'gemini-1.5-flash-latest' # Default model

# --- API Helper Functions ---
def query_backend(current_query, chat_history):
    """Sends the current query and chat history to the Flask backend's /chat endpoint."""
    payload = {
        "query": current_query,
        "chat_history": chat_history,
        "model": st.session_state.get('selected_model', 'gemini-1.5-flash-latest') # Pass selected model
    }
    chat_url = f"{BACKEND_URL}/chat"
    print(f"Frontend: Sending to {chat_url}") # For debugging
    try:
        response = requests.post(chat_url, json=payload, timeout=120)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to connect to the backend. Please ensure it's running. Details: {e}"}

def get_star_rating(percentage):
    """Converts a match percentage to a 1-5 star rating string."""
    if percentage is None: percentage = 0
    if percentage <= 20:
        return "★☆☆☆☆"  # 1 star
    elif percentage <= 40:
        return "★★☆☆☆"  # 2 stars
    elif percentage <= 60:
        return "★★★☆☆"  # 3 stars
    elif percentage <= 80:
        return "★★★★☆"  # 4 stars
    else:
        return "★★★★★"  # 5 stars

def handle_feedback(message_index, feedback_type):
    """Sends feedback to the backend and updates the session state."""
    st.session_state.chat_history[message_index]['feedback'] = feedback_type
    
    payload = {
        "message_index": message_index,
        "feedback_type": feedback_type,
        "user_id": st.session_state.user_id,
        "chat_history": st.session_state.chat_history
    }
    feedback_url = f"{BACKEND_URL}/feedback"
    try:
        response = requests.post(feedback_url, json=payload, timeout=10)
        response.raise_for_status()
        st.toast("Thanks for your feedback!", icon="✅")
        print(f"Successfully sent feedback to backend: {feedback_type}")
    except requests.exceptions.RequestException as e:
        st.error("Could not submit feedback. Please try again.")
        print(f"Error sending feedback: {e}")
        # Revert feedback state on error
        del st.session_state.chat_history[message_index]['feedback']

def update_recommendations(force_update=False):
    """Fetches recommendations based on the entire chat history."""
    if force_update or (st.session_state.chat_history and st.session_state.chat_history[-1]['role'] == 'assistant'):
        print("\nUpdating recommendations based on full chat history...")
        payload = {"chat_history": st.session_state.chat_history}
        recommendations_url = f"{BACKEND_URL}/recommendations"
        try:
            response = requests.post(recommendations_url, json=payload, timeout=20)
            response.raise_for_status()
            recs = response.json()
            st.session_state.recommendations = recs
            print(f"Recommendations updated: {len(recs.get('freelancers', []))} freelancers, {len(recs.get('articles', []))} articles")
        except requests.exceptions.RequestException as e:
            print(f"Error fetching recommendations: {e}")
            st.session_state.recommendations = {"freelancers": [], "articles": [], "error": str(e)}

def handle_submit():
    """
    Callback function to handle chat submission.
    Processes the user query, gets a response, updates recommendations,
    and clears the input box.
    """
    user_query = st.session_state.user_query_input
    if user_query:
        # Add user query to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Get AI response from backend
        with st.spinner("Makers AI is thinking..."):
            backend_response = query_backend(user_query, st.session_state.chat_history)
            
            if "error" in backend_response:
                ai_response_content = f"Sorry, I encountered an error: {backend_response['error']}"
                sources = []
            else:
                ai_response_content = backend_response.get("answer", "Sorry, I couldn't find an answer.")
                sources = backend_response.get("sources", [])

        # Add AI response to chat history
        assistant_message = {"role": "assistant", "content": ai_response_content, "sources": sources}
        st.session_state.chat_history.append(assistant_message)
        
        # Attach the query-specific recommendations to the message for in-chat display
        if "query_recommendations" in backend_response:
            assistant_message['recommendations'] = backend_response["query_recommendations"]

        # After processing the main response, update the global recommendations for the sidebar
        update_recommendations(force_update=True)

        # Clear the input box. This is safe inside a callback.
        st.session_state.user_query_input = ""

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

# --- Sidebar for Recommendations and Controls ---
st.sidebar.title("Tools & Recommendations 🛠️")

# Add a model selector to the sidebar
st.session_state.selected_model = st.sidebar.selectbox(
    "Choose your AI Model:",
    ("gemini-1.5-flash-latest", "llama3-8b-8192", "qwen-qwq-32b"),
    key='model_selector', # Add a key to persist selection
    format_func=lambda x: {
        "gemini-1.5-flash-latest": "Gemini 1.5 Flash (Balanced)",
        "llama3-8b-8192": "Groq Llama3 8B (Fast)",
        "qwen-qwq-32b": "Qwen QwQ 32B (Reasoning)"
    }.get(x, x),
    help="Gemini is great for complex reasoning. Groq offers near-instant responses for general queries. Qwen excels at reasoning tasks."
)

st.title("Makers AI Assistant 🤝")
st.write("Ask me anything about the Makers platform. I can help with questions about payments, finding talent, and best practices.")

# --- Main Chat Interface ---
st.header("💬 Makers AI Assistant")

# Use a form for the chat input to allow submission on Enter
# Use a form for the chat input to allow submission on Enter and clear after.
with st.form(key='chat_form'):
    st.text_input(
        "Your question", 
        placeholder="e.g., How do I get paid for a fixed-price project?", 
        key="user_query_input"
    )
    st.form_submit_button(label='Ask', on_click=handle_submit)

# Display chat history
for i, message in enumerate(st.session_state.chat_history):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # If the message is from the assistant, show extras like recommendations, sources, and feedback
        if message["role"] == "assistant":
            # Display recommendations if they exist for this message
            if "recommendations" in message and message["recommendations"]:
                freelancers = message["recommendations"].get("freelancers", [])
                articles = message["recommendations"].get("articles", [])

                if freelancers:
                    st.write("--- ")
                    st.subheader("✨ Recommended Freelancers")
                    for f in freelancers:
                        match_percentage = f.get('match_strength', {}).get('percentage', 0)
                        star_rating = get_star_rating(match_percentage)
                        st.markdown(f"""
                        <div class="freelancer-card-chat">
                            <h4>{f.get('name', 'N/A')}</h4>
                            <div class="title">{f.get('title', 'N/A')}</div>
                            <div class="details">
                                <b>Match:</b> {match_percentage:.0f}% {star_rating}<br>
                                <b>Specialties:</b> {', '.join(f.get('specialties', ['N/A']))}<br>
                            </div>
                            <div class="matched-on">
                                <i>Matched on: {', '.join(f.get('match_strength', {}).get('matched_on', []))}</i>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                if articles:
                    st.write("--- ")
                    st.subheader("📚 Suggested Articles")
                    for article in articles:
                        st.markdown(f"- [{article['title']}]({article['url']})")

            # Display sources if they exist
            if "sources" in message and message["sources"]:
                with st.expander("View Sources"):
                    for source in message["sources"]:
                        st.caption(f"- {source}")
            
            # Feedback buttons
            st.write("") # Spacer
            feedback_key_base = f"feedback_{i}"
            cols = st.columns(12)
            with cols[0]:
                st.button("👍", key=f"{feedback_key_base}_up", on_click=handle_feedback, args=(i, "positive"))
            with cols[1]:
                st.button("👎", key=f"{feedback_key_base}_down", on_click=handle_feedback, args=(i, "negative"))

# --- Sidebar ---
with st.sidebar:
    st.header("About Makers AI")
    st.markdown("""
    This AI assistant is designed to help you navigate the Makers platform, 
    answer your questions, and provide recommendations.
    
    **Powered by Gemini, Groq & RAG.**
    """)
    st.markdown("---")
    st.header("⭐ Top Matches")

    if 'recommendations' not in st.session_state or st.session_state.recommendations is None or not (st.session_state.recommendations.get("freelancers") or st.session_state.recommendations.get("articles")):
        st.info("Top recommendations based on the conversation will appear here.")
    else:
        freelancers = st.session_state.recommendations.get("freelancers", [])
        articles = st.session_state.recommendations.get("articles", [])

        if freelancers:
            st.subheader("Recommended Freelancers")
            for f_rec in freelancers:
                f = f_rec.get("freelancer", {})
                match_percentage = f_rec.get('match_percentage', 0)
                star_rating = get_star_rating(match_percentage)
                matched_on = f_rec.get('match_details', {})
                
                # Consolidate matched keywords for display
                matched_on_parts = []
                for key, keywords in matched_on.items():
                    if keywords:
                        matched_on_parts.append(f"{', '.join(keywords)}")
                matched_on_str = "; ".join(matched_on_parts)

                st.markdown(f"""
                <div class="freelancer-card-sidebar">
                    <h4>{f.get('name', 'N/A')}</h4>
                    <div class="title">{f.get('title', 'N/A')}</div>
                    <div class="details">
                        <b>Match:</b> {match_percentage:.0f}% {star_rating}<br>
                        <b>Specialties:</b> {', '.join(f.get('specialties', ['N/A']))}
                    </div>
                    <div class="matched-on">
                        <i>Matched on: {matched_on_str}</i>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        if articles:
            st.subheader("Suggested Articles")
            for article_rec in articles:
                article_name = article_rec.get('article_name', 'N/A')
                display_name = article_name.replace('_', ' ').replace('.md', '').title()
                st.markdown(f"📄 **{display_name}**")
                if article_rec.get('matched_keywords'):
                    st.caption(f"Keywords: {', '.join(article_rec.get('matched_keywords'))}")
