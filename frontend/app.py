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
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:5001")

# --- Session State Initialization ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None # To store fetched recommendations
if 'user_id' not in st.session_state: # Example user_id, can be dynamic
    st.session_state.user_id = "user_123"

# --- API Helper Functions ---
def query_backend(current_query, chat_history):
    """Sends the current query and chat history to the Flask backend's /chat endpoint."""
    payload = {
        "query": current_query,
        "chat_history": chat_history
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
    st.session_state.chat_history[message_index]['feedback'] = feedback_type
    print(f"Feedback received for message {message_index}: {feedback_type}")
    st.toast("Thanks for your feedback!", icon="✅")

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

# --- Main Chat Interface ---
st.header("💬 Makers AI Assistant")

# Use a form for the chat input to allow submission on Enter
with st.form(key='chat_form'):
    user_query = st.text_input("Your question", placeholder="e.g., How do I get paid for a fixed-price project?", key="user_query_input")
    submit_button = st.form_submit_button(label='Ask')

if submit_button and user_query:
    # Add user query to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Get AI response from backend
    with st.spinner("Makers AI is thinking..."):
        # Pass the current user_query and the existing st.session_state.chat_history
        backend_response = query_backend(user_query, st.session_state.chat_history)
        
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
    
    # After getting a response from the assistant, update recommendations
    update_recommendations(force_update=True)

    # Rerun to update the sidebar with new recommendations immediately
    st.rerun()

# Display chat history
for i, message in enumerate(st.session_state.chat_history):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If the message is from the assistant, show sources and feedback buttons
        if message["role"] == "assistant":
            if message.get("sources"):
                st.info(f"Sources: {', '.join(message['sources'])}", icon="📚")
            
            # Feedback buttons
            feedback_key_base = f"feedback_{i}"
            # Check if feedback has already been given for this message
            if 'feedback' not in message:
                cols = st.columns([1, 1, 10]) # Adjust column ratios as needed
                with cols[0]:
                    st.button("👍", key=f"{feedback_key_base}_up", on_click=handle_feedback, args=(i, 'positive'))
                with cols[1]:
                    st.button("👎", key=f"{feedback_key_base}_down", on_click=handle_feedback, args=(i, 'negative'))

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
            st.markdown("### Suggested Freelancers")
            if st.session_state.recommendations.get('freelancers'):
                for rec in st.session_state.recommendations['freelancers']:
                    freelancer = rec['freelancer']
                    match_details = rec.get('match_details', {})

                    # Specialties
                    specialties_html = ""
                    if freelancer.get('specialties'):
                        specialties_list = ", ".join(freelancer['specialties'])
                        specialties_html = f"<p class='details'><b>Specialties:</b> {specialties_list}</p>"

                    # Skills as tags
                    skills_html = ""
                    if freelancer.get('skills'):
                        skills_tags = ""
                        for skill_item in freelancer['skills']:
                            # Simple tag for now, can be enhanced with proficiency later
                            skills_tags += f"<span style='background-color:#e0e0e0; color:#333; padding: 2px 6px; border-radius:4px; margin-right:4px; font-size:0.8em;'>{skill_item.get('name')}</span> "
                        if skills_tags:
                            skills_html = f"<p class='details'><b>Skills:</b> {skills_tags}</p>"

                    # Matched on details
                    matched_on_parts = []
                    if match_details.get('specialties'):
                        matched_on_parts.append(f"specialties: {', '.join(match_details['specialties'])}")
                    if match_details.get('skills'):
                        matched_on_parts.append(f"skills: {', '.join(match_details['skills'])}")
                    if match_details.get('title'):
                        matched_on_parts.append(f"title: {', '.join(match_details['title'])}")
                    if match_details.get('bio'):
                        matched_on_parts.append(f"bio: {', '.join(match_details['bio'])}")
                    
                    matched_on_str = "; ".join(matched_on_parts)
                    if not matched_on_str:
                        matched_on_str = "General relevance"

                    card_html = f"""
                    <div class="freelancer-card">
                        <h4>{freelancer.get('name', 'N/A')}</h4>
                        <p class="title">{freelancer.get('title', 'N/A')}</p>
                        {specialties_html}
                        {skills_html}
                        <p class="details">Rate: ${freelancer.get('hourly_rate_usd', 'N/A')}/hr</p>
                        <p class="details">Availability: {freelancer.get('availability', 'N/A')}</p>
                        <p class="matched-on">Matched on: {matched_on_str}</p>
                        <p class="details"><b>Match Strength:</b> {rec.get('match_percentage', 0)}% {get_star_rating(rec.get('match_percentage', 0))}</p>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)
            else:
                st.info("No specific freelancer recommendations at this time.")
            articles = st.session_state.recommendations.get("articles", [])
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
