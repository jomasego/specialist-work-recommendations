import streamlit as st
import requests
import logging
import html
import os
import pandas as pd
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
if 'api_call_costs' not in st.session_state:
    st.session_state.api_call_costs = [] # To store total_cost after each API call


# --- API Helper Functions ---
def query_backend(current_query, chat_history):
    """Sends the current query and chat history to the Flask backend's /chat endpoint."""
    payload = {
        "query": current_query,
        "chat_history": chat_history
        # Model selection is now handled by the multi-agent backend.
    }
    chat_url = f"{BACKEND_URL}/chat"
    print(f"Frontend: Sending to {chat_url}") # For debugging
    try:
        response = requests.post(chat_url, json=payload, timeout=120)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        error_message = f"HTTP error occurred: {http_err}"
        if http_err.response is not None:
            try:
                error_detail = http_err.response.json()
                error_message += f" - Backend Details: {json.dumps(error_detail)}"
            except json.JSONDecodeError:
                error_message += f" - Backend Response (not JSON): {http_err.response.text}"
        return {"error": error_message}
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
        "session_id": st.session_state.get("session_id", "unknown"),
        "message_id": message_index,  # Use message index as message ID
        "rating": 1 if feedback_type == "positive" else -1,
        "comment": None,  # Optional comment
        "user_id": st.session_state.user_id,
        "recommendation_id": None  # Optional recommendation ID
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
            
            # Ensure we have the right data structure for recommendations
            # But don't modify the response structure, just store it as-is
            st.session_state.recommendations = recs
            print(f"Recommendations updated: {len(recs.get('freelancers', []))} freelancers, {len(recs.get('articles', []))} articles")
            
            # Print first few freelancers for debugging
            for i, f in enumerate(recs.get('freelancers', [])):
                if i < 5:  # Only print first 5 for brevity
                    print(f"Freelancer {i+1}: {f.get('freelancer', {}).get('name', 'N/A') if 'freelancer' in f else f.get('name', 'N/A')}")
                else:
                    break
            print(f"Total freelancers in recommendations: {len(recs.get('freelancers', []))}")
            print(f"Recommendation structure: {list(recs.keys())}")
            print(f"First recommendation item structure: {list(recs.get('freelancers', [])[0].keys()) if recs.get('freelancers') else 'No freelancers'}")

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
                agent_outcomes = []
            else:
                ai_response_content = backend_response.get("answer", "Sorry, I couldn't find an answer.")
                sources = backend_response.get("sources", [])
                agent_outcomes = backend_response.get("agent_outcomes", [])

        # Add AI response to chat history
        assistant_message = {
            "role": "assistant", 
            "content": ai_response_content, 
            "sources": sources,
            "agent_outcomes": agent_outcomes # Store the agent steps
        }
                # Attach the query-specific recommendations to the message for in-chat display
        if "query_recommendations" in backend_response:
            assistant_message['recommendations'] = backend_response["query_recommendations"]
            print(f"DEBUG In-Chat Recs: Received query_recommendations: {json.dumps(backend_response['query_recommendations'], indent=2)}")
            if not backend_response['query_recommendations'].get('freelancers'):
                print("DEBUG In-Chat Recs: 'freelancers' key is missing or empty in query_recommendations for in-chat display.")
        else:
            print("DEBUG In-Chat Recs: 'query_recommendations' key NOT FOUND in backend_response for in-chat display.")

        st.session_state.chat_history.append(assistant_message)
        
        # Update the agent logs in the right panel directly when response is received
        if agent_outcomes:
            print(f"Updating agent logs with {len(agent_outcomes)} items")
            st.session_state.agent_logs = agent_outcomes

        # Update API call costs for graph
        if "api_token_usage" in backend_response and isinstance(backend_response["api_token_usage"], dict):
            current_call_cost = backend_response["api_token_usage"].get("total_cost", 0.0)
            # We want to show accumulated cost, so we add to the previous total if it exists
            previous_total_cost = st.session_state.api_call_costs[-1] if st.session_state.api_call_costs else 0.0
            # The backend already returns the *accumulated* cost for the current agent graph run.
            # For a series of chat turns, we need to track the cost of *each turn* and then accumulate *that*.
            # However, the current backend `api_token_usage` in the response is the *total for that specific call's graph run*,
            # not an accumulation over multiple chat turns from the backend's perspective.
            # For simplicity in the frontend, we'll assume each 'total_cost' from the backend is the cost of *that specific interaction*.
            # We will then accumulate these on the frontend side for the graph.
            # Let's re-evaluate: The backend's `api_token_usage` in the response *is* the total for that call.
            # The `multi_agent_service` initializes `api_token_usage` to zeros for each `invoke` call if not passed in.
            # The `backend/app.py` *does* initialize it to zeros before calling `agent_graph.invoke`.
            # So, `backend_response["api_token_usage"]['total_cost']` is the cost of the *current* interaction only.
            st.session_state.api_call_costs.append(current_call_cost) 
            print(f"DEBUG: Appended cost {current_call_cost} to api_call_costs. New list: {st.session_state.api_call_costs}")

        # After processing the main response, update the global recommendations for the sidebar
        update_recommendations(force_update=True)

        # Clear the input box. This is safe inside a callback.
        st.session_state.user_query_input = ""

# --- UI Layout ---

# Use sidebars for tools/recommendations and agent workflow
sidebar_col = st.sidebar

# Main content area (not using columns anymore)
main_content = st

# Display the logo in the main content area
script_dir = os.path.dirname(os.path.abspath(__file__))
logo_filename = "MAKERS-logo.svg"
logo_path = os.path.join(script_dir, logo_filename)

try:
    st.image(logo_path, width=200) # Adjust width as needed
except Exception as e:
    st.warning(f"Could not load logo from {logo_path}. Please ensure the file exists. Error: {e}")

# --- Sidebar for Recommendations and Controls ---
sidebar_col.title("Cost, Agentic workflow, and Recommendations Monitoring")

sidebar_col.markdown("--- ")
# Cost Evolution Chart
sidebar_col.markdown("### 📈 API Cost Evolution")
if 'api_call_costs' in st.session_state and st.session_state.api_call_costs:
    call_numbers = list(range(1, len(st.session_state.api_call_costs) + 1))
    accumulated_costs_for_chart = []
    current_total_chart = 0
    for cost_item in st.session_state.api_call_costs:
        current_total_chart += cost_item # Cost from backend is per-interaction
        accumulated_costs_for_chart.append(current_total_chart)

    cost_data = {"Interaction": call_numbers, "Accumulated Cost ($)": accumulated_costs_for_chart}
    cost_df = pd.DataFrame(cost_data)
    sidebar_col.line_chart(cost_df.set_index("Interaction"), height=200)
else:
    sidebar_col.write("No API calls made yet.")
sidebar_col.markdown("--- ")

sidebar_col.info("Model selection is now automated by the multi-agent backend.")

# Set up the main title in the main content area
st.title("Makers AI Assistant 🤝")
st.write("Ask me anything about the Makers platform. I can help with questions about payments, finding talent, and best practices. Made with 💖 in Madrid 🇪🇸")

# Set up the agent log panel on the right sidebar
with st.sidebar:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧠 Agent Workflow")
    st.sidebar.info("This panel shows the multi-agent workflow and reasoning steps in real-time.")
    
    # Initialize session state for agent logs if it doesn't exist
    if 'agent_logs' not in st.session_state:
        st.session_state.agent_logs = []
    
    # Display current agent logs from session state
    if st.session_state.agent_logs:
        for i, log in enumerate(st.session_state.agent_logs):
            # Use different colors for different agent types
            if "Research Agent" in log:
                st.sidebar.success(f"• {log}")
            elif "Customer-Facing Agent" in log:
                st.sidebar.info(f"• {log}")
            elif "Manager Agent" in log:
                st.sidebar.warning(f"• {log}")
            elif "ERROR" in log:
                st.sidebar.error(f"• {log}")
            else:
                st.sidebar.info(f"• {log}")
    else:
        st.sidebar.write("No agent activities yet. Ask a question to see the agents in action.")
        
    st.sidebar.markdown("---")
    st.sidebar.header("About Makers AI")
    st.sidebar.markdown("""
    This AI assistant is designed to help you navigate the Makers platform, 
    answer your questions, and provide recommendations.
    
    **Powered by Gemini, Groq & RAG.**
    """)
    st.sidebar.markdown("---")
    st.sidebar.header("⭐ All Ranked Matches")
    
    # Display all recommended freelancers in the sidebar under 'All Ranked Matches'
    if 'recommendations' in st.session_state and st.session_state.recommendations is not None:
        freelancers = st.session_state.recommendations.get("freelancers", [])
        
        if freelancers:
            # Function to extract match percentage from any freelancer format
            def sidebar_get_match_percentage(f_rec):
                if isinstance(f_rec, dict):
                    if 'match_percentage' in f_rec:
                        return f_rec.get('match_percentage', 0)
                    elif 'match_strength' in f_rec:
                        return f_rec.get('match_strength', {}).get('percentage', 0)
                return 0
                
            # Sort freelancers by match percentage
            sidebar_sorted_freelancers = sorted(freelancers, key=sidebar_get_match_percentage, reverse=True)
            
            # Display all freelancers in the sidebar
            for f_rec in sidebar_sorted_freelancers:
                # Handle both data structures
                if 'freelancer' in f_rec:
                    f = f_rec.get("freelancer", {})
                    match_percentage = f_rec.get('match_percentage', 0)
                    matched_on = f_rec.get('match_details', {})
                    
                    # Consolidate matched keywords for display
                    matched_on_parts = []
                    for key, keywords in matched_on.items():
                        if keywords:
                            matched_on_parts.append(f"{', '.join(keywords)}")
                    matched_on_str = "; ".join(matched_on_parts)
                else:
                    f = f_rec
                    match_percentage = f_rec.get('match_strength', {}).get('percentage', 0)
                    matched_on = f_rec.get('match_strength', {}).get('matched_on', [])
                    matched_on_str = ', '.join(matched_on)
                
                star_rating = get_star_rating(match_percentage)
                
                hourly_rate = f.get('hourly_rate_usd', 'N/A')
                if hourly_rate != 'N/A':
                    hourly_rate = f"${hourly_rate}/hr"
                    
                st.sidebar.markdown(f"""
                <div class="freelancer-card-sidebar">
                    <h4>{f.get('name', 'N/A')}</h4>
                    <div class="title">{f.get('title', 'N/A')}</div>
                    <div class="details">
                        <b>Match:</b> {match_percentage:.0f}% {star_rating}<br>
                        <b>Rate:</b> {hourly_rate}<br>
                        <b>Specialties:</b> {', '.join(f.get('specialties', ['N/A']))}
                    </div>
                    <div class="matched-on">
                        <i>Matched on: {matched_on_str}</i>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.sidebar.info("No freelancer matches yet. Ask a question to get recommendations.")

# Function to extract match percentage from any freelancer format
def get_match_percentage(f_rec):
    if isinstance(f_rec, dict):
        if 'match_percentage' in f_rec: # New structure from RAG tool
            return f_rec.get('match_percentage', 0)
        elif 'match_strength' in f_rec: # Older structure from sidebar/direct recs
            return f_rec.get('match_strength', {}).get('percentage', 0)
    return 0

# Display chat messages from history
for i, message in enumerate(st.session_state.chat_history):
    with main_content.chat_message(message["role"]):
        assistant_content = message["content"]
        cards_html_parts = []

        # Prepare HTML for query-specific recommendations if available
        if message["role"] == "assistant" and "recommendations" in message and message["recommendations"]:
            query_recs = message["recommendations"]
            freelancers = query_recs.get("freelancers", [])
            articles = query_recs.get("articles", [])
            # kb_chunks = query_recs.get("kb_chunks", []) # Not currently displayed as cards

            if freelancers:
                cards_html_parts.append("<br><hr><b>✨ Top Recommended Freelancers:</b>")
                # Function to extract match percentage (defined earlier, ensure it's in scope or redefine if necessary)
                # For simplicity, assuming get_match_percentage is available or defined globally/locally if needed.
                top_freelancers = sorted(freelancers, key=get_match_percentage, reverse=True)[:3]
                for f_rec in top_freelancers:
                    if 'freelancer' in f_rec: # New structure
                        f_data = f_rec.get("freelancer", {})
                        match_percentage = f_rec.get('match_percentage', 0)
                        matched_on_details = f_rec.get('match_details', {})
                        matched_on_parts = [f"{', '.join(kw)}" for k, kw in matched_on_details.items() if kw]
                        matched_on_str = "; ".join(matched_on_parts)
                    else: # Old structure (fallback)
                        f_data = f_rec
                        match_percentage = f_rec.get('match_strength', {}).get('percentage', 0)
                        matched_on_list = f_rec.get('match_strength', {}).get('matched_on', [])
                        matched_on_str = ', '.join(matched_on_list)
                    
                    star_rating = get_star_rating(match_percentage)
                    hourly_rate = f_data.get('hourly_rate_usd', 'N/A')
                    if hourly_rate != 'N/A': hourly_rate = f"${hourly_rate}/hr"

                    cards_html_parts.append(f"""
                    <div class="freelancer-card-chat">
                        <h4>{html.escape(str(f_data.get('name', 'N/A')))}</h4>
                        <div class="title">{html.escape(str(f_data.get('title', 'N/A')))}</div>
                        <div class="details">
                            <b>Match:</b> {match_percentage:.0f}% {star_rating}<br>
                            <b>Rate:</b> {html.escape(str(hourly_rate))}<br>
                            <b>Specialties:</b> {html.escape(str(', '.join(f_data.get('specialties', ['N/A']))))}
                        </div>
                        <div class="matched-on">
                            <i>Matched on: {html.escape(str(matched_on_str))}</i>
                        </div>
                    </div>
                    """)
            
            if articles:
                cards_html_parts.append("<br><hr><b>📚 Recommended Articles:</b>")
                for article_rec in articles[:3]: # Limit to top 3 articles as well
                    article_name = article_rec.get('article_name', 'N/A').replace('_', ' ').replace('.md', '').title()
                    article_snippet = article_rec.get('snippet', 'Click to read more.')
                    # Simplistic article card, can be enhanced with CSS class .article-card-chat
                    cards_html_parts.append(f"""
                    <div class="freelancer-card-chat"> 
                        <h4>📄 {html.escape(str(article_name))}</h4>
                        <p style='font-size:0.9em;'>{html.escape(str(article_snippet[:150]))}...</p>
                    </div>
                    """)

        main_content.markdown(assistant_content, unsafe_allow_html=True) # Render main agent text

        if cards_html_parts: # If there are cards to display, render each part
            for part_html in cards_html_parts:
                 main_content.markdown(part_html, unsafe_allow_html=True)

        # Display sources if available (can be an expander within the message or separate)
        if message["role"] == "assistant" and message.get("sources"):
            with main_content.expander("View Sources"):
                for source in message["sources"]:
                    main_content.markdown(f"- {source['name']} (Snippet: _{source.get('snippet', 'N/A')}_)")

        # Feedback buttons
        if message["role"] == "assistant":
            feedback_key_suffix = f"_feedback_{i}"
            cols = main_content.columns(10)
            if cols[0].button("👍", key=f"positive{feedback_key_suffix}", help="Good response"):
                handle_feedback(i, "positive")
            if cols[1].button("👎", key=f"negative{feedback_key_suffix}", help="Bad response"):
                handle_feedback(i, "negative")
            # Add feedback buttons for assistant messages (outside of any form)
            feedback_col1, feedback_col2, spacer = st.columns([1,1,8])
            with feedback_col1:
                st.button("👍", key=f"chat_feedback_positive_{i}", on_click=handle_feedback, args=(i, "positive"))
            with feedback_col2:
                st.button("👎", key=f"chat_feedback_negative_{i}", on_click=handle_feedback, args=(i, "negative"))

# Place the form for user input at the bottom
with st.form(key="user_input_form"):
    st.text_input(
        "Ask me anything about freelance projects and the Makers platform!", 
        placeholder="e.g., How do I get paid for a fixed-price project?", 
        key="user_query_input"
    )
    st.form_submit_button(label="Ask", on_click=handle_submit)
