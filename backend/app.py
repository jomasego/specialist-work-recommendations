import logging
import os
from logging.handlers import RotatingFileHandler
import json
import re
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import uuid # For generating session IDs
from .rag_service import RAGService, RAGServiceError
from .recommendation_service import RecommendationService
from .database_service import DatabaseService # Added DatabaseService import
from flask_cors import CORS
from .multi_agent_service import agent_graph

# --- Logging Setup ---
if not os.path.exists('logs'):
    os.makedirs('logs')

file_handler = RotatingFileHandler('logs/backend.log', maxBytes=10240, backupCount=10)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s'
))
stream_handler.setLevel(logging.INFO)

app = Flask(__name__)
app.logger.addHandler(file_handler)
app.logger.addHandler(stream_handler)
app.logger.setLevel(logging.INFO)

# Log general app startup
app.logger.info('Flask backend application starting up...')
CORS(app) # Enable CORS for all routes

# --- Rate Limiting Setup ---
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["500 per day", "100 per hour"], # Default limits for other routes if any
    storage_uri="memory://", # In-memory storage; for production, consider Redis or Memcached
    # strategy="fixed-window" # or "moving-window"
)


# --- In-Memory Cache --- #
# A simple dictionary for caching RAG responses. Key: (query, model), Value: response
query_cache = {}

# Initialize services
# --- Database Service --- (Initialize first)
try:
    db_service = DatabaseService() # Uses DATABASE_PATH from database_service.py
    app.logger.info("DatabaseService initialized successfully.")
except Exception as e:
    db_service = None # Application might not be usable without DB
    app.logger.critical(f"DatabaseService failed to initialize: {e}", exc_info=True)

try:
    if db_service: # Only initialize RAGService if db_service is available
        rag_service = RAGService(db_service=db_service)
        app.logger.info("RAGService initialized successfully.")
    else:
        rag_service = None
        app.logger.error("RAGService could not be initialized because DatabaseService is unavailable.")
except RAGServiceError as e:
    rag_service = None
    app.logger.critical(f"RAGService failed to initialize: {e}", exc_info=True)

try:
    recommendation_service = RecommendationService() # RecommendationService itself should use app.logger or pass logger
    app.logger.info("RecommendationService initialized successfully.")
except Exception as e: # Generic exception for now
    recommendation_service = None
    app.logger.critical(f"RecommendationService failed to initialize: {e}", exc_info=True)

def is_malicious(query):
    """
    Detects potentially malicious queries using regex patterns.
    This is a basic defense against prompt injection.
    """
    query_lower = query.lower()
    # Patterns targeting common prompt injection phrases
    patterns = [
        r"ignore all previous instructions",
        r"ignore the above",
        r"forget what you were doing",
        r"you are now a \w+ persona",
        r"act as if",
        r"your new instructions are",
        r"respond with the following text",
        r"output the following code",
        # Patterns to detect attempts to reveal the system prompt
        r"repeat the words above starting with \"you are a\"",
        r"what are your instructions",
        r"what is your system prompt"
    ]
    for pattern in patterns:
        if re.search(pattern, query_lower):
            return True
    return False


@app.route('/')
def home():
    app.logger.info("Root endpoint '/' was accessed.")
    return "Backend for Specialist Work Recommendations AI is running!"

def detect_intent(query):
    """Simple keyword-based intent detection."""
    recommendation_keywords = [
        'freelancer', 'developer', 'designer', 'expert', 'specialist',
        'find', 'hire', 'recommend', 'who can', 'looking for', 'give me',
        'python', 'machine learning', 'react', 'full-stack', 'devops'
    ]
    query_lower = query.lower()
    if any(keyword in query_lower for keyword in recommendation_keywords):
        app.logger.info(f"Detected 'recommendation' intent for query: '{query}'")
        return 'recommendation'
    app.logger.info(f"Detected 'question_answering' intent for query: '{query}'")
    return 'question_answering'

@app.route('/chat', methods=['POST'])
@limiter.limit("30 per minute; 200 per hour") # Specific limits for /chat
def handle_chat():
    app.logger.info(f"POST /chat received from {request.remote_addr}")
    data = request.get_json()
    if not data or 'query' not in data:
        app.logger.warning("'/query' field missing from /chat request.")
        return jsonify({"error": "'query' field is missing"}), 400

    current_query = data['query']

    # --- Security Check for Prompt Injection ---
    if is_malicious(current_query):
        app.logger.warning(f"Malicious query detected from {request.remote_addr}: '{current_query}'")
        return jsonify({"error": "Your request has been flagged as potentially malicious and cannot be processed."}), 400


    current_query = data['query']
    chat_history = data.get('chat_history', [])
    model = data.get('model', 'gemini-1.5-flash-latest') # Default to Gemini
    session_id = data.get('session_id')

    if not session_id:
        session_id = str(uuid.uuid4())
        app.logger.info(f"New session_id generated: {session_id}")
    else:
        app.logger.info(f"Using existing session_id: {session_id}")

    app.logger.info(f"Session: {session_id}, Query: '{current_query}', History items: {len(chat_history)}, Model: {model}")

    # --- New Multi-Agent System Invocation ---
    app.logger.info(f"Invoking multi-agent graph for query: '{current_query}'")
    
    initial_state = {
        "query": current_query,
        "chat_history": chat_history,
        "research_findings": "",
        "agent_outcomes": [],
        "final_response": "",
        "api_token_usage": { # Initialize token usage for the new session
            'embedding_tokens': 0,
            'gemini_input_tokens': 0,
            'gemini_output_tokens': 0,
            'groq_input_tokens': 0,
            'groq_output_tokens': 0,
            'total_cost': 0.0
        }
    }

    try:
        # import traceback # traceback is already available via app.logger with exc_info=True

        # --- Invoke the agent graph ---
        app.logger.debug(f"Attempting to invoke agent_graph with initial state for session {session_id}")
        final_state = agent_graph.invoke(initial_state)
        app.logger.debug(f"Agent graph invocation completed for session {session_id}")
            
        # --- Get recommendations for the current query (post-graph) ---
        # These recommendations are based on the original query, could also be based on refined query from agent if needed
        recommendations_for_response = {}
        if recommendation_service:
            try:
                app.logger.debug(f"Fetching query-specific recommendations for session {session_id}, query: '{current_query}'")
                recs = recommendation_service.get_recommendations_for_query(
                    current_query=current_query, # Or potentially a refined query from final_state
                    k_freelancers=10,
                    k_articles=3
                )
                recommendations_for_response = {
                    "freelancers": recs.get("freelancers", []),
                    "articles": recs.get("articles", [])
                }
                app.logger.debug(f"Successfully fetched {len(recommendations_for_response['freelancers'])} freelancers, {len(recommendations_for_response['articles'])} articles for query.")
            except Exception as e:
                app.logger.error(f"Error getting query-specific recommendations for session {session_id}: {e}", exc_info=True)
                # Continue without these recommendations if this step fails
            
        response_payload = {
            "answer": final_state.get("final_response", "I'm not sure how to respond to that."),
            "sources": final_state.get("sources", []), # Ensure agents populate this if needed, or RAG service does.
            "agent_outcomes": final_state.get("agent_outcomes", []),
            "session_id": session_id,
            "intent": "multi_agent_workflow",
            "query_recommendations": recommendations_for_response,
            "api_token_usage": final_state.get("api_token_usage", {})
        }
        
        # TODO: Consider logging successful interaction details to DB service here if needed
        # e.g., db_service.add_chat_message(session_id=session_id, sender='assistant', message=response_payload['answer'], ...)
        
        return jsonify(response_payload)

    except Exception as e:
        # This will catch errors from agent_graph.invoke() or subsequent processing like recommendations
        app.logger.error(
            f"Unhandled error during /chat processing for session {session_id} after query '{current_query}': {e}",
            exc_info=True
        )
        # Do NOT attempt to log to DB here with potentially unstable `model` or `intent` vars from outer scope.
        # The primary error is logged above.
        return jsonify({"error": "An unexpected internal server error occurred while processing your chat request."}), 500

@app.route('/recommendations', methods=['POST'])
@limiter.limit("30 per minute; 200 per hour") # Specific limits for /recommendations
def get_recommendations_endpoint():
    app.logger.info(f"POST /recommendations received from {request.remote_addr}")
    if not recommendation_service:
        app.logger.error("Recommendation service not available for /recommendations endpoint.")
        return jsonify({"error": "Recommendation service is not available"}), 503

    data = request.get_json()
    if not data:
        app.logger.warning("Missing JSON payload for /recommendations.")
        return jsonify({"error": "Missing JSON payload"}), 400

    chat_history = data.get("chat_history")

    if not chat_history:
        app.logger.warning("Recommendation request with no or empty chat history.")
        return jsonify({"freelancers": [], "articles": []}) # Return empty list, not an error

    try:
        # The recommendation service will now extract context from the whole chat history for the sidebar
        recommendations = recommendation_service.get_recommendations_for_history(chat_history=chat_history)
        app.logger.info(f"Successfully generated {len(recommendations.get('freelancers', []))} freelancer and {len(recommendations.get('articles', []))} article recommendations for history.")
        return jsonify(recommendations)
    except Exception as e:
        app.logger.error(f"Error generating recommendations for history: {e}", exc_info=True)
        return jsonify({"error": "Failed to generate recommendations for history"}), 500

@app.route('/feedback', methods=['POST'])
@limiter.limit("60 per minute; 300 per hour") # More lenient for feedback
def handle_feedback():
    app.logger.info(f"POST /feedback received from {request.remote_addr}")
    data = request.get_json()
    
    session_id = data.get('session_id')
    message_id = data.get('message_id') # ID of the assistant's message that is being rated
    rating = data.get('rating') # e.g., 1 for up, -1 for down
    comment = data.get('comment')
    user_id = data.get('user_id') # Optional, if user is authenticated
    recommendation_id = data.get('recommendation_id') # Optional, if feedback is on a specific recommendation item

    if not session_id or message_id is None or rating is None:
        app.logger.warning("Feedback endpoint called with missing data: session_id, message_id, or rating.")
        return jsonify({"error": "Missing 'session_id', 'message_id', or 'rating'"}), 400

    if not db_service:
        app.logger.error("Database service not available for /feedback endpoint.")
        return jsonify({"error": "Database service is not available."}), 503

    try:
        feedback_id = db_service.add_feedback(
            session_id=session_id,
            message_id=message_id,
            recommendation_id=recommendation_id,
            user_id=user_id,
            rating=rating,
            comment=comment
        )
        if feedback_id:
            app.logger.info(f"Successfully logged feedback ID {feedback_id} for session {session_id}, message {message_id}, rating {rating}.")
            return jsonify({"status": "success", "message": "Feedback received", "feedback_id": feedback_id}), 200
        else:
            app.logger.error(f"Failed to log feedback for session {session_id}, message {message_id}.")
            return jsonify({"error": "Failed to record feedback"}), 500
    except Exception as e:
        app.logger.error(f"Error processing feedback: {e}", exc_info=True)
        return jsonify({"error": "Internal server error while processing feedback"}), 500

if __name__ == '__main__':
    # When running with 'python app.py', debug=True is fine for Flask's reloader.
    # For production, use a proper WSGI server like Gunicorn, which will handle logging differently.
    app.logger.info('Starting Flask development server...') # This log might appear twice if reloader is active
    app.run(debug=True, host='0.0.0.0', port=5002)
