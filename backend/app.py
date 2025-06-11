import logging
import os
from logging.handlers import RotatingFileHandler
from flask import Flask, request, jsonify
from rag_service import RAGService, RAGServiceError
from recommendation_service import RecommendationService
from flask_cors import CORS

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

# Initialize services
try:
    rag_service = RAGService() # RAGService itself should use app.logger if needed, or pass logger
    app.logger.info("RAGService initialized successfully.")
except RAGServiceError as e:
    rag_service = None
    app.logger.critical(f"RAGService failed to initialize: {e}", exc_info=True)

try:
    recommendation_service = RecommendationService() # RecommendationService itself should use app.logger or pass logger
    app.logger.info("RecommendationService initialized successfully.")
except Exception as e: # Generic exception for now
    recommendation_service = None
    app.logger.critical(f"RecommendationService failed to initialize: {e}", exc_info=True)

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
def handle_chat():
    app.logger.info(f"POST /chat received from {request.remote_addr}")
    data = request.get_json()
    if not data or 'query' not in data:
        app.logger.warning("'/query' field missing from /chat request.")
        return jsonify({"error": "'query' field is missing"}), 400

    current_query = data['query']
    chat_history = data.get('chat_history', [])
    app.logger.info(f"Query: '{current_query}', History items: {len(chat_history)}")

    # --- Intent Detection --- #
    intent = detect_intent(current_query)

    if intent == 'recommendation':
        # For a recommendation request, we don't need to call the RAG service.
        # The frontend will independently call the /recommendations endpoint.
        # We just need to provide a friendly, non-RAG response here.
        response_text = "Of course! I'm updating the recommendations based on your request. Take a look at the specialists I've found for you."
        app.logger.info("Responding with canned recommendation message.")
        return jsonify({"answer": response_text, "sources": []})

    # --- RAG Service for Question Answering --- #
    if not rag_service:
        app.logger.error("RAG service not available for /chat endpoint.")
        return jsonify({"error": "The RAG service is not available."}), 503
    
    try:
        result = rag_service.answer_query(current_query=current_query, chat_history=chat_history)
        app.logger.info(f"Successfully processed /chat (RAG) for: '{current_query}'.")
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"Error processing /chat (RAG) for '{current_query}': {e}", exc_info=True)
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.route('/recommendations', methods=['POST'])
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
        # The recommendation service will now extract context from the whole chat history
        recommendations = recommendation_service.get_recommendations(chat_history=chat_history)
        app.logger.info(f"Successfully generated {len(recommendations.get('freelancers', []))} freelancer and {len(recommendations.get('articles', []))} article recommendations.")
        return jsonify(recommendations)
    except Exception as e:
        app.logger.error(f"Error generating recommendations: {e}", exc_info=True)
        return jsonify({"error": "Failed to generate recommendations"}), 500

if __name__ == '__main__':
    # When running with 'python app.py', debug=True is fine for Flask's reloader.
    # For production, use a proper WSGI server like Gunicorn, which will handle logging differently.
    app.logger.info('Starting Flask development server...') # This log might appear twice if reloader is active
    app.run(host='0.0.0.0', port=5001, debug=True)
