from flask import Flask, request, jsonify
from rag_service import RAGService, RAGServiceError
from recommendation_service import RecommendationService
from flask_cors import CORS

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Initialize services
try:
    rag_service = RAGService()
    print("RAGService initialized successfully.")
except RAGServiceError as e:
    rag_service = None
    print(f"CRITICAL: RAGService failed to initialize: {e}")

try:
    recommendation_service = RecommendationService()
    print("RecommendationService initialized successfully.")
except Exception as e: # Generic exception for now
    recommendation_service = None
    print(f"CRITICAL: RecommendationService failed to initialize: {e}")

@app.route('/')
def home():
    return "Backend for Specialist Work Recommendations AI is running!"

@app.route('/query', methods=['POST'])
def handle_query():
    if not rag_service:
        return jsonify({"error": "The RAG service is not available. Please check server logs."}), 503

    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "'query' field is missing from request body."}), 400

    query = data['query']
    
    try:
        result = rag_service.answer_query(query)
        return jsonify(result)
    except Exception as e:
        print(f"An error occurred while processing the query: {e}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.route('/recommendations', methods=['POST'])
def get_recommendations_endpoint():
    if not recommendation_service:
        return jsonify({"error": "Recommendation service is not available"}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing JSON payload"}), 400

    query_history = data.get('query_history', [])
    current_query = data.get('current_query')

    if not isinstance(query_history, list):
        return jsonify({"error": "'query_history' must be a list"}), 400
    if current_query and not isinstance(current_query, str):
        return jsonify({"error": "'current_query' must be a string"}), 400

    try:
        recommendations = recommendation_service.get_recommendations(query_history, current_query)
        return jsonify(recommendations), 200
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        return jsonify({"error": "Internal server error while fetching recommendations", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
