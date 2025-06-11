from flask import Flask, request, jsonify
from rag_service import RAGService

app = Flask(__name__)

# Initialize the RAG service once when the app starts
try:
    rag_service = RAGService()
except Exception as e:
    # If the service fails to initialize (e.g., missing files), log the error.
    # The app will run but the /query endpoint will fail gracefully.
    rag_service = None
    print(f"CRITICAL: Failed to initialize RAGService: {e}")

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
        return jsonify({"error": "An internal error occurred."}), 500

if __name__ == '__main__':
    # Use a different port to avoid conflicts with Streamlit if run on the same machine
    app.run(host='0.0.0.0', port=5001, debug=True)

