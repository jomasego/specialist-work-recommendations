# 🤝 Makers AI Talent-Matching Assistant

Welcome to the Makers AI Assistant! This project is an intelligent, conversational AI designed to help clients find the perfect freelancer for their needs. It combines a powerful question-answering system with a personalized recommendation engine to create a seamless and helpful user experience.

![Makers Logo](./frontend/MAKERS-logo.svg)

## 🌟 Core Features

*   **🤖 Conversational AI Chat:** A friendly and interactive chat interface built with Streamlit.
*   **⚡ Dynamic Model Selection:** Choose between Google's powerful `Gemini 1.5 Flash` for deep reasoning or Groq's blazing-fast `Llama3 8B` for near-instant responses.
*   **🧠 Intent Detection:** The assistant intelligently understands whether you're asking a question or looking for a freelancer.
*   **📚 Knowledge Base Q&A:** Utilizes a Retrieval-Augmented Generation (RAG) service to answer questions about the platform using a library of documents.
*   **✨ Personalized Freelancer Recommendations:** Recommends the best freelancers for the job based on the conversation history, scoring them on skills, specialties, and more.
*   **📊 Dynamic UI:** The recommendation panel updates in real-time as the conversation evolves.
*   **👍 Feedback Mechanism:** Users can provide direct feedback (👍/👎) on the assistant's answers, which is logged for future analysis.
*   **✅ Comprehensive Testing:** Includes a full suite of unit and integration tests to ensure reliability.
*   **⚡ Optimized Performance:** Implements smart caching for the `/chat` endpoint, significantly speeding up responses to repeated queries.
*   **🛡️ Enhanced Security:**
    *   **Prompt Injection Defense:** Incoming queries are scanned for common prompt injection patterns and rejected if detected.
    *   **Inappropriate Content Filtering:** Utilizes Gemini's built-in safety settings and Groq's `Llama-Guard-4` model to filter harmful or inappropriate content.

## 🛠️ Tech Stack

*   **Frontend:** [Streamlit](https://streamlit.io/) - For the interactive web application.
*   **Backend:** [Flask](https://flask.palletsprojects.com/) - For the robust API server.
*   **AI & Machine Learning:**
    *   **LLMs:** [Google Gemini](https://ai.google.dev/), [Groq Llama3](https://groq.com/)
    *   **Embeddings:** [Google `text-embedding-004`](https://ai.google.dev/edge/docs/embedding/get_text_embeddings)
    *   **Vector Search:** [FAISS](https://github.com/facebookresearch/faiss) - For efficient similarity search.
*   **Testing:** [Pytest](https://pytest.org/), [Pytest-Mock](https://pypi.org/project/pytest-mock/)
*   **Programming Language:** Python 3.10+

## 🚀 Getting Started

Follow these steps to get the project running on your local machine.

### Prerequisites

*   Python 3.10 or higher
*   `pip` and `venv` for package management

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/personalized-recommendation-capabilities.git
    cd personalized-recommendation-capabilities
    ```

2.  **Set up the virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your environment variables:**
    *   Create a file named `.env` by copying the example: `cp .env.example .env`
    *   Add your API keys to the `.env` file:
        ```dotenv
        GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"
        GROQ_API_KEY="YOUR_GROQ_API_KEY_HERE"
        ```

### Running the Application

1.  **Start the backend server:**
    ```bash
    python backend/app.py
    ```

2.  **Start the frontend application (in a new terminal):**
    ```bash
    streamlit run frontend/app.py
    ```

### ✅ Running Tests

To ensure everything is working correctly, run the test suite:
```bash
source .venv/bin/activate
pytest
```

## ⚙️ How It Works

1.  **Frontend (`frontend/app.py`):** The Streamlit app captures user input and the selected AI model.
2.  **API Call:** It sends the query, chat history, and chosen model to the backend `/chat` endpoint.
3.  **Backend (`backend/app.py`):**
    *   The Flask server receives the request.
    *   **Intent Detection:** It analyzes the query to determine if it's a `question_answering` or `recommendation` intent.
    *   **Routing:**
        *   For questions, it calls the `RAGService` to find relevant documents and generate an answer using the selected model (Gemini or Groq).
        *   For freelancer requests, it returns a confirmation message, triggering the frontend to fetch recommendations.
4.  **Recommendation Update:** The frontend makes a separate call to the `/recommendations` endpoint. This uses the `RecommendationService` to score and rank freelancers based on the full chat history.
5.  **Security & Optimization Pre-checks (for `/chat`):**
    *   **Caching:** The system first checks an in-memory cache. If the same query (and model) was processed recently, the cached response is returned instantly.
    *   **Prompt Injection Scan:** If not cached, the query is scanned for malicious patterns. If flagged, it's rejected.
    *   **Content Moderation (Groq):** If using a Groq model, the query is then sent to `Llama-Guard-4` for a safety check. If deemed unsafe, it's rejected.
6.  **RAG Service & LLM Interaction:**
    *   For questions, the `RAGService` retrieves relevant documents.
    *   It then calls the selected LLM (Gemini or Groq) to generate an answer, incorporating context and chat history.
    *   **Content Moderation (Gemini):** Gemini's API has safety settings configured to block harmful responses. If a response is blocked, a generic safety message is returned.
7.  **Recommendation Update:** The frontend makes a separate call to the `/recommendations` endpoint. This uses the `RecommendationService` to score and rank freelancers based on the full chat history.
8.  **Feedback Loop:** User feedback is sent to the `/feedback` endpoint and logged as structured JSON for future analysis.

## 🔌 API Endpoints

*   `POST /chat`: Main endpoint for conversational interactions. Handles intent detection and model selection.
*   `POST /recommendations`: Fetches updated freelancer and article recommendations.
*   `POST /feedback`: Logs user feedback on assistant responses.

## 🔮 Future Improvements

*   **Advanced Interaction:** Implement sentiment detection for adaptive responses.
*   **Robustness & Security:** Implement rate limiting and explore more advanced security measures.
*   **Monitoring & Analytics:** Integrate comprehensive tracking for latency, cost, API call metrics, and user engagement.
*   **Database Integration:** Migrate from JSON files to a robust database (e.g., SQLite, PostgreSQL) for persistent storage of freelancer profiles, chat history, and feedback.
*   **Automated Evaluation:** Develop scripts for automated evaluation against ground truth datasets to measure accuracy and relevance.
*   **Advanced Prompting:** Experiment with techniques like few-shot prompting and chain-of-thought for more complex reasoning.
*   **Context Compression:** Implement strategies to compress context for LLM API calls, optimizing for cost and token limits.
*   **Scalability:** Deploy the application using a production-ready WSGI server like Gunicorn and explore containerization (e.g., Docker).
