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
5.  **Feedback Loop:** User feedback is sent to the `/feedback` endpoint and logged as structured JSON for future analysis.

## 🔌 API Endpoints

*   `POST /chat`: Main endpoint for conversational interactions. Handles intent detection and model selection.
*   `POST /recommendations`: Fetches updated freelancer and article recommendations.
*   `POST /feedback`: Logs user feedback on assistant responses.

## 🔮 Future Improvements

*   **Advanced Interaction:** Implement sentiment detection and smart caching.
*   **Robustness & Security:** Add malicious query detection and rate limiting.
*   **Monitoring:** Track latency, cost, and API call metrics.
*   **Database Integration:** Store chat history, user feedback, and freelancer profiles in a persistent database (e.g., SQLite or PostgreSQL).
*   **Scalability:** Deploy the application using a production-ready WSGI server like Gunicorn.
