# 🤝 Makers AI Talent-Matching Assistant

Welcome to the Makers AI Assistant! This project is an intelligent, conversational AI designed to help clients find the perfect freelancer for their needs. It combines a powerful question-answering system with a personalized recommendation engine to create a seamless and helpful user experience.

![Makers Logo](https://raw.githubusercontent.com/your-username/your-repo/main/assets/MAKERS-logo.svg) <!-- Replace with the actual path to your logo once pushed -->

## 🌟 Core Features

*   **🤖 Conversational AI Chat:** A friendly and interactive chat interface built with Streamlit.
*   **🧠 Intent Detection:** The assistant intelligently understands whether you're asking a question or looking for a freelancer, routing your request accordingly.
*   **📚 Knowledge Base Q&A:** Utilizes a Retrieval-Augmented Generation (RAG) service to answer questions about the platform using a library of documents.
*   **✨ Personalized Freelancer Recommendations:** Recommends the best freelancers for the job based on the conversation history, scoring them on skills, specialties, and more.
*   **📊 Dynamic UI:** The recommendation panel updates in real-time as the conversation evolves.
*   **👍 Feedback Mechanism:** Users can provide feedback on the assistant's answers.

## 🛠️ Tech Stack

*   **Frontend:** [Streamlit](https://streamlit.io/) - For the interactive web application.
*   **Backend:** [Flask](https://flask.palletsprojects.com/) - For the robust API server.
*   **AI & Machine Learning:**
    *   [Google Gemini](https://ai.google.dev/) - For language model and embeddings.
    *   [FAISS](https://github.com/facebookresearch/faiss) - For efficient similarity search in our vector database.
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
    *   Create a file named `.env` in the root directory.
    *   Add your Google Gemini API key to it:
        ```
        GEMINI_API_KEY="YOUR_API_KEY_HERE"
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

## ⚙️ How It Works

1.  **Frontend (`frontend/app.py`):** The Streamlit app captures user input.
2.  **API Call:** It sends the user's query and the chat history to the backend `/chat` endpoint.
3.  **Backend (`backend/app.py`):**
    *   The Flask server receives the request.
    *   **Intent Detection:** It analyzes the query to determine if it's a `question_answering` or `recommendation` intent.
    *   **Routing:**
        *   For questions, it calls the `RAGService` to find relevant documents and generate an answer.
        *   For freelancer requests, it returns a simple confirmation message.
4.  **Recommendation Update:** The frontend receives the chat response and immediately makes a separate call to the `/recommendations` endpoint. This endpoint uses the `RecommendationService` to score and rank freelancers based on the full chat history, returning the top matches to be displayed in the sidebar.

## 🔌 API Endpoints

*   `POST /chat`: The main endpoint for conversational interactions. Handles intent detection.
*   `POST /recommendations`: The endpoint dedicated to fetching updated freelancer and article recommendations.

## 🔮 Future Improvements

*   **Advanced Intent Detection:** Replace the keyword-based system with a more robust NLP model.
*   **Database Integration:** Store chat history, user feedback, and freelancer profiles in a persistent database (e.g., SQLite or PostgreSQL).
*   **Scalability:** Deploy the application using a production-ready WSGI server like Gunicorn.
*   **Enhanced Filtering:** Allow users to filter and sort recommendations on the frontend.
