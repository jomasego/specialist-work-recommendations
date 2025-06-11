import pytest
import json
from backend.app import app as flask_app

@pytest.fixture
def app():
    """Create and configure a new app instance for each test."""
    # Flask's test client provides a simple interface to the application.
    # No need to run a live server.
    flask_app.config.update({
        "TESTING": True,
    })
    yield flask_app

@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()

def test_chat_endpoint_question_answering(client, mocker):
    """Test the /chat endpoint for a standard Q&A query."""
    # Mock the RAG service to prevent actual API calls and file access
    mock_rag_answer = {
        "answer": "This is a mocked answer from the RAG service.",
        "sources": ["doc1.md"]
    }
    mocker.patch('backend.app.rag_service.answer_query', return_value=mock_rag_answer)

    # The payload mimics a request from the frontend
    payload = {
        "query": "What is a test?",
        "chat_history": [],
        "model": "gemini-1.5-flash-latest"
    }

    # Make the POST request to the /chat endpoint
    response = client.post('/chat', data=json.dumps(payload), content_type='application/json')

    # Assertions
    assert response.status_code == 200
    response_data = response.get_json()
    assert response_data['answer'] == mock_rag_answer['answer']
    assert response_data['sources'] == mock_rag_answer['sources']
    # Ensure the RAG service's answer_query method was called once with the correct query
    from backend.app import rag_service
    rag_service.answer_query.assert_called_once_with(
        current_query=payload['query'], 
        chat_history=payload['chat_history'], 
        model=payload['model']
    )

def test_recommendations_endpoint(client, mocker):
    """Test the /recommendations endpoint."""
    # Mock the recommendation service
    mock_recs = {
        "freelancers": [{"name": "Jane Doe", "title": "React Developer"}],
        "articles": [{"title": "Intro to React", "url": "http://example.com"}]
    }
    mocker.patch('backend.app.recommendation_service.get_recommendations', return_value=mock_recs)

    payload = {
        "chat_history": [{"role": "user", "content": "I need a React developer"}]
    }

    response = client.post('/recommendations', data=json.dumps(payload), content_type='application/json')

    assert response.status_code == 200
    response_data = response.get_json()
    assert response_data['freelancers'][0]['name'] == "Jane Doe"
    assert len(response_data['articles']) == 1
    from backend.app import recommendation_service
    recommendation_service.get_recommendations.assert_called_once_with(chat_history=payload['chat_history'])
