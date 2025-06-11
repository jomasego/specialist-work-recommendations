import pytest
from unittest.mock import MagicMock, patch
from backend.rag_service import RAGService, RAGServiceError
import numpy as np

# Mock data for testing
SAMPLE_DOCS = {
    "doc1.md": "This is a test document about Python.",
    "doc2.md": "This document discusses frontend development with React."
}

@pytest.fixture
def mock_env(mocker):
    """Mocks environment variables and external dependencies."""
    # Mock environment variables
    mocker.patch('os.getenv', side_effect=lambda key, default=None: {
        'GEMINI_API_KEY': 'fake_gemini_key',
        'GROQ_API_KEY': 'fake_groq_key'
    }.get(key, default))

    # Mock API clients
    mocker.patch('google.generativeai.GenerativeModel', return_value=MagicMock())
    mocker.patch('groq.Groq', return_value=MagicMock())

    # Mock file system and FAISS
    mocker.patch('os.path.exists', return_value=True)
    mocker.patch('faiss.read_index', return_value=MagicMock())
    mocker.patch('builtins.open', mocker.mock_open(read_data='{}'))
    mocker.patch('pickle.load', return_value={})

@pytest.fixture
def rag_service(mock_env):
    """Provides a RAGService instance with mocked dependencies."""
    with patch.object(RAGService, '_load_or_create_index', return_value=None) as mock_load:
        service = RAGService()
        # Manually set attributes that would be created in _load_or_create_index
        service.index = MagicMock()
        service.documents = list(SAMPLE_DOCS.values())
        service.doc_map = {i: name for i, name in enumerate(SAMPLE_DOCS.keys())}
        service.embed_model = MagicMock()
        service.embed_model.embed_query.return_value = np.random.rand(768).tolist()
        return service

def test_find_relevant_documents_found(rag_service, mocker):
    """Test that relevant documents are returned when the index search is successful."""
    # Arrange
    mock_index = rag_service.index
    # Simulate FAISS returning distances and indices of the top 2 matches
    mock_index.search.return_value = (np.array([[0.1, 0.2]]), np.array([[0, 1]]))
    
    # Act
    context, sources = rag_service._find_relevant_documents("test query")

    # Assert
    assert len(sources) == 2
    assert "doc1.md" in sources
    assert "doc2.md" in sources
    assert "This is a test document about Python." in context
    assert "frontend development with React" in context

def test_find_relevant_documents_not_found(rag_service):
    """Test that an empty context is returned when no documents are found."""
    # Arrange
    mock_index = rag_service.index
    # Simulate FAISS returning no matches (indices are -1)
    mock_index.search.return_value = (np.array([[]]), np.array([[-1]]))

    # Act
    context, sources = rag_service._find_relevant_documents("unrelated query")

    # Assert
    assert context == ""
    assert sources == []

def test_answer_query_gemini_call(rag_service, mocker):
    """Test that the Gemini model is called correctly."""
    # Arrange
    rag_service._find_relevant_documents = MagicMock(return_value=("Some context", ["doc1.md"]))
    mock_gemini_model = rag_service.gemini_client
    mock_gemini_model.generate_content.return_value.text = "Gemini Response"

    # Act
    result = rag_service.answer_query("test query", model='gemini-1.5-flash-latest')

    # Assert
    mock_gemini_model.generate_content.assert_called_once()
    assert result['answer'] == "Gemini Response"
    assert result['sources'] == ["doc1.md"]

def test_answer_query_groq_call(rag_service, mocker):
    """Test that the Groq Llama3 model is called correctly."""
    # Arrange
    rag_service._find_relevant_documents = MagicMock(return_value=("Some context", ["doc1.md"]))
    mock_groq_model = rag_service.groq_client.chat.completions
    # Mock the response structure for Groq
    mock_groq_response = MagicMock()
    mock_groq_response.choices = [MagicMock()]
    mock_groq_response.choices[0].message.content = "Groq Response"
    mock_groq_model.create.return_value = mock_groq_response

    # Act
    result = rag_service.answer_query("test query", model='llama3-8b-8192')

    # Assert
    mock_groq_model.create.assert_called_once()
    assert result['answer'] == "Groq Response"
    assert result['sources'] == ["doc1.md"]
