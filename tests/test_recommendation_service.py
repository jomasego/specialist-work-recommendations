import pytest
from backend.recommendation_service import RecommendationService

# Sample data for testing
SAMPLE_FREELANCERS = [
    {
        "id": "f1",
        "name": "Alex Doe",
        "title": "Senior Python Developer",
        "bio": "Expert in Python, machine learning, and backend systems.",
        "specialties": ["Backend Development", "AI/ML"],
        "skills": [
            {"category": "Programming Languages", "name": "Python", "proficiency": 5},
            {"category": "Frameworks", "name": "Flask", "proficiency": 4}
        ]
    },
    {
        "id": "f2",
        "name": "Jane Smith",
        "title": "Frontend React Specialist",
        "bio": "Loves building beautiful UIs with React and modern JavaScript.",
        "specialties": ["Frontend Development"],
        "skills": [
            {"category": "Programming Languages", "name": "JavaScript", "proficiency": 5},
            {"category": "Frameworks", "name": "React", "proficiency": 5}
        ]
    }
]

@pytest.fixture
def recommendation_service():
    """Provides a RecommendationService instance with sample data."""
    service = RecommendationService()
    # Override the real freelancer data with our sample data for isolated testing
    service.freelancers = SAMPLE_FREELANCERS
    return service

def test_calculate_match_strength_perfect_match(recommendation_service):
    """Tests match calculation when the query perfectly matches a skill."""
    query = "I need a python developer"
    freelancer = SAMPLE_FREELANCERS[0]
    match_strength = recommendation_service._calculate_match_strength(freelancer, query)

    assert match_strength['percentage'] > 0
    assert "python" in match_strength['matched_on']
    assert "developer" in match_strength['matched_on']

def test_calculate_match_strength_no_match(recommendation_service):
    """Tests match calculation when the query has no matching terms."""
    query = "Looking for a graphic designer"
    freelancer = SAMPLE_FREELANCERS[0]
    match_strength = recommendation_service._calculate_match_strength(freelancer, query)

    assert match_strength['percentage'] == 0
    assert len(match_strength['matched_on']) == 0

def test_get_recommendations_returns_sorted_freelancers(recommendation_service):
    """Tests that get_recommendations returns freelancers sorted by match strength."""
    chat_history = [
        {"role": "user", "content": "I need a senior python backend developer"}
    ]
    recommendations = recommendation_service.get_recommendations(chat_history)
    freelancers = recommendations['freelancers']

    assert len(freelancers) > 0
    # Alex Doe (Python Developer) should be the first recommendation
    assert freelancers[0]['id'] == 'f1'
    assert freelancers[0]['match_strength']['percentage'] > freelancers[1]['match_strength']['percentage']

def test_recommendation_structure(recommendation_service):
    """Tests that the recommendation output has the correct structure."""
    chat_history = [{"role": "user", "content": "react"}]
    recommendations = recommendation_service.get_recommendations(chat_history)
    freelancers = recommendations['freelancers']

    assert "freelancers" in recommendations
    assert "articles" in recommendations
    assert len(freelancers) > 0

    # Check the structure of the first freelancer
    freelancer = freelancers[0]
    assert "id" in freelancer
    assert "name" in freelancer
    assert "title" in freelancer
    assert "match_strength" in freelancer
    assert "percentage" in freelancer['match_strength']
    assert "matched_on" in freelancer['match_strength']
