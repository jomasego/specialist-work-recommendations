import json
import os
import re
import logging

# Constants
FREELANCER_DB_PATH = "data/freelancer_database.json"
KNOWLEDGE_BASE_DIR = "data/knowledge_base"
MAX_EXPECTED_SCORE = 25 # Used for calculating match percentage

class RecommendationService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing RecommendationService...")
        self.freelancers = self._load_freelancer_database()
        self.knowledge_base_articles = self._list_kb_articles()
        # Simple keyword lists for matching
        self.platform_keywords = [
            'payment', 'dispute', 'hire', 'find talent', 'post job', 
            'profile', 'account', 'getting started', 'best practice'
        ]
        self.logger.info("RecommendationService initialized successfully.")

    def _load_freelancer_database(self):
        if not os.path.exists(FREELANCER_DB_PATH):
            self.logger.warning(f"Freelancer database not found at {FREELANCER_DB_PATH}. Returning empty list.")
            return []
        try:
            with open(FREELANCER_DB_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading freelancer database from {FREELANCER_DB_PATH}: {e}", exc_info=True)
            return []

    def _list_kb_articles(self):
        if not os.path.isdir(KNOWLEDGE_BASE_DIR):
            self.logger.warning(f"Knowledge base directory not found at {KNOWLEDGE_BASE_DIR}. Returning empty list.")
            return []
        try:
            return [f for f in os.listdir(KNOWLEDGE_BASE_DIR) if f.endswith('.md')]
        except Exception as e:
            self.logger.error(f"Error listing knowledge base articles from {KNOWLEDGE_BASE_DIR}: {e}", exc_info=True)
            return []

    def _extract_keywords(self, text):
        """Extracts simple keywords from text."""
        text = text.lower()
        # Remove punctuation and split into words
        words = re.findall(r'\b\w+\b', text)
                # Simple stop word list
        stop_words = set(['a', 'an', 'the', 'in', 'on', 'at', 'for', 'to', 'of', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'is', 'are', 'was', 'were', 'and', 'or', 'but', 'if', 'so', 'that', 'about', 'with', 'from', 'into', 'during', 'including', 'unless', 'while', 'as', 'until', 'up', 'down', 'out', 'through', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'])
        return set(word for word in words if word not in stop_words)

    def get_recommendations(self, chat_history, k_freelancers=3, k_articles=2):
        """
        Generates recommendations based on the full conversation history.
        chat_history: A list of message objects, each with 'role' and 'content'.
        """
        self.logger.info(f"Generating recommendations based on chat history of length {len(chat_history)}.")
        freelancer_recs = []
        article_recs = []

        if not chat_history:
            return {"freelancers": [], "articles": []}

        # Combine content from both 'user' and 'assistant' for a richer context
        full_conversation_text = " ".join([msg['content'] for msg in chat_history])
        
        query_keywords = self._extract_keywords(full_conversation_text)
        self.logger.debug(f"Extracted query keywords for recommendations: {query_keywords}")

        # 1. Recommend Freelancers based on skills and title
        if self.freelancers:
            scored_freelancers = []
            for freelancer in self.freelancers:
                score = 0
                match_details = {"specialties": [], "skills": [], "title": [], "bio": []}

                # Extract keywords from different fields
                specialty_keywords = self._extract_keywords(" ".join(freelancer.get("specialties", [])).lower())
                
                # Extract skill names from the list of skill objects
                skill_names = [skill.get('name', '') for skill in freelancer.get("skills", [])]
                skill_keywords = self._extract_keywords(" ".join(skill_names).lower())
                
                title_keywords = self._extract_keywords(freelancer.get("title", "").lower())
                bio_keywords = self._extract_keywords(freelancer.get("bio", "").lower())

                # Weighted scoring
                common_specialties = query_keywords.intersection(specialty_keywords)
                score += len(common_specialties) * 4 # Highest weight for specialty match
                if common_specialties:
                    match_details["specialties"] = list(common_specialties)

                common_skills = query_keywords.intersection(skill_keywords)
                score += len(common_skills) * 3  # High weight for direct skill match
                if common_skills:
                    match_details["skills"] = list(common_skills)

                common_title = query_keywords.intersection(title_keywords)
                score += len(common_title) * 2  # Medium weight for title match
                if common_title:
                    match_details["title"] = list(common_title)

                common_bio = query_keywords.intersection(bio_keywords)
                score += len(common_bio) * 1  # Low weight for bio match
                if common_bio:
                    match_details["bio"] = list(common_bio)
                
                if score > 0:
                    match_percentage = min(int((score / MAX_EXPECTED_SCORE) * 100), 100)
                    self.logger.debug(f"Freelancer '{freelancer.get('name')}' scored {score}, percentage: {match_percentage}%")
                    scored_freelancers.append({
                        "freelancer": freelancer, 
                        "score": score, 
                        "match_percentage": match_percentage,
                        "match_details": match_details
                    })
            
            # Sort by score and take top k
            scored_freelancers.sort(key=lambda x: x["score"], reverse=True)
            freelancer_recs = scored_freelancers[:k_freelancers]
            self.logger.info(f"Found {len(freelancer_recs)} freelancer recommendations matching criteria.")

        # 2. Recommend Knowledge Base Articles
        is_platform_query = any(pk in full_conversation_text.lower() for pk in self.platform_keywords)
        
        if is_platform_query and self.knowledge_base_articles:
            scored_articles = []
            for article_name in self.knowledge_base_articles:
                score = 0
                article_keywords = self._extract_keywords(article_name.replace('_', ' ').replace('.md', ''))
                common_article_keywords = query_keywords.intersection(article_keywords)
                score += len(common_article_keywords)

                # Boost articles that seem generally relevant to platform usage
                if any(pk_word in article_name.lower() for pk_word in self.platform_keywords):
                    score += 1 # Small boost for platform-related article names
                
                if score > 0:
                    scored_articles.append({"article_name": article_name, "score": score, "matched_keywords": list(common_article_keywords)})
            
            scored_articles.sort(key=lambda x: x["score"], reverse=True)
            article_recs = scored_articles[:k_articles]
            self.logger.info(f"Found {len(article_recs)} article recommendations matching criteria.")

        return {
            "freelancers": freelancer_recs,
            "articles": article_recs
        }

# Example Usage (for testing)
if __name__ == '__main__':
    service = RecommendationService()
    sample_history_client = [
        "How do I post a job for a UX designer?",
        "Find me a freelance graphic designer with experience in branding."
    ]
    recs_client = service.get_recommendations(sample_history_client, current_query="Need a Python developer for a web app")
    print("\n--- Client Recommendations ---")
    print(json.dumps(recs_client, indent=2))

    sample_history_freelancer = [
        "How do payments work?",
        "best practices for my profile"
    ]
    recs_freelancer = service.get_recommendations(sample_history_freelancer)
    print("\n--- Freelancer Recommendations ---")
    print(json.dumps(recs_freelancer, indent=2))
