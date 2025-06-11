import json
import os
import re

# Constants
USER_PROFILES_PATH = "data/user_profiles/sample_user_profiles.json"
KNOWLEDGE_BASE_DIR = "data/knowledge_base"

class RecommendationService:
    def __init__(self):
        print("Initializing RecommendationService...")
        self.user_profiles = self._load_user_profiles()
        self.knowledge_base_articles = self._list_kb_articles()
        # Simple keyword lists for matching
        self.platform_keywords = [
            'payment', 'dispute', 'hire', 'find talent', 'post job', 
            'profile', 'account', 'getting started', 'best practice'
        ]
        print("RecommendationService initialized successfully.")

    def _load_user_profiles(self):
        if not os.path.exists(USER_PROFILES_PATH):
            print(f"Warning: User profiles not found at {USER_PROFILES_PATH}")
            return []
        try:
            with open(USER_PROFILES_PATH, 'r') as f:
                data = json.load(f)
                # We are interested in the 'freelancers' list from the sample data
                return data.get("freelancers", []) 
        except Exception as e:
            print(f"Error loading user profiles: {e}")
            return []

    def _list_kb_articles(self):
        if not os.path.isdir(KNOWLEDGE_BASE_DIR):
            print(f"Warning: Knowledge base directory not found at {KNOWLEDGE_BASE_DIR}")
            return []
        try:
            return [f for f in os.listdir(KNOWLEDGE_BASE_DIR) if f.endswith('.md')]
        except Exception as e:
            print(f"Error listing knowledge base articles: {e}")
            return []

    def _extract_keywords(self, text):
        """Extracts simple keywords from text."""
        text = text.lower()
        # Remove punctuation and split into words
        words = re.findall(r'\b\w+\b', text)
        # Could add more sophisticated NLP here (e.g., stemming, stop word removal)
        return set(words)

    def get_recommendations(self, query_history, current_query=None, k_freelancers=3, k_articles=2):
        """
        Generates recommendations based on query history and current query.
        query_history: A list of past query strings.
        current_query: The most recent query string (optional).
        """
        print(f"Generating recommendations for history: {query_history}, current: {current_query}")
        freelancer_recs = []
        article_recs = []

        all_queries_text = " ".join(query_history)
        if current_query:
            all_queries_text += " " + current_query
        
        query_keywords = self._extract_keywords(all_queries_text)
        print(f"Extracted query keywords: {query_keywords}")

        # 1. Recommend Freelancers based on skills
        if self.user_profiles:
            scored_freelancers = []
            for freelancer in self.user_profiles:
                score = 0
                freelancer_skills_text = " ".join(freelancer.get("skills", [])) + " " + freelancer.get("bio", "")
                freelancer_keywords = self._extract_keywords(freelancer_skills_text.lower())
                
                # Simple keyword matching for skills
                common_skills = query_keywords.intersection(freelancer_keywords)
                score += len(common_skills)
                
                if score > 0:
                    scored_freelancers.append({"freelancer": freelancer, "score": score, "matched_skills": list(common_skills)})
            
            # Sort by score and take top k
            scored_freelancers.sort(key=lambda x: x["score"], reverse=True)
            freelancer_recs = scored_freelancers[:k_freelancers]
            print(f"Found {len(freelancer_recs)} freelancer recommendations.")

        # 2. Recommend Knowledge Base Articles
        is_platform_query = any(pk in all_queries_text.lower() for pk in self.platform_keywords)
        
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
            print(f"Found {len(article_recs)} article recommendations.")

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
