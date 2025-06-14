import sqlite3
import json
import logging
from datetime import datetime
import os # Added import

# Determine the project root directory dynamically
_SERVICE_FILE_PATH = os.path.abspath(__file__)
_BACKEND_DIR = os.path.dirname(_SERVICE_FILE_PATH)
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)
DATABASE_PATH = os.path.join(_PROJECT_ROOT, "data", "talent_matcher.db")

class DatabaseService:
    def __init__(self, db_path=DATABASE_PATH):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._create_tables()

    def _get_db_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row # Access columns by name
        return conn

    def _create_tables(self):
        conn = None
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()

            # User Profiles Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    preferences TEXT  -- JSON string for user preferences
                )
            ''')

            # Chat History Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_history (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_id TEXT, -- Can be NULL if user is not logged in
                    sender TEXT NOT NULL, -- 'user' or 'assistant'
                    message TEXT NOT NULL,
                    model_used TEXT, -- e.g., 'gemini-1.5-flash', 'groq-llama3-70b'
                    intent_detected TEXT, -- e.g., 'qa', 'recommendation_request'
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES user_profiles (user_id)
                )
            ''')

            # API Metrics Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS api_metrics (
                    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    api_type TEXT NOT NULL, -- 'gemini_generate', 'groq_generate', 'embedding', 'llama_guard'
                    model_name TEXT, -- Specific model if applicable
                    call_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    latency_ms REAL,
                    cost REAL,
                    tokens_input INTEGER,
                    tokens_output INTEGER,
                    error_occurred BOOLEAN DEFAULT FALSE,
                    error_message TEXT
                )
            ''')

            # Feedback Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    message_id INTEGER, -- Link to the specific message that received feedback
                    recommendation_id TEXT, -- If feedback is for a specific recommendation item
                    user_id TEXT, 
                    rating INTEGER, -- e.g., 1 for thumbs up, -1 for thumbs down, 0 for neutral/no rating
                    comment TEXT,
                    feedback_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (message_id) REFERENCES chat_history (message_id),
                    FOREIGN KEY (user_id) REFERENCES user_profiles (user_id)
                )
            ''')
            conn.commit()
            self.logger.info("Database tables ensured to exist.")
        except sqlite3.Error as e:
            self.logger.error(f"Error creating database tables: {e}", exc_info=True)
        finally:
            if conn:
                conn.close()

    # --- User Profile Methods ---
    def add_user_profile(self, user_id, username, preferences=None):
        conn = None
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO user_profiles (user_id, username, preferences) VALUES (?, ?, ?)",
                (user_id, username, json.dumps(preferences) if preferences else None)
            )
            conn.commit()
            self.logger.info(f"Added user profile: {user_id} - {username}")
            return True
        except sqlite3.IntegrityError:
            self.logger.warning(f"User profile {user_id} or username {username} already exists.")
            return False
        except sqlite3.Error as e:
            self.logger.error(f"Error adding user profile {user_id}: {e}", exc_info=True)
            return False
        finally:
            if conn:
                conn.close()

    def get_user_profile(self, user_id):
        conn = None
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM user_profiles WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            if row:
                profile = dict(row)
                if profile.get('preferences'):
                    profile['preferences'] = json.loads(profile['preferences'])
                return profile
            return None
        except sqlite3.Error as e:
            self.logger.error(f"Error getting user profile {user_id}: {e}", exc_info=True)
            return None
        finally:
            if conn:
                conn.close()

    # --- Chat History Methods ---
    def add_chat_message(self, session_id, sender, message, user_id=None, model_used=None, intent_detected=None):
        conn = None
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO chat_history (session_id, user_id, sender, message, model_used, intent_detected) VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, user_id, sender, message, model_used, intent_detected)
            )
            conn.commit()
            message_id = cursor.lastrowid
            self.logger.info(f"Added chat message {message_id} to session {session_id}")
            return message_id
        except sqlite3.Error as e:
            self.logger.error(f"Error adding chat message to session {session_id}: {e}", exc_info=True)
            return None
        finally:
            if conn:
                conn.close()

    def get_chat_history(self, session_id, limit=50):
        conn = None
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM chat_history WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?", 
                (session_id, limit)
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows][::-1] # Return in chronological order
        except sqlite3.Error as e:
            self.logger.error(f"Error getting chat history for session {session_id}: {e}", exc_info=True)
            return []
        finally:
            if conn:
                conn.close()

    # --- API Metrics Methods ---
    def add_api_metric(self, session_id, api_type, latency_ms, cost, model_name=None, tokens_input=None, tokens_output=None, error_occurred=False, error_message=None):
        conn = None
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO api_metrics (session_id, api_type, model_name, latency_ms, cost, tokens_input, tokens_output, error_occurred, error_message) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (session_id, api_type, model_name, latency_ms, cost, tokens_input, tokens_output, error_occurred, error_message)
            )
            conn.commit()
            metric_id = cursor.lastrowid
            self.logger.info(f"Added API metric {metric_id} for API type {api_type} in session {session_id}")
            return metric_id
        except sqlite3.Error as e:
            self.logger.error(f"Error adding API metric for {api_type} in session {session_id}: {e}", exc_info=True)
            return None
        finally:
            if conn:
                conn.close()

    # --- Feedback Methods ---
    def add_feedback(self, session_id, rating, message_id=None, recommendation_id=None, user_id=None, comment=None):
        conn = None
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO feedback (session_id, message_id, recommendation_id, user_id, rating, comment) VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, message_id, recommendation_id, user_id, rating, comment)
            )
            conn.commit()
            feedback_id = cursor.lastrowid
            self.logger.info(f"Added feedback {feedback_id} for session {session_id}")
            return feedback_id
        except sqlite3.Error as e:
            self.logger.error(f"Error adding feedback for session {session_id}: {e}", exc_info=True)
            return None
        finally:
            if conn:
                conn.close()

if __name__ == '__main__':
    # Basic test and setup
    logging.basicConfig(level=logging.INFO)
    db_service = DatabaseService(db_path='../data/test_talent_matcher.db') # Use a test DB for direct script run
    
    # Test user profile
    db_service.add_user_profile('user123', 'testuser', {'theme': 'dark'})
    profile = db_service.get_user_profile('user123')
    print(f"User Profile: {profile}")

    # Test chat history
    msg_id = db_service.add_chat_message('sessionABC', 'user', 'Hello, I need a Python developer.', user_id='user123')
    db_service.add_chat_message('sessionABC', 'assistant', 'Sure, I can help with that!', user_id='user123', model_used='gemini-1.5-flash', intent_detected='recommendation_request')
    history = db_service.get_chat_history('sessionABC')
    print(f"Chat History: {history}")

    # Test API metric
    db_service.add_api_metric('sessionABC', 'gemini_generate', 120.5, 0.0001, model_name='gemini-1.5-flash', tokens_input=100, tokens_output=50)

    # Test feedback
    db_service.add_feedback('sessionABC', 1, message_id=msg_id, user_id='user123', comment='Great response!')

    print("Database service basic tests completed.")
