import os
import pickle
import faiss
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np
import logging

class RAGServiceError(Exception):
    """Custom exception for RAGService errors."""
    pass

# Constants
INDEX_PATH = "data/vector_store.faiss"
DOC_CHUNKS_PATH = "data/doc_chunks.pkl"

class RAGService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing RAGService...")
        try:
            self._load_dotenv_and_configure_api()
            self.embedding_model = "models/embedding-001"
            self.llm = genai.GenerativeModel('gemini-1.5-flash-latest')
            self.index = self._load_faiss_index()
            self.doc_chunks = self._load_doc_chunks()
            self.logger.info("RAGService initialized successfully.")
        except Exception as e:
            self.logger.critical(f"Failed to initialize RAGService: {e}", exc_info=True)
            # RAGServiceError is defined, but we might want to raise it here
            # or let the app.py handle the fact that RAGService might be None.
            # For now, just logging the critical failure.
            raise RAGServiceError(f"RAGService initialization failed: {e}") from e

    def _load_dotenv_and_configure_api(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            self.logger.error("GEMINI_API_KEY not found in environment variables.")
            raise RAGServiceError("GEMINI_API_KEY not found in environment variables.") # Changed to RAGServiceError
        genai.configure(api_key=api_key)
        self.logger.info("Gemini API configured.")

    def _load_faiss_index(self):
        if not os.path.exists(INDEX_PATH):
            self.logger.error(f"FAISS index not found at {INDEX_PATH}.")
            raise RAGServiceError(f"FAISS index not found at {INDEX_PATH}. Please run the indexing script first.")
        self.logger.info(f"Loading FAISS index from {INDEX_PATH}...")
        return faiss.read_index(INDEX_PATH)

    def _load_doc_chunks(self):
        if not os.path.exists(DOC_CHUNKS_PATH):
            self.logger.error(f"Document chunks not found at {DOC_CHUNKS_PATH}.")
            raise RAGServiceError(f"Document chunks not found at {DOC_CHUNKS_PATH}. Please run the indexing script first.")
        self.logger.info(f"Loading document chunks from {DOC_CHUNKS_PATH}...")
        with open(DOC_CHUNKS_PATH, 'rb') as f:
            return pickle.load(f)

    def _get_query_embedding(self, query):
        self.logger.debug(f"Generating embedding for query: '{query[:100]}...'" )
        try:
            result = genai.embed_content(model=self.embedding_model, content=query, task_type="RETRIEVAL_QUERY")
            self.logger.debug("Query embedding generated successfully.")
            return result['embedding']
        except Exception as e:
            self.logger.error(f"Error generating query embedding: {e}", exc_info=True)
            raise RAGServiceError(f"Error generating query embedding: {e}") from e

    def search_knowledge_base(self, query, k=5):
        """Searches the vector store for the most relevant document chunks."""
        try:
            query_embedding = self._get_query_embedding(query)
            query_vector = np.array([query_embedding])
            
            self.logger.debug(f"Searching FAISS index with query vector of shape: {query_vector.shape}")
            distances, indices = self.index.search(query_vector, k)
            self.logger.debug(f"FAISS search completed. Distances: {distances}, Indices: {indices}")
            
            threshold = 1.0 # This threshold might need tuning
            results = []
            if indices.size > 0:
                for i in range(len(indices[0])):
                    if distances[0][i] < threshold:
                        chunk_index = indices[0][i]
                        if 0 <= chunk_index < len(self.doc_chunks):
                            results.append(self.doc_chunks[chunk_index])
                        else:
                            self.logger.warning(f"Invalid chunk index {chunk_index} from FAISS search (distance: {distances[0][i]}).")
            self.logger.info(f"Found {len(results)} relevant chunks after filtering by threshold {threshold}.")
            return results
        except Exception as e:
            self.logger.error(f"Error during knowledge base search for query '{query[:100]}...': {e}", exc_info=True)
            return [] # Return empty list on error to allow graceful failure

    def answer_query(self, current_query, chat_history=None):
        """Answers a user query using the RAG pipeline, considering chat history for context."""
        self.logger.info(f"Received current_query: '{current_query[:100]}...', with chat_history: {chat_history is not None}")
        
        # Knowledge base search is primarily driven by the current query for relevance
        relevant_chunks = self.search_knowledge_base(current_query)

        if not relevant_chunks:
            self.logger.info("No relevant information found after search for the current query.")
            # Construct a conversational no-info response
            history_context_for_prompt = ""
            if chat_history:
                for message in chat_history:
                    history_context_for_prompt += f"{message['role'].capitalize()}: {message['content']}\n"
            
            no_info_prompt = f"""
            You are an expert assistant for the Makers platform. 
            The user asked: "{current_query}"
            Previously, the conversation was:
            {history_context_for_prompt}
            Based on the available knowledge base, you could not find specific documents relevant to "{current_query}".
            Respond to the user acknowledging their current question and explaining that you don't have specific information for it from the knowledge base, but try to be helpful based on the general conversation if possible or suggest rephrasing.
            Do not invent information. If the conversation doesn't give clues, just say you can't find info for the current query.
            ANSWER:
            """
            self.logger.debug(f"No-info prompt for LLM: {no_info_prompt[:300]}...")
            try:
                response = self.llm.generate_content(no_info_prompt)
                self.logger.info("Generated no-info response from LLM.")
                return {"answer": response.text, "sources": []}
            except Exception as e:
                self.logger.error(f"Error generating no-info response from LLM: {e}", exc_info=True)
                return {"answer": "I'm sorry, I couldn't find information for your query and had trouble generating a response.", "sources": []}

        self.logger.info(f"Found {len(relevant_chunks)} relevant chunks for current query.")
        # for i, chunk in enumerate(relevant_chunks):
        #     self.logger.debug(f"  Chunk {i+1} (ID: {chunk.get('id', 'N/A')} from {chunk.get('metadata', {}).get('doc_name', 'N/A')}): {chunk.get('text', '')[:100]}...")

        context = "\n\n---\n\n".join([chunk['text'] for chunk in relevant_chunks])
        sources = sorted(list(set([chunk['metadata']['doc_name'] for chunk in relevant_chunks])))

        self.logger.info(f"Context prepared for LLM (length: {len(context)} chars). Sources: {sources}")

        # Construct conversational prompt
        history_for_prompt = ""
        if chat_history:
            for message in chat_history:
                # We only want to pass the actual content to the LLM for history
                if message['role'] == 'user':
                    history_for_prompt += f"User: {message['content']}\n"
                elif message['role'] == 'assistant':
                    history_for_prompt += f"Assistant: {message['content']}\n"

        prompt = f"""
        You are an expert assistant for the Makers platform. 
        Your task is to answer the user's current question based on the provided context and the preceding conversation history.
        If the context does not contain the answer to the current question, state that you don't have enough information from the provided documents for *this specific question*.
        Be clear, concise, and helpful. Refer to the conversation history if it helps clarify the current question or if the user is referring to something said earlier.
        If you use information from the context, mention the source document names listed if relevant (e.g., 'According to 01_getting_started.md...').

        CONVERSATION HISTORY:
        {history_for_prompt}
        PROVIDED CONTEXT for the current question "{current_query}":
        {context}

        CURRENT QUESTION:
        User: {current_query}

        ANSWER (Respond as Assistant):
        """
        self.logger.debug(f"Conversational prompt prepared for LLM (first 500 chars):\n{prompt[:500]}...\n")

        try:
            self.logger.info("Generating response from LLM...")
            response = self.llm.generate_content(prompt)
            self.logger.info("Successfully generated response from LLM.")
            return {"answer": response.text, "sources": sources}
        except Exception as e:
            self.logger.error(f"Error generating response from LLM: {e}", exc_info=True)
            return {
                "answer": "I'm sorry, but I encountered an issue while trying to generate a response. Please try again later.",
                "sources": []
            }

# Example usage:
if __name__ == '__main__':
    rag_service = RAGService()
    test_query = "How do payments work for freelancers?"
    result = rag_service.answer_query(test_query)
    print("\n--- Query Result ---")
    print(f"Question: {test_query}")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")

    test_query_2 = "What is the best way to find a good restaurant?"
    result_2 = rag_service.answer_query(test_query_2)
    print("\n--- Query Result ---")
    print(f"Question: {test_query_2}")
    print(f"Answer: {result_2['answer']}")
    print(f"Sources: {result_2['sources']}")
