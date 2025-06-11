import os
import pickle
import faiss
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np

class RAGServiceError(Exception):
    """Custom exception for RAGService errors."""
    pass

# Constants
INDEX_PATH = "data/vector_store.faiss"
DOC_CHUNKS_PATH = "data/doc_chunks.pkl"

class RAGService:
    def __init__(self):
        print("Initializing RAGService...")
        self._load_dotenv_and_configure_api()
        self.embedding_model = "models/embedding-001"
        self.llm = genai.GenerativeModel('gemini-1.5-flash-latest')
        self.index = self._load_faiss_index()
        self.doc_chunks = self._load_doc_chunks()
        print("RAGService initialized successfully.")

    def _load_dotenv_and_configure_api(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        genai.configure(api_key=api_key)

    def _load_faiss_index(self):
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}. Please run the indexing script first.")
        print(f"Loading FAISS index from {INDEX_PATH}...")
        return faiss.read_index(INDEX_PATH)

    def _load_doc_chunks(self):
        if not os.path.exists(DOC_CHUNKS_PATH):
            raise FileNotFoundError(f"Document chunks not found at {DOC_CHUNKS_PATH}. Please run the indexing script first.")
        print(f"Loading document chunks from {DOC_CHUNKS_PATH}...")
        with open(DOC_CHUNKS_PATH, 'rb') as f:
            return pickle.load(f)

    def _get_query_embedding(self, query):
        print(f"Generating embedding for query: '{query[:100]}...'")
        try:
            result = genai.embed_content(model=self.embedding_model, content=query, task_type="RETRIEVAL_QUERY")
            print("Query embedding generated successfully.")
            return result['embedding']
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            raise # Re-raise the exception to be caught by the caller

    def search_knowledge_base(self, query, k=5):
        """Searches the vector store for the most relevant document chunks."""
        try:
            query_embedding = self._get_query_embedding(query)
            query_vector = np.array([query_embedding])
            
            print(f"Searching FAISS index with query vector of shape: {query_vector.shape}")
            distances, indices = self.index.search(query_vector, k)
            print(f"FAISS search completed. Distances: {distances}, Indices: {indices}")
            
            threshold = 1.0 
            results = []
            if indices.size > 0:
                for i in range(len(indices[0])):
                    if distances[0][i] < threshold:
                        chunk_index = indices[0][i]
                        if 0 <= chunk_index < len(self.doc_chunks):
                            results.append(self.doc_chunks[chunk_index])
                        else:
                            print(f"Warning: Invalid chunk index {chunk_index} from FAISS search.")
            print(f"Found {len(results)} relevant chunks after filtering by threshold.")
            return results
        except Exception as e:
            print(f"Error during knowledge base search: {e}")
            return [] # Return empty list on error to allow graceful failure

    def answer_query(self, query):
        """Answers a user query using the RAG pipeline."""
        print(f"RAGService: Received query: '{query}'")
        relevant_chunks = self.search_knowledge_base(query)

        if not relevant_chunks:
            print("RAGService: No relevant information found after search.")
            return {
                "answer": "I'm sorry, but I don't have enough information to answer that question based on the available documents. Please try asking in a different way or about a topic covered in our knowledge base.",
                "sources": []
            }

        print(f"RAGService: Found {len(relevant_chunks)} relevant chunks.")
        for i, chunk in enumerate(relevant_chunks):
            print(f"  Chunk {i+1} (ID: {chunk.get('id', 'N/A')} from {chunk.get('metadata', {}).get('doc_name', 'N/A')}): {chunk.get('text', '')[:100]}...")

        context = "\n\n---\n\n".join([chunk['text'] for chunk in relevant_chunks])
        sources = sorted(list(set([chunk['metadata']['doc_name'] for chunk in relevant_chunks])))

        print(f"RAGService: Context prepared for LLM (length: {len(context)} chars). Sources: {sources}")

        prompt = f"""
        You are an expert assistant for the Makers platform. Your task is to answer the user's question based *only* on the provided context.
        If the context does not contain the answer, state that you don't have enough information from the provided documents.
        Be clear, concise, and helpful. If you use information from the context, mention the source document names listed if relevant (e.g., 'According to 01_getting_started.md...').

        CONTEXT:
        {context}

        QUESTION:
        {query}

        ANSWER:
        """
        print(f"RAGService: Prompt prepared for LLM:\n{prompt[:500]}...\n")

        try:
            print("RAGService: Generating response from LLM...")
            # Note: Consider adding safety_settings if needed, e.g.,
            # response = self.llm.generate_content(prompt, safety_settings={'HARASSMENT': 'BLOCK_NONE'}) 
            # Check Gemini API documentation for appropriate settings.
            response = self.llm.generate_content(prompt)
            generated_text = response.text
            print(f"RAGService: LLM response received: {generated_text[:200]}...")
            return {
                "answer": generated_text,
                "sources": sources
            }
        except Exception as e:
            print(f"RAGService: Error generating content from LLM: {e}")
            # The 'response' object may not exist if the API call itself fails, so we cannot check for prompt_feedback.
            return {
                "answer": f"I encountered an error while trying to generate a response. The API call failed. Details: {str(e)}",
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
