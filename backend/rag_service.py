import os
import pickle
import faiss
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np

# Constants
INDEX_PATH = "data/vector_store.faiss"
DOC_CHUNKS_PATH = "data/doc_chunks.pkl"

class RAGService:
    def __init__(self):
        print("Initializing RAGService...")
        self._load_dotenv_and_configure_api()
        self.embedding_model = "models/embedding-001"
        self.llm = genai.GenerativeModel('gemini-pro')
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
        return genai.embed_content(model=self.embedding_model, content=query, task_type="RETRIEVAL_QUERY")['embedding']

    def search_knowledge_base(self, query, k=5):
        """Searches the vector store for the most relevant document chunks."""
        query_embedding = self._get_query_embedding(query)
        query_vector = np.array([query_embedding])
        
        # D: distances, I: indices
        distances, indices = self.index.search(query_vector, k)
        
        # Filter results based on a distance threshold to manage uncertainty
        # This threshold is heuristic and may need tuning.
        # A lower L2 distance means higher similarity.
        threshold = 1.0 
        results = []
        for i in range(len(indices[0])):
            if distances[0][i] < threshold:
                chunk_index = indices[0][i]
                results.append(self.doc_chunks[chunk_index])
        
        return results

    def answer_query(self, query):
        """Answers a user query using the RAG pipeline."""
        print(f"Received query: {query}")
        relevant_chunks = self.search_knowledge_base(query)

        if not relevant_chunks:
            print("No relevant information found.")
            return {
                "answer": "I'm sorry, but I don't have enough information to answer that question. Please try asking in a different way.",
                "sources": []
            }

        context = "\n\n---\n\n".join([chunk['text'] for chunk in relevant_chunks])
        sources = sorted(list(set([chunk['metadata']['doc_name'] for chunk in relevant_chunks])))

        prompt = f"""
        You are an expert assistant for the Shakers platform. Your task is to answer the user's question based *only* on the provided context.
        If the context does not contain the answer, state that you don't have enough information.
        Be clear, concise, and helpful. Cite the sources used in your answer.

        CONTEXT:
        {context}

        QUESTION:
        {query}

        ANSWER:
        """

        try:
            print("Generating response from LLM...")
            response = self.llm.generate_content(prompt)
            return {
                "answer": response.text,
                "sources": sources
            }
        except Exception as e:
            print(f"Error generating content from LLM: {e}")
            return {
                "answer": "I encountered an error while trying to generate a response. Please try again later.",
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
