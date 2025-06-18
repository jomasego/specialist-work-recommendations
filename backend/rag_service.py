import os
import pickle
import faiss
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from groq import Groq
from dotenv import load_dotenv
import numpy as np
import logging
import time
from typing import Optional, Dict, Any # Added for type hinting

from .database_service import DatabaseService # Assuming database_service.py is in backend directory

class RAGServiceError(Exception):
    """Custom exception for RAGService errors."""
    pass

# Constants
INDEX_PATH = "data/vector_store.faiss"
DOC_CHUNKS_PATH = "data/doc_chunks.pkl"
MAX_HISTORY_TOKENS = 1000  # Approx 4000 chars
MAX_CONTEXT_CHARS = 8000   # Max characters for combined document context

# --- API Cost Constants (per 1,000 tokens) ---
# Last updated: 2025-06-12
# Gemini API (via google-generativeai SDK)
GEMINI_FLASH_INPUT_COST_PER_1K_TOKENS = 0.00035  # $0.35 / 1M tokens
GEMINI_FLASH_OUTPUT_COST_PER_1K_TOKENS = 0.00105 # $1.05 / 1M tokens
EMBEDDING_COST_PER_1K_TOKENS = 0.0001          # $0.10 / 1M tokens for text-embedding-004

# Placeholder pricing for Groq models - UPDATE WITH ACTUALS
# Using a general placeholder as specific model prices vary.
# Llama Guard, Mixtral, Llama3 etc. would have their own rates.
GROQ_GENERIC_INPUT_COST_PER_1K_TOKENS = 0.00025  # Placeholder e.g., $0.25 / 1M tokens
GROQ_GENERIC_OUTPUT_COST_PER_1K_TOKENS = 0.00025 # Placeholder e.g., $0.25 / 1M tokens
GROQ_LLAMA3_8B_INPUT_COST_PER_1K_TOKENS = 0.00005 # $0.05 / 1M tokens
GROQ_LLAMA3_8B_OUTPUT_COST_PER_1K_TOKENS = 0.00008 # $0.08 / 1M tokens
GROQ_QWEN_32B_INPUT_COST_PER_1K_TOKENS = 0.00029 # $0.29 / 1M tokens
GROQ_QWEN_32B_OUTPUT_COST_PER_1K_TOKENS = 0.00039 # $0.39 / 1M tokens

class RAGService:
    def __init__(self, db_service: DatabaseService, knowledge_base_dir="data/knowledge_base"):
        self.logger = logging.getLogger(__name__)
        self.db_service = db_service
        if not isinstance(db_service, DatabaseService):
            self.logger.error("RAGService initialized without a valid DatabaseService instance.")
            # Depending on strictness, could raise an error here
            # raise TypeError("db_service must be an instance of DatabaseService")
            # For now, allow it but log an error. Calls to db_service will fail if it's None/wrong type.
            pass 
        self.logger.info("Initializing RAGService...")
        try:
            load_dotenv()
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            groq_api_key = os.getenv("GROQ_API_KEY")

            if not gemini_api_key:
                raise RAGServiceError("GEMINI_API_KEY not found in environment variables.")
            if not groq_api_key:
                raise RAGServiceError("GROQ_API_KEY not found in environment variables.")

            genai.configure(api_key=gemini_api_key)
            self.groq_client = Groq(api_key=groq_api_key)

            self.embedding_model_name = "models/embedding-001" 
            # For genai.embed_content and genai.count_tokens, we pass the model name string directly.
            # No need to instantiate a GenerativeModel for embeddings if only using these functions.
            
            self.gemini_llm = genai.GenerativeModel(
                'gemini-1.5-flash-latest',
                safety_settings={ # More restrictive safety settings
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                }
            )
            # For Groq, model selection happens at call time in answer_query

            self.knowledge_base_dir = knowledge_base_dir
            self.doc_chunks_path = DOC_CHUNKS_PATH
            self.index_path = INDEX_PATH

            # If index or chunks don't exist, build them.
            if not os.path.exists(self.index_path) or not os.path.exists(self.doc_chunks_path):
                self.logger.info(f"Index or chunks not found. Building new index from source: {self.knowledge_base_dir}")
                self._build_index()
            
            self.index = self._load_faiss_index()
            self.doc_chunks = self._load_doc_chunks()

            self.api_call_counts = {
                'gemini_generate': 0, 
                'groq_generate': 0, 
                'embedding': 0, 
                'llama_guard': 0,
                'gemini_no_info': 0,
                'groq_no_info': 0
            }
            self.api_total_latency = {
                'gemini_generate': 0.0, 
                'groq_generate': 0.0, 
                'embedding': 0.0, 
                'llama_guard': 0.0,
                'gemini_no_info': 0.0,
                'groq_no_info': 0.0
            }
            self.api_total_cost = {
                'gemini_generate': 0.0, 
                'groq_generate': 0.0, 
                'embedding': 0.0, 
                'llama_guard': 0.0,
                'gemini_no_info': 0.0,
                'groq_no_info': 0.0
            }
            # Load FAISS index and doc_chunks after successful API client initialization
            self.index = self._load_faiss_index()
            self.doc_chunks = self._load_doc_chunks()
            self.logger.info("RAGService initialized successfully.")
        except Exception as e:
            self.logger.critical(f"Failed to initialize RAGService: {e}", exc_info=True)
            raise RAGServiceError(f"RAGService initialization failed: {e}") from e

    def _load_api_keys(self):
        load_dotenv()
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not gemini_api_key:
            self.logger.error("GEMINI_API_KEY not found in environment variables.")
            raise RAGServiceError("GEMINI_API_KEY not found.")
        if not groq_api_key:
            self.logger.error("GROQ_API_KEY not found in environment variables.")
            raise RAGServiceError("GROQ_API_KEY not found.")
        self.logger.info("API keys loaded successfully.")
        return gemini_api_key, groq_api_key

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

    def _get_query_embedding(self, query: str, session_id: Optional[str] = None):
        self.logger.debug(f"Generating embedding for query: '{query[:100]}...'" )
        self.logger.info(f"DEBUG: google.generativeai version: {genai.__version__ if hasattr(genai, '__version__') else 'unknown'}")
        start_time = time.perf_counter()
        try:
            # Cost calculation for embedding
            # TEMPORARY BYPASS: Suspected issue with genai.count_tokens. Assuming 0 for now.
            self.logger.warning("TEMPORARY BYPASS: genai.count_tokens for embeddings is being skipped due to suspected library issue. Token count and cost set to 0.")
            token_count = 0
            cost = 0.0
            # token_count_response = genai.count_tokens(model=self.embedding_model_name, contents=query) # Original line causing AttributeError
            # token_count = token_count_response.total_tokens # Original line
            # cost = (token_count / 1000) * EMBEDDING_COST_PER_1K_TOKENS # Original line
            self.api_total_cost['embedding'] += cost
            self.logger.info(f"Embedding call cost: ${cost:.6f} for {token_count} tokens.")

            result = genai.embed_content(model=self.embedding_model_name, content=query, task_type="RETRIEVAL_QUERY")
            latency = time.perf_counter() - start_time
            self.api_call_counts['embedding'] += 1
            self.api_total_latency['embedding'] += latency
            self.logger.info(f"Embedding generation took {latency:.4f}s.")
            self.logger.debug("Query embedding generated successfully.")

            if self.db_service:
                self.db_service.add_api_metric(
                    session_id=session_id,
                    api_type='embedding',
                    model_name=self.embedding_model_name,
                    latency_ms=latency * 1000,
                    cost=cost,
                    tokens_input=token_count,
                    tokens_output=0 # Embeddings typically don't have 'output' tokens in the same sense as LLMs
                )
            return result['embedding'], token_count # Return embedding and token count
        except Exception as e:
            self.logger.error(f"Error generating query embedding: {e}", exc_info=True)
            if self.db_service:
                self.db_service.add_api_metric(
                    session_id=session_id,
                    api_type='embedding',
                    model_name=self.embedding_model_name,
                    latency_ms= (time.perf_counter() - start_time) * 1000 if 'start_time' in locals() else 0,
                    cost=0, # Failed call
                    error_occurred=True,
                    error_message=str(e)
                )
            # Ensure we return a value even in error cases for consistent tuple unpacking, though caller should handle Exception
            raise RAGServiceError(f"Error generating query embedding: {e}") from e
            # Unreachable, but for clarity: return None, 0 

    def search_knowledge_base(self, query: str, k: int = 5, session_id: Optional[str] = None) -> tuple[list, int]:
        """Searches the vector store for the most relevant document chunks. Returns (results, embedding_token_count)."""
        embedding_token_count = 0 # Default in case of early exit or error before embedding
        try:
            query_embedding, embedding_token_count = self._get_query_embedding(query, session_id=session_id)
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
            return results, embedding_token_count
        except Exception as e:
            self.logger.error(f"Error during knowledge base search for query '{query[:100]}...': {e}", exc_info=True)
            return [], embedding_token_count # Return empty list and current token count on error

    def _is_query_safe_groq(self, query: str, history_context: str, session_id: Optional[str] = None) -> bool:
        """
        Uses Llama Guard 4 via Groq to check if a query is safe.
        (Placeholder - actual Llama Guard call needs to be implemented)
        """
        self.logger.info(f"Checking query safety with Llama Guard 4 for: '{query[:100]}...' (Stubbed)")
        # Placeholder for Llama Guard prompt and API call
        # llama_guard_prompt = f"""[INST] Task: Check if there is unsafe content in 'User' messages in conversations. Categories of unsafe content are: violence, sexual content, hate speech, harassment, self-harm, criminal planning, guns, and illicit drugs.

        # <BEGIN CONVERSATION>
        # User: {query}
        # {history_context}
        # <END CONVERSATION>
        # [/INST]"""
        # try:
        #     start_time = time.perf_counter()
        #     # response = self.groq_client.chat.completions.create(
        #     #     messages=[{"role": "user", "content": llama_guard_prompt}],
        #     #     model="meta-llama/LlamaGuard-2-8b" # Ensure this model is available and correct
        #     # )
        #     # latency = time.perf_counter() - start_time
        #     # self.api_call_counts['llama_guard'] += 1
        #     # self.api_total_latency['llama_guard'] += latency
        #     # result_text = response.choices[0].message.content.strip()
        #     # self.logger.info(f"Llama Guard check took {latency:.4f}s. Result: {result_text}")
        #     # is_safe = "unsafe" not in result_text.lower()

        #     # For now, assume safe until fully implemented
        #     is_safe = True 
            
        #     # Cost for Llama Guard (if applicable and known)
        #     # input_tokens = response.usage.prompt_tokens if hasattr(response, 'usage') else len(llama_guard_prompt) // 4
        #     # output_tokens = response.usage.completion_tokens if hasattr(response, 'usage') else len(result_text) // 4
        #     # cost = ... (calculate if pricing is known)
        #     # self.api_total_cost['llama_guard'] += cost
        #     # if self.db_service:
        #     #     self.db_service.add_api_metric(
        #     #         session_id=session_id, api_type='llama_guard', model_name='meta-llama/LlamaGuard-2-8b',
        #     #         latency_ms=latency * 1000, cost=cost, tokens_input=input_tokens, tokens_output=output_tokens
        #     #     )
        #     return is_safe
        # except Exception as e:
        #     self.logger.error(f"Error during Llama Guard safety check: {e}", exc_info=True)
        #     if self.db_service:
        #          self.db_service.add_api_metric(
        #             session_id=session_id, api_type='llama_guard', model_name='meta-llama/LlamaGuard-2-8b',
        #             latency_ms=0, cost=0, error_occurred=True, error_message=str(e)
        #         )
        #     return False # Default to not safe on error
        return True # Stubbed to always return True

    def answer_query(self, current_query: str, chat_history: Optional[list] = None, model: str = "gemini-1.5-flash-latest", session_id: Optional[str] = None, k_chunks: int = 5) -> Dict[str, Any]:
        """Generates an answer to a query using RAG, including token counting and cost calculation."""
        token_usage_output = {
            "embedding_tokens": 0,
            "llm_input_tokens": 0,
            "llm_output_tokens": 0,
            "cost": 0.0, # This will be the sum of embedding and LLM costs for this specific call
            "model_name": model
        }
        assistant_message_id = None
        sources_for_output = []
        embedding_cost_this_call = 0.0
        llm_cost_this_call = 0.0

        try:
            self.logger.info(f"Answering query: '{current_query[:100]}...' with model: {model}, session: {session_id}")
            
            # 1. Search knowledge base and get embedding tokens/cost
            # _get_query_embedding already updates self.api_total_cost['embedding'] and logs to DB
            relevant_chunks, embedding_tokens = self.search_knowledge_base(current_query, k=k_chunks, session_id=session_id)
            token_usage_output["embedding_tokens"] = embedding_tokens
            embedding_cost_this_call = (embedding_tokens / 1000) * EMBEDDING_COST_PER_1K_TOKENS
            
            history_context_str = "".join([
                f"{msg['role'].capitalize()}: {msg['content']}\n" 
                for msg in chat_history
            ]) if chat_history else ""

            # Optional: Safety check (currently stubbed)
            # if not self._is_query_safe_groq(current_query, history_context_str, session_id):
            #     self.logger.warning(f"Query flagged as unsafe: {current_query[:100]}...")
            #     answer = "I'm sorry, but I cannot process this request due to safety guidelines (content policy)."
            #     if self.db_service:
            #         assistant_message_id = self.db_service.add_chat_message(session_id=session_id, sender='assistant', message=answer, model_used=model + "_unsafe")
            #     token_usage_output["cost"] = embedding_cost_this_call # Only embedding cost if query is unsafe before LLM
            #     return {"answer": answer, "sources": [], "assistant_message_id": assistant_message_id, "token_usage": token_usage_output}

            # 2. Handle no relevant chunks (No-Info Path)
            if not relevant_chunks:
                self.logger.info(f"No relevant chunks found for query: '{current_query[:100]}...'. Generating no-info response.")
                no_info_prompt = f"""The user asked: "{current_query}"
Conversation history:
{history_context_str}
You could not find specific documents for this question. Acknowledge this and respond helpfully based on the conversation, or suggest rephrasing. Do not invent information.
ANSWER:"""
                
                start_time_no_info = time.perf_counter()
                answer_text = "I'm sorry, I couldn't find specific information for your query in the knowledge base."
                input_tokens, output_tokens = 0, 0
                api_type_no_info = ""

                if model.startswith('gemini'):
                    api_type_no_info = 'gemini_no_info'
                    response = self.gemini_llm.generate_content(no_info_prompt)
                    latency_no_info = time.perf_counter() - start_time_no_info
                    self.api_call_counts[api_type_no_info] += 1
                    self.api_total_latency[api_type_no_info] += latency_no_info
                    self.logger.info(f"Gemini no-info response generation took {latency_no_info:.4f}s.")

                    if response.prompt_feedback.block_reason:
                        self.logger.warning(f"Gemini no-info response blocked. Reason: {response.prompt_feedback.block_reason.name}")
                        answer_text = "I'm sorry, but I cannot process this request due to safety guidelines."
                        input_tokens = self.gemini_llm.count_tokens(no_info_prompt).total_tokens # Still count input tokens
                        # Output tokens remain 0 for blocked
                        if self.db_service:
                            self.db_service.add_api_metric(
                                session_id=session_id, api_type=api_type_no_info, model_name=self.gemini_llm.model_name,
                                latency_ms=latency_no_info * 1000, cost=0,
                                tokens_input=input_tokens, tokens_output=0,
                                error_occurred=True, error_message=f"Blocked: {response.prompt_feedback.block_reason.name}"
                            )
                    else:
                        answer_text = response.text
                        input_tokens = self.gemini_llm.count_tokens(no_info_prompt).total_tokens
                        output_tokens = response.usage_metadata.candidates_token_count if response.usage_metadata else self.gemini_llm.count_tokens(answer_text).total_tokens
                        llm_cost_this_call = ((input_tokens / 1000) * GEMINI_FLASH_INPUT_COST_PER_1K_TOKENS) + \
                                   ((output_tokens / 1000) * GEMINI_FLASH_OUTPUT_COST_PER_1K_TOKENS)
                        token_usage_output["llm_input_tokens"] = input_tokens
                        token_usage_output["llm_output_tokens"] = output_tokens
                        token_usage_output["cost"] = embedding_cost_this_call + llm_cost_this_call
                        self.api_total_cost[api_type_no_info] += llm_cost_this_call # Update global service cost
                        self.logger.info(f"Gemini no-info call cost: ${llm_cost_this_call:.6f} for {input_tokens}+{output_tokens} tokens.")
                        if self.db_service:
                            self.db_service.add_api_metric(
                                session_id=session_id, api_type=api_type_no_info, model_name=self.gemini_llm.model_name,
                                latency_ms=latency_no_info * 1000, cost=llm_cost_this_call,
                                tokens_input=input_tokens, tokens_output=output_tokens
                            )
                else: # Groq no-info
                    api_type_no_info = 'groq_no_info'
                    # Use the full model string from the 'model' parameter for Groq, e.g., "groq/llama3-8b-8192"
                    # The Groq client expects the specific model ID like "llama3-8b-8192"
                    groq_model_id = model.split('/')[-1] if '/' in model else model
                    response = self.groq_client.chat.completions.create(
                        messages=[{"role": "user", "content": no_info_prompt}], model=groq_model_id)
                    latency_no_info = time.perf_counter() - start_time_no_info
                    self.api_call_counts[api_type_no_info] += 1
                    self.api_total_latency[api_type_no_info] += latency_no_info
                    self.logger.info(f"Groq ({groq_model_id}) no-info response generation took {latency_no_info:.4f}s.")
                    answer_text = response.choices[0].message.content
                    
                    input_tokens = response.usage.prompt_tokens if hasattr(response, 'usage') and response.usage else len(no_info_prompt) // 4
                    output_tokens = response.usage.completion_tokens if hasattr(response, 'usage') and response.usage else len(answer_text) // 4

                    if 'llama3-8b' in groq_model_id:
                        input_cost_rate, output_cost_rate = GROQ_LLAMA3_8B_INPUT_COST_PER_1K_TOKENS, GROQ_LLAMA3_8B_OUTPUT_COST_PER_1K_TOKENS
                    elif 'qwen-32b' in groq_model_id: # Match specific model names
                        input_cost_rate, output_cost_rate = GROQ_QWEN_32B_INPUT_COST_PER_1K_TOKENS, GROQ_QWEN_32B_OUTPUT_COST_PER_1K_TOKENS
                    else: # Fallback to generic Groq pricing if model not specifically listed
                        input_cost_rate, output_cost_rate = GROQ_GENERIC_INPUT_COST_PER_1K_TOKENS, GROQ_GENERIC_OUTPUT_COST_PER_1K_TOKENS
                    
                    llm_cost_this_call = ((input_tokens / 1000) * input_cost_rate) + ((output_tokens / 1000) * output_cost_rate)
                    token_usage_output["llm_input_tokens"] = input_tokens
                    token_usage_output["llm_output_tokens"] = output_tokens
                    token_usage_output["cost"] = embedding_cost_this_call + llm_cost_this_call
                    self.api_total_cost[api_type_no_info] += llm_cost_this_call # Update global service cost
                    self.logger.info(f"Groq ({groq_model_id}) no-info call cost: ${llm_cost_this_call:.6f} for ~{input_tokens}+{output_tokens} tokens.")
                    if self.db_service:
                        self.db_service.add_api_metric(
                            session_id=session_id, api_type=api_type_no_info, model_name=groq_model_id,
                            latency_ms=latency_no_info * 1000, cost=llm_cost_this_call,
                            tokens_input=input_tokens, tokens_output=output_tokens
                        )
                
                token_usage_output['llm_input_tokens'] = input_tokens
                token_usage_output['llm_output_tokens'] = output_tokens
                token_usage_output['cost'] = embedding_cost_this_call + llm_cost_this_call
                
                if self.db_service:
                    assistant_message_id = self.db_service.add_chat_message(session_id=session_id, sender='assistant', message=answer_text, model_used=f"{model}_no_info")
                
                self.logger.info(f"FINAL RAG_SERVICE TOKEN USAGE (no-info path): {token_usage_output}")
                return {"answer": answer_text, "sources": [], "assistant_message_id": assistant_message_id, "token_usage": token_usage_output}

            # 3. Main RAG Path: Relevant chunks were found.
            self.logger.info(f"Found {len(relevant_chunks)} relevant chunks. Generating RAG answer.")
            # Assuming chunk is a dict with 'content' and 'source' (filename string)
            context_str = "\n---\n".join([
                f"Source Document: {chunk.get('source', 'Unknown Source')}\nContent: {chunk['content']}" 
                for chunk in relevant_chunks
            ])
            
            rag_prompt = f"""You are an expert assistant for the Makers platform. Your goal is to provide clear, interpretive answers based on the provided context documents. Do not just summarize; synthesize the information to directly answer the user's question.

User Question: "{current_query}"

Conversation History:
{history_context_str}
Context from Knowledge Base:
---
{context_str}
---

Based on the context and conversation history, provide a direct, synthesized answer to the user's question. If the context is insufficient, state that you couldn't find a definitive answer in the available documents and suggest what to do next. Cite the source documents by their name (e.g., [source: file_name.md]) where the information was found.
ANSWER:"""

            start_time_generate = time.perf_counter()
            answer_text = "I was unable to generate an answer based on the provided documents."
            input_tokens, output_tokens = 0, 0
            api_type_generate = ""

            if model.startswith('gemini'):
                api_type_generate = 'gemini_generate'
                response = self.gemini_llm.generate_content(rag_prompt)
                latency_generate = time.perf_counter() - start_time_generate
                self.api_call_counts[api_type_generate] += 1
                self.api_total_latency[api_type_generate] += latency_generate
                self.logger.info(f"Gemini RAG response generation took {latency_generate:.4f}s.")

                if response.prompt_feedback.block_reason:
                    self.logger.warning(f"Gemini RAG response blocked. Reason: {response.prompt_feedback.block_reason.name}")
                    answer_text = "I'm sorry, but I cannot process this request due to safety guidelines."
                    input_tokens = self.gemini_llm.count_tokens(rag_prompt).total_tokens
                    if self.db_service:
                        self.db_service.add_api_metric(
                            session_id=session_id, api_type=api_type_generate, model_name=self.gemini_llm.model_name,
                            latency_ms=latency_generate * 1000, cost=0,
                            tokens_input=input_tokens, tokens_output=0,
                            error_occurred=True, error_message=f"Blocked: {response.prompt_feedback.block_reason.name}"
                        )
                else:
                    answer_text = response.text
                    input_tokens = self.gemini_llm.count_tokens(rag_prompt).total_tokens
                    output_tokens = response.usage_metadata.candidates_token_count if response.usage_metadata else self.gemini_llm.count_tokens(answer_text).total_tokens
                    llm_cost_this_call = ((input_tokens / 1000) * GEMINI_FLASH_INPUT_COST_PER_1K_TOKENS) + \
                           ((output_tokens / 1000) * GEMINI_FLASH_OUTPUT_COST_PER_1K_TOKENS)
                    token_usage_output["llm_input_tokens"] = input_tokens
                    token_usage_output["llm_output_tokens"] = output_tokens
                    token_usage_output["cost"] = embedding_cost_this_call + llm_cost_this_call
                    self.api_total_cost[api_type_generate] += llm_cost_this_call # Update global service cost
                    self.logger.info(f"Gemini RAG call cost: ${llm_cost_this_call:.6f} for {input_tokens}+{output_tokens} tokens.")
                    if self.db_service:
                        self.db_service.add_api_metric(
                            session_id=session_id, api_type=api_type_generate, model_name=self.gemini_llm.model_name,
                            latency_ms=latency_generate * 1000, cost=llm_cost_this_call,
                            tokens_input=input_tokens, tokens_output=output_tokens
                        )
            else: # Groq RAG Path
                api_type_generate = 'groq_generate'
                groq_model_id = model.split('/')[-1] if '/' in model else model
                response = self.groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": rag_prompt}], model=groq_model_id)
                latency_generate = time.perf_counter() - start_time_generate
                self.api_call_counts[api_type_generate] += 1
                self.api_total_latency[api_type_generate] += latency_generate
                self.logger.info(f"Groq ({groq_model_id}) RAG response generation took {latency_generate:.4f}s.")
                
                answer_text = response.choices[0].message.content
                input_tokens = response.usage.prompt_tokens if hasattr(response, 'usage') and response.usage else len(rag_prompt) // 4
                output_tokens = response.usage.completion_tokens if hasattr(response, 'usage') and response.usage else len(answer_text) // 4

                if 'llama3-8b' in groq_model_id:
                    input_cost_rate, output_cost_rate = GROQ_LLAMA3_8B_INPUT_COST_PER_1K_TOKENS, GROQ_LLAMA3_8B_OUTPUT_COST_PER_1K_TOKENS
                elif 'qwen-32b' in groq_model_id:
                    input_cost_rate, output_cost_rate = GROQ_QWEN_32B_INPUT_COST_PER_1K_TOKENS, GROQ_QWEN_32B_OUTPUT_COST_PER_1K_TOKENS
                else:
                    input_cost_rate, output_cost_rate = GROQ_GENERIC_INPUT_COST_PER_1K_TOKENS, GROQ_GENERIC_OUTPUT_COST_PER_1K_TOKENS
                
                llm_cost_this_call = ((input_tokens / 1000) * input_cost_rate) + ((output_tokens / 1000) * output_cost_rate)
                token_usage_output["llm_input_tokens"] = input_tokens
                token_usage_output["llm_output_tokens"] = output_tokens
                token_usage_output["cost"] = embedding_cost_this_call + llm_cost_this_call
                self.api_total_cost[api_type_generate] += llm_cost_this_call # Update global service cost
                self.logger.info(f"Groq ({groq_model_id}) RAG call cost: ${llm_cost_this_call:.6f} for ~{input_tokens}+{output_tokens} tokens.")
                if self.db_service:
                    self.db_service.add_api_metric(
                        session_id=session_id, api_type=api_type_generate, model_name=groq_model_id,
                        latency_ms=latency_generate * 1000, cost=llm_cost_this_call,
                        tokens_input=input_tokens, tokens_output=output_tokens
                    )
            
            token_usage_output['llm_input_tokens'] = input_tokens
            token_usage_output['llm_output_tokens'] = output_tokens
            token_usage_output['cost'] = embedding_cost_this_call + llm_cost_this_call

            # Prepare sources for the return value and for DB logging
            # Assumes chunk is a dict and has a 'source' key for the filename and 'content' for preview
            sources_for_output = [
                {"source_name": chunk.get('source', 'Unknown Source'), "content_preview": chunk.get('content', '')[:150] + "..."}
                for chunk in relevant_chunks
            ]
            
            if self.db_service:
                assistant_message_id = self.db_service.add_chat_message(
                    session_id=session_id, 
                    sender='assistant', 
                    message=answer_text, 
                    model_used=model # Log the full model identifier, e.g., gemini-1.5-flash-latest or groq/llama3-8b-8192
                )

            self.logger.info(f"FINAL RAG_SERVICE TOKEN USAGE (RAG path): {token_usage_output}")
            return {"answer": answer_text, "sources": sources_for_output, "assistant_message_id": assistant_message_id, "token_usage": token_usage_output}

        except Exception as e:
            self.logger.error(f"Error in answer_query for query '{current_query[:100]}...': {e}", exc_info=True)
            error_answer_text = "I encountered an error while trying to process your request. Please try again later."
            # Ensure cost in token_usage_output reflects what happened before the error
            token_usage_output['cost'] = embedding_cost_this_call + llm_cost_this_call # llm_cost_this_call might be 0 if error was before LLM
            
            if self.db_service:
                 self.db_service.add_api_metric(
                    session_id=session_id, 
                    api_type=model.split('/')[0] if '/' in model else model.split('-')[0], # e.g. 'gemini' or 'groq'
                    model_name=model,
                    latency_ms=0, # Latency might not be fully captured
                    cost=0, # Cost for this specific failed operation part is 0, but overall cost in token_usage_output is preserved
                    tokens_input=token_usage_output.get('llm_input_tokens',0),
                    tokens_output=0, 
                    error_occurred=True, 
                    error_message=str(e)
                )
                 try:
                     assistant_message_id = self.db_service.add_chat_message(session_id=session_id, sender='assistant', message=error_answer_text, model_used=model + "_error")
                 except Exception as db_e:
                     self.logger.error(f"Failed to log error chat message to DB: {db_e}", exc_info=True)
            
            return {"answer": error_answer_text, "sources": [], "assistant_message_id": assistant_message_id, "token_usage": token_usage_output}

# Example usage:
if __name__ == '__main__':
    # This example usage will not work directly anymore as RAGService now requires a DatabaseService instance.
    # You would need to instantiate DatabaseService first and pass it to RAGService.
    # For example:
    # from backend.database_service import DatabaseService
    # logging.basicConfig(level=logging.INFO)
    # db_service_instance = DatabaseService(db_path='../data/test_rag_integration.db')
    # rag_service = RAGService(db_service=db_service_instance)
    # test_session_id = "test_session_001"
    # test_query = "How do payments work for freelancers?"
    # result = rag_service.answer_query(session_id=test_session_id, current_query=test_query, model='llama3-8b-8192')
    print("RAGService example usage needs to be updated to include DatabaseService and session_id.")
    # print("\n--- Query Result ---")
    # print(f"Question: {test_query}")
    # print(f"Answer: {result['answer']}")
    # print(f"Sources: {result['sources']}")

    # test_query_2 = "What is the best way to find a good restaurant?"
    # result_2 = rag_service.answer_query(session_id=test_session_id, current_query=test_query_2, model='gemini-1.5-flash-latest')
    # print("\n--- Query Result ---")
    # print(f"Question: {test_query_2}")
    # print(f"Answer: {result_2['answer']}")
    # print(f"Sources: {result_2['sources']}")

    print("\n--- Query Result ---")
    print(f"Question: {test_query}")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")

    test_query_2 = "What is the best way to find a good restaurant?"
    result_2 = rag_service.answer_query(test_query_2, model='gemini-1.5-flash-latest')
    print("\n--- Query Result ---")
    print(f"Question: {test_query_2}")
    print(f"Answer: {result_2['answer']}")
    print(f"Sources: {result_2['sources']}")
