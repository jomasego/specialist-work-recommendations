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

from database_service import DatabaseService # Assuming database_service.py is in backend directory

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
        start_time = time.perf_counter()
        try:
            # Cost calculation for embedding
            token_count_response = genai.count_tokens(model=self.embedding_model_name, contents=query)
            token_count = token_count_response.total_tokens
            cost = (token_count / 1000) * EMBEDDING_COST_PER_1K_TOKENS
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
            return result['embedding']
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
            raise RAGServiceError(f"Error generating query embedding: {e}") from e

    def search_knowledge_base(self, query: str, k: int = 5, session_id: Optional[str] = None):
        """Searches the vector store for the most relevant document chunks."""
        try:
            query_embedding = self._get_query_embedding(query, session_id=session_id)
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

    def _is_query_safe_groq(self, query: str, session_id: Optional[str] = None) -> bool:
        """
        Uses Llama Guard 4 via Groq to check if a query is safe.
        """
        self.logger.info(f"Checking query safety with Llama Guard 4 for: '{query[:100]}...'")
        try:
            # The prompt format for Llama Guard is specific.
            llama_guard_prompt = f"""[INST] Task: Check if there is unsafe content in 'User' messages in conversations. Categories of unsafe content are: violence, sexual content, hate speech, harassment, self-harm, criminal planning, guns, and illicit drugs.

<BEGIN CONVERSATION>

User: {query}

<END CONVERSATION>

Provide your safety assessment for the User message in the above conversation:
- Your determination MUST be either "safe" or "unsafe".
- If you deem the message unsafe, you MUST provide the category of unsafe content.
[/INST]"""

            start_time_guard = time.perf_counter()
            chat_completion = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": llama_guard_prompt}],
                model="meta-llama/Llama-Guard-4-12B",
                temperature=0.0,
            )
            latency_guard = time.perf_counter() - start_time_guard
            self.api_call_counts['llama_guard'] += 1
            self.api_total_latency['llama_guard'] += latency_guard
            self.logger.info(f"Llama Guard 4 check took {latency_guard:.4f}s.")
            response_text = chat_completion.choices[0].message.content.lower()
            
            # Cost calculation for Llama Guard (Groq)
            # NOTE: Groq SDK does not provide a direct tokenizer. Using character count as an approximation (4 chars ~ 1 token).
            # This is a rough estimate and should be refined if a Groq tokenizer becomes available or pricing changes.
            input_tokens_approx = len(llama_guard_prompt) // 4
            output_tokens_approx = len(response_text) // 4
            # Llama Guard is based on Llama, so we'll use Llama3-8B pricing as a proxy.
            # This is an approximation.
            input_cost_per_1k = GROQ_LLAMA3_8B_INPUT_COST_PER_1K_TOKENS
            output_cost_per_1k = GROQ_LLAMA3_8B_OUTPUT_COST_PER_1K_TOKENS
            cost = ((input_tokens_approx / 1000) * input_cost_per_1k) + \
                   ((output_tokens_approx / 1000) * output_cost_per_1k)
            self.api_total_cost['llama_guard'] += cost
            self.logger.info(f"Llama Guard (Groq) call cost: ${cost:.6f} for ~{input_tokens_approx}+{output_tokens_approx} tokens (approx).")

            self.logger.debug(f"Llama Guard 4 response: '{response_text}'")

            if "unsafe" in response_text:
                self.logger.warning(f"Query flagged as unsafe by Llama Guard 4: '{query}'")
                return False
            
            self.logger.info("Query deemed safe by Llama Guard 4.")
            if self.db_service:
                self.db_service.add_api_metric(
                    session_id=session_id,
                    api_type='llama_guard',
                    model_name="meta-llama/Llama-Guard-4-12B",
                    latency_ms=latency_guard * 1000,
                    cost=cost,
                    tokens_input=input_tokens_approx,
                    tokens_output=output_tokens_approx,
                    error_occurred=False
                )
            return True
        except Exception as e:
            self.logger.error(f"Error during Llama Guard 4 safety check: {e}", exc_info=True)
            if self.db_service:
                self.db_service.add_api_metric(
                    session_id=session_id,
                    api_type='llama_guard',
                    model_name="meta-llama/Llama-Guard-4-12B",
                    latency_ms= (time.perf_counter() - start_time_guard) * 1000 if 'start_time_guard' in locals() else 0,
                    cost=0, # Failed call
                    error_occurred=True,
                    error_message=str(e)
                )
            # Fail-safe: if the moderation model fails, we assume the query is unsafe.
            return False

    def _compress_chat_history(self, chat_history, max_tokens=MAX_HISTORY_TOKENS):
        """Compresses chat history to stay within a token limit, prioritizing recent messages."""
        if not chat_history:
            return []

        compressed_history = []
        current_tokens = 0
        # Simple token estimation: 1 token ~ 4 chars
        # Iterate in reverse (newest first)
        for message in reversed(chat_history):
            message_content = message.get('content', '')
            # Estimate tokens for current message (role + content)
            # Add some overhead for role and formatting
            message_tokens = (len(message_content) + len(message.get('role', '')) + 10) // 4 
            if current_tokens + message_tokens <= max_tokens:
                compressed_history.append(message)
                current_tokens += message_tokens
            else:
                # Stop if adding this message exceeds the limit
                self.logger.info(f"Chat history truncated. Original messages: {len(chat_history)}, Compressed: {len(compressed_history)}")
                break 
        return list(reversed(compressed_history)) # Return in original order

    def answer_query(self, session_id: str, current_query: str, chat_history: Optional[list] = None, model: str = 'gemini-1.5-flash-latest') -> Dict[str, Any]:
        """Answers a user query using the RAG pipeline, considering chat history and safety."""
        assistant_message_id: Optional[int] = None # Initialize

        # Log user query
        if self.db_service:
            # We don't strictly need the user_message_id for the return value, but good practice to log.
            _user_message_id = self.db_service.add_chat_message(
                session_id=session_id,
                sender='user',
                message=current_query
            )

        self.logger.info(f"Received query: '{current_query[:100]}...', Model: {model}")

        # --- Content Moderation Pre-check (for Groq) ---
        if not model.startswith('gemini'):
            if not self._is_query_safe_groq(current_query, session_id=session_id):
                self.logger.warning(f"Groq query blocked by Llama Guard: '{current_query}'")
                answer_text = "I'm sorry, but your request could not be processed due to our safety guidelines."
                response_data = {"answer": answer_text, "sources": []}
                if self.db_service:
                    assistant_message_id = self.db_service.add_chat_message(session_id=session_id, sender='assistant', message=answer_text, model_used='llama_guard_blocked')
                response_data['assistant_message_id'] = assistant_message_id
                return response_data

        relevant_chunks = self.search_knowledge_base(current_query, session_id=session_id)

        # --- Handle No Relevant Chunks ---
        if not relevant_chunks:
            self.logger.info("No relevant information found for the query.")
            history_context = "".join([f"{msg['role'].capitalize()}: {msg['content']}\n" for msg in chat_history]) if chat_history else ""
            no_info_prompt = f"""You are an expert assistant for the Makers platform.
The user asked: "{current_query}"
Conversation history:
{history_context}
You could not find specific documents for this question. Acknowledge this and respond helpfully based on the conversation, or suggest rephrasing. Do not invent information.
ANSWER:"""
            try:
                start_time_no_info = time.perf_counter()
                if model.startswith('gemini'):
                    response = self.gemini_llm.generate_content(no_info_prompt)
                    latency_no_info = time.perf_counter() - start_time_no_info
                    self.api_call_counts['gemini_no_info'] += 1
                    self.api_total_latency['gemini_no_info'] += latency_no_info
                    self.logger.info(f"Gemini no-info response generation took {latency_no_info:.4f}s.")
                    if response.prompt_feedback.block_reason:
                        self.logger.warning(f"Gemini no-info response blocked. Reason: {response.prompt_feedback.block_reason.name}")
                        if self.db_service:
                            self.db_service.add_api_metric(
                                session_id=session_id,
                                api_type='gemini_no_info',
                                model_name=self.gemini_llm.model_name,
                                latency_ms=latency_no_info * 1000,
                                cost=0, # Or input cost only
                                tokens_input=self.gemini_llm.count_tokens(no_info_prompt).total_tokens,
                                error_occurred=True,
                                error_message=f"Blocked: {response.prompt_feedback.block_reason.name}"
                            )
                            blocked_answer_text = "I'm sorry, but I cannot process this request due to safety guidelines."
                            if self.db_service:
                                assistant_message_id = self.db_service.add_chat_message(session_id=session_id, sender='assistant', message=blocked_answer_text, model_used=self.gemini_llm.model_name + "_blocked")
                            return {"answer": blocked_answer_text, "sources": [], "assistant_message_id": assistant_message_id}
                    answer = response.text
                    # Cost calculation for Gemini no-info
                    input_tokens = self.gemini_llm.count_tokens(no_info_prompt).total_tokens
                    output_tokens = self.gemini_llm.count_tokens(answer).total_tokens
                    cost = ((input_tokens / 1000) * GEMINI_FLASH_INPUT_COST_PER_1K_TOKENS) + \
                           ((output_tokens / 1000) * GEMINI_FLASH_OUTPUT_COST_PER_1K_TOKENS)
                    self.api_total_cost['gemini_no_info'] += cost
                    self.logger.info(f"Gemini no-info call cost: ${cost:.6f} for {input_tokens}+{output_tokens} tokens.")
                    if self.db_service:
                        self.db_service.add_api_metric(
                            session_id=session_id,
                            api_type='gemini_no_info',
                            model_name=self.gemini_llm.model_name,
                            latency_ms=latency_no_info * 1000,
                            cost=cost,
                            tokens_input=input_tokens,
                            tokens_output=output_tokens
                        )
                    if self.db_service:
                        assistant_message_id = self.db_service.add_chat_message(session_id=session_id, sender='assistant', message=answer, model_used=self.gemini_llm.model_name + "_no_info")
                else: # Groq no-info
                    response = self.groq_client.chat.completions.create(
                        messages=[{"role": "user", "content": no_info_prompt}],
                        model=model,
                    )
                    latency_no_info = time.perf_counter() - start_time_no_info
                    self.api_call_counts['groq_no_info'] += 1
                    self.api_total_latency['groq_no_info'] += latency_no_info
                    self.logger.info(f"Groq ({model}) no-info response generation took {latency_no_info:.4f}s.")
                    answer = response.choices[0].message.content
                    # Cost calculation for Groq no-info
                    input_tokens_approx = len(no_info_prompt) // 4
                    output_tokens_approx = len(answer) // 4
                    if 'llama3-8b' in model:
                        input_cost_per_1k = GROQ_LLAMA3_8B_INPUT_COST_PER_1K_TOKENS
                        output_cost_per_1k = GROQ_LLAMA3_8B_OUTPUT_COST_PER_1K_TOKENS
                    elif 'qwen-qwq-32b' in model:
                        input_cost_per_1k = GROQ_QWEN_32B_INPUT_COST_PER_1K_TOKENS
                        output_cost_per_1k = GROQ_QWEN_32B_OUTPUT_COST_PER_1K_TOKENS
                    else:
                        input_cost_per_1k, output_cost_per_1k = 0.0, 0.0
                    cost = ((input_tokens_approx / 1000) * input_cost_per_1k) + \
                           ((output_tokens_approx / 1000) * output_cost_per_1k)
                    self.api_total_cost['groq_no_info'] += cost
                    self.logger.info(f"Groq ({model}) no-info call cost: ${cost:.6f} for ~{input_tokens_approx}+{output_tokens_approx} tokens (approx).")
                    if self.db_service:
                        self.db_service.add_api_metric(
                            session_id=session_id,
                            api_type='groq_no_info',
                            model_name=model,
                            latency_ms=latency_no_info * 1000,
                            cost=cost,
                            tokens_input=input_tokens_approx,
                            tokens_output=output_tokens_approx
                        )
                    if self.db_service:
                        assistant_message_id = self.db_service.add_chat_message(session_id=session_id, sender='assistant', message=answer, model_used=model + "_no_info")
                response_data = {"answer": answer, "sources": [], "assistant_message_id": assistant_message_id}
                self.logger.info(f"No-info response: {answer[:100]}...")
                return response_data
            except Exception as e:
                self.logger.error(f"Error generating no-info response: {e}", exc_info=True)
                if self.db_service:
                    api_type_on_error = 'gemini_no_info' if model.startswith('gemini') else 'groq_no_info'
                    model_name_on_error = self.gemini_llm.model_name if model.startswith('gemini') else model
                    self.db_service.add_api_metric(
                        session_id=session_id,
                        api_type=api_type_on_error,
                        model_name=model_name_on_error,
                        latency_ms=(time.perf_counter() - start_time_no_info) * 1000 if 'start_time_no_info' in locals() else 0,
                        cost=0,
                        error_occurred=True,
                        error_message=str(e)
                    )
                    self.db_service.add_chat_message(session_id=session_id, sender='assistant', message="I'm sorry, but I had trouble generating a response.", model_used=model_name_on_error + "_error")
                return {"answer": "I'm sorry, but I had trouble generating a response.", "sources": []}

        # --- Generate Response with Context ---
        sources = sorted(list(set([chunk['metadata']['doc_name'] for chunk in relevant_chunks])))
        context_str = "\n\n---\n\n".join([chunk['text'] for chunk in relevant_chunks])
        if len(context_str) > MAX_CONTEXT_CHARS:
            self.logger.info(f"Truncating document context from {len(context_str)} to {MAX_CONTEXT_CHARS} chars.")
            context_str = context_str[:MAX_CONTEXT_CHARS] + "... [truncated]"

        compressed_chat_history = self._compress_chat_history(chat_history)
        history_for_prompt = "".join([f"{msg['role'].capitalize()}: {msg['content']}\n" for msg in compressed_chat_history]) if compressed_chat_history else ""

        prompt = f"""You are an expert assistant for the Makers platform.\nFollow these examples to structure your answer, citing sources when used:\n\n--- EXAMPLE 1 ---_BEGIN_EXAMPLE_USER_QUESTION_How do I get paid?_END_EXAMPLE_USER_QUESTION_\n_BEGIN_EXAMPLE_CONTEXT_Payments are processed via Stripe every Friday. Ensure your bank details are up to date in your profile. (Source: payments_guide.md)_END_EXAMPLE_CONTEXT_\n_BEGIN_EXAMPLE_ANSWER_Payments are processed via Stripe every Friday. You can ensure your bank details are up to date in your profile. (Source: payments_guide.md)_END_EXAMPLE_ANSWER_\n--- END EXAMPLE 1 ---\n\n--- EXAMPLE 2 ---_BEGIN_EXAMPLE_USER_QUESTION_What are the platform fees?_END_EXAMPLE_USER_QUESTION_\n_BEGIN_EXAMPLE_CONTEXT_Our platform charges a 10% service fee on all completed projects. This fee is automatically deducted. (Source: fee_structure.md)_END_EXAMPLE_CONTEXT_\n_BEGIN_EXAMPLE_ANSWER_The platform charges a 10% service fee on all completed projects, which is automatically deducted. (Source: fee_structure.md)_END_EXAMPLE_ANSWER_\n--- END EXAMPLE 2 ---\n\nNow, answer the user's current question based on the provided context and conversation history.\nIf the context is irrelevant, state that you don't have information for this specific question.\nBe clear, concise, and helpful. Cite sources if you use them (e.g., 'According to 01_getting_started.md...').\n\nCONVERSATION HISTORY:\n{history_for_prompt}\n\nCONTEXT FROM KNOWLEDGE BASE:\n{context_str}\n\nUSER'S CURRENT QUESTION: {current_query}\nYOUR ANSWER:""" 

        try:
            self.logger.info(f"Generating response from LLM: {model}...")
            start_time_llm = time.perf_counter()
            if model.startswith('gemini'):
                response = self.gemini_llm.generate_content(prompt)
                latency_llm = time.perf_counter() - start_time_llm
                self.api_call_counts['gemini_generate'] += 1
                self.api_total_latency['gemini_generate'] += latency_llm
                self.logger.info(f"Gemini response generation took {latency_llm:.4f}s.")
                if response.prompt_feedback.block_reason:
                    self.logger.warning(f"Gemini response blocked. Reason: {response.prompt_feedback.block_reason.name}")
                    if self.db_service:
                        self.db_service.add_api_metric(
                            session_id=session_id,
                            api_type='gemini_generate',
                            model_name=self.gemini_llm.model_name,
                            latency_ms=latency_llm * 1000,
                            cost=0, # Or input cost only
                            tokens_input=self.gemini_llm.count_tokens(prompt).total_tokens,
                            error_occurred=True,
                            error_message=f"Blocked: {response.prompt_feedback.block_reason.name}"
                        )
                        blocked_answer_text = "I'm sorry, but I cannot process this request due to safety guidelines."
                        if self.db_service:
                            assistant_message_id = self.db_service.add_chat_message(session_id=session_id, sender='assistant', message=blocked_answer_text, model_used=self.gemini_llm.model_name + "_blocked")
                        return {"answer": blocked_answer_text, "sources": sources, "assistant_message_id": assistant_message_id}
                answer = response.text
                # Cost calculation for Gemini generate
                input_tokens = self.gemini_llm.count_tokens(prompt).total_tokens
                output_tokens = self.gemini_llm.count_tokens(answer).total_tokens
                cost = ((input_tokens / 1000) * GEMINI_FLASH_INPUT_COST_PER_1K_TOKENS) + \
                       ((output_tokens / 1000) * GEMINI_FLASH_OUTPUT_COST_PER_1K_TOKENS)
                self.api_total_cost['gemini_generate'] += cost
                self.logger.info(f"Gemini call cost: ${cost:.6f} for {input_tokens}+{output_tokens} tokens.")
                if self.db_service:
                    self.db_service.add_api_metric(
                        session_id=session_id,
                        api_type='gemini_generate',
                        model_name=self.gemini_llm.model_name,
                        latency_ms=latency_llm * 1000,
                        cost=cost,
                        tokens_input=input_tokens,
                        tokens_output=output_tokens
                    )
            else: # Assume Groq
                chat_completion = self.groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}], # Use the full prompt for Groq as well
                    model=model
                )
                latency_llm = time.perf_counter() - start_time_llm
                self.api_call_counts['groq_generate'] += 1
                self.api_total_latency['groq_generate'] += latency_llm
                self.logger.info(f"Groq response generation took {latency_llm:.4f}s.")
                answer = chat_completion.choices[0].message.content
                # Cost calculation for Groq generate (approximate)
                input_tokens_approx = len(prompt) // 4
                output_tokens_approx = len(answer) // 4
                if 'llama3-8b' in model:
                    input_cost_per_1k = GROQ_LLAMA3_8B_INPUT_COST_PER_1K_TOKENS
                    output_cost_per_1k = GROQ_LLAMA3_8B_OUTPUT_COST_PER_1K_TOKENS
                elif 'qwen-qwq-32b' in model:
                    input_cost_per_1k = GROQ_QWEN_32B_INPUT_COST_PER_1K_TOKENS
                    output_cost_per_1k = GROQ_QWEN_32B_OUTPUT_COST_PER_1K_TOKENS
                else:
                    self.logger.warning(f"No specific pricing for Groq model '{model}'. Using generic rate.")
                    input_cost_per_1k = GROQ_GENERIC_INPUT_COST_PER_1K_TOKENS
                    output_cost_per_1k = GROQ_GENERIC_OUTPUT_COST_PER_1K_TOKENS
                cost = ((input_tokens_approx / 1000) * input_cost_per_1k) + \
                       ((output_tokens_approx / 1000) * output_cost_per_1k)
                self.api_total_cost['groq_generate'] += cost
                self.logger.info(f"Groq call cost: ${cost:.6f} for ~{input_tokens_approx}+{output_tokens_approx} tokens (approx).")
                if self.db_service:
                    self.db_service.add_api_metric(
                        session_id=session_id,
                        api_type='groq_generate',
                        model_name=model,
                        latency_ms=latency_llm * 1000,
                        cost=cost,
                        tokens_input=input_tokens_approx,
                        tokens_output=output_tokens_approx
                    )

            self.logger.info(f"Successfully generated response from {model}.")
            # Log current metrics totals (optional, could be exposed via an endpoint later)
            self.logger.debug(f"API Call Counts: {self.api_call_counts}")
            self.logger.debug(f"API Total Latency (s): {self.api_total_latency}")
            self.logger.debug(f"API Total Cost ($): {self.api_total_cost}")
            if self.db_service:
                assistant_message_id = self.db_service.add_chat_message(session_id=session_id, sender='assistant', message=answer, model_used=model, relevant_chunks=relevant_chunks)
            response_data = {"answer": answer, "sources": sources, "assistant_message_id": assistant_message_id}
            self.logger.info(f"Final response prepared. Answer: {answer[:100]}... Sources: {len(sources)}, AssistantMsgID: {assistant_message_id}")
            return response_data
        except Exception as e:
            self.logger.error(f"Error during LLM content generation for query '{current_query[:100]}...': {e}", exc_info=True)
            final_answer = "I'm sorry, but I encountered an error while generating a response."
            if self.db_service:
                self.logger.error(f"Error in answer_query for query '{current_query[:100]}...': {e}", exc_info=True)
            error_answer_text = "I'm sorry, an unexpected error occurred."
            # Ensure db_service logging for the error if an API call failed before it could log itself
            # (Most API calls now log their own errors, but this is a fallback)
            if self.db_service:
                self.db_service.add_api_metric(
                    session_id=session_id,
                    api_type=model, # General model type as context
                    model_name=model,
                    latency_ms=0,
                    cost=0,
                    error_occurred=True,
                    error_message=str(e)
                )
                assistant_message_id = self.db_service.add_chat_message(session_id=session_id, sender='assistant', message=error_answer_text, model_used=model + "_error")
            return {"answer": error_answer_text, "sources": [], "assistant_message_id": assistant_message_id}

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
