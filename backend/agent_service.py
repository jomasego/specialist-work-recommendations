import os
import time
import logging
from typing import Dict, Any, List, Tuple
import google.generativeai as genai


from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from .rag_service import RAGService
from .database_service import DatabaseService
from .rag_service import (
    GEMINI_FLASH_INPUT_COST_PER_1K_TOKENS,
    GEMINI_FLASH_OUTPUT_COST_PER_1K_TOKENS,
    GROQ_MIXTRAL_8X7B_INPUT_COST_PER_1K_TOKENS,
    GROQ_MIXTRAL_8X7B_OUTPUT_COST_PER_1K_TOKENS
    # Note: GROQ_LLAMA3_8B constants are also in rag_service but not directly used by AgentService's own LLM calls.
)

# Model names
CLIENT_INTERACTION_MODEL_GEMINI = "gemini-1.5-flash-latest"
PLATFORM_MANAGER_MODEL_MIXTRAL = "mixtral-8x7b-32768"
RECRUITER_MODEL_LLAMA3 = "llama3-8b-8192" # As used in RAGService

class AgentService:
    def __init__(self, rag_service: RAGService, db_service: DatabaseService = None):
        self.logger = logging.getLogger(__name__)
        self.rag_service = rag_service
        self.db_service = db_service
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")

        if not self.groq_api_key:
            self.logger.warning("GROQ_API_KEY not found in environment variables.")
        if not self.gemini_api_key:
            self.logger.warning("GEMINI_API_KEY not found in environment variables.")

        self.client_interaction_llm = None
        self.platform_manager_llm = None
        self._initialize_llms()

        self.api_call_counts = {
            f"{CLIENT_INTERACTION_MODEL_GEMINI}_agent_client_interaction": 0,
            f"{PLATFORM_MANAGER_MODEL_MIXTRAL}_agent_platform_manager": 0,
        }
        self.api_total_latency = {
            f"{CLIENT_INTERACTION_MODEL_GEMINI}_agent_client_interaction": 0.0,
            f"{PLATFORM_MANAGER_MODEL_MIXTRAL}_agent_platform_manager": 0.0,
        }
        self.api_total_cost = {
            f"{CLIENT_INTERACTION_MODEL_GEMINI}_agent_client_interaction": 0.0,
            f"{PLATFORM_MANAGER_MODEL_MIXTRAL}_agent_platform_manager": 0.0,
        }

    def _initialize_llms(self):
        try:
            if self.gemini_api_key:
                genai.configure(api_key=self.gemini_api_key)
                # System instructions are passed during model initialization for the native client
                system_instruction = "You are a helpful assistant interacting with a user looking for freelancers. Your role is to understand their needs clearly. Rephrase their request if necessary to be a clear instruction for a recruiter agent. Focus on skills, project type, and experience needed."
                self.client_interaction_llm = genai.GenerativeModel(
                    CLIENT_INTERACTION_MODEL_GEMINI,
                    system_instruction=system_instruction
                )
                self.logger.info(f"{CLIENT_INTERACTION_MODEL_GEMINI} LLM for Client Interaction Agent initialized using native SDK.")
            else:
                self.logger.error(f"Cannot initialize {CLIENT_INTERACTION_MODEL_GEMINI} LLM: GEMINI_API_KEY not set.")

            if self.groq_api_key:
                self.platform_manager_llm = ChatGroq(
                    temperature=0.2, # Balanced for review
                    groq_api_key=self.groq_api_key,
                    model_name=PLATFORM_MANAGER_MODEL_MIXTRAL
                )
                self.logger.info(f"{PLATFORM_MANAGER_MODEL_MIXTRAL} LLM for Platform Manager Agent initialized.")
            else:
                self.logger.error(f"Cannot initialize {PLATFORM_MANAGER_MODEL_QWEN} LLM: GROQ_API_KEY not set.")

        except Exception as e:
            self.logger.error(f"Error initializing LLMs for AgentService: {e}", exc_info=True)
            # Not raising here to allow partial functionality if one key is missing, but logging error.

    def _estimate_token_cost(self, text: str, model_type: str, direction: str) -> Tuple[int, float]:
        """Estimates token count and cost."""
        tokens = 0
        cost = 0.0
        if model_type == "gemini_flash":
            # Gemini has its own tokenizer, but for cost, LangChain might not expose it directly for simple strings.
            # Using char/4 as a rough fallback if direct token count is hard.
            # However, ChatGoogleGenerativeAi.get_num_tokens can be used.
            try:
                # Use the native Google AI SDK's token counter for Gemini
                tokens = self.client_interaction_llm.count_tokens(text).total_tokens if self.client_interaction_llm else len(text) // 4
            except Exception:
                 tokens = len(text) // 4 # Fallback
            input_cost_rate = GEMINI_FLASH_INPUT_COST_PER_1K_TOKENS
            output_cost_rate = GEMINI_FLASH_OUTPUT_COST_PER_1K_TOKENS
            cost = (tokens / 1000) * (input_cost_rate if direction == "input" else output_cost_rate)
        elif model_type == "mixtral_8x7b":
            tokens = len(text) // 4 # Groq approximation
            input_cost_rate = GROQ_MIXTRAL_8X7B_INPUT_COST_PER_1K_TOKENS
            output_cost_rate = GROQ_MIXTRAL_8X7B_OUTPUT_COST_PER_1K_TOKENS
            cost = (tokens / 1000) * (input_cost_rate if direction == "input" else output_cost_rate)
        return tokens, cost

    def _call_llm_with_tracking(self, llm, prompt_messages: List, model_name_key: str, model_type_for_cost: str, agent_name_suffix: str) -> str:
        if not llm:
            self.logger.error(f"LLM for {agent_name_suffix} not initialized. Skipping call.")
            return "Error: LLM not available for this agent."
        
        start_time = time.perf_counter()
        # Construct input text from messages for token estimation
        input_text_for_estimation = " ".join([msg.content for msg in prompt_messages if hasattr(msg, 'content')])
        
        # Check if this is the native Gemini model or a LangChain model
        if model_name_key == CLIENT_INTERACTION_MODEL_GEMINI and self.client_interaction_llm and hasattr(llm, "generate_content"):
            # Native google-generativeai call
            # System prompt is handled at initialization. Extract human message content.
            human_prompt = "\n".join([msg.content for msg in prompt_messages if isinstance(msg, HumanMessage)])
            
            if not human_prompt:
                self.logger.error("No HumanMessage found for native Gemini call.")
                return "Error: Invalid prompt for Gemini agent."

            response = llm.generate_content(human_prompt)
            latency = time.perf_counter() - start_time
            response_text = response.text
        else:
            # LangChain .invoke() call for other models like Groq
            response = llm.invoke(prompt_messages)
            latency = time.perf_counter() - start_time
            response_text = response.content if hasattr(response, 'content') else str(response)

        input_tokens, input_cost = self._estimate_token_cost(input_text_for_estimation, model_type_for_cost, "input")
        output_tokens, output_cost = self._estimate_token_cost(response_text, model_type_for_cost, "output")
        total_call_cost = input_cost + output_cost

        metric_key = f"{model_name_key}_agent_{agent_name_suffix}"
        self.api_call_counts[metric_key] = self.api_call_counts.get(metric_key, 0) + 1
        self.api_total_latency[metric_key] = self.api_total_latency.get(metric_key, 0) + latency
        self.api_total_cost[metric_key] = self.api_total_cost.get(metric_key, 0) + total_call_cost

        self.logger.info(f"Agent {agent_name_suffix} ({model_name_key}) call took {latency:.4f}s. Cost: ${total_call_cost:.6f}. Tokens In/Out: {input_tokens}/{output_tokens}")
        if self.db_service:
            self.db_service.add_api_metric(
                session_id=None, # Session ID might not be directly available here, or pass it down
                api_type=f"agent_{agent_name_suffix}",
                model_name=model_name_key,
                latency_ms=latency * 1000,
                cost=total_call_cost,
                tokens_input=input_tokens,
                tokens_output=output_tokens
            )
        return response_text

    def process_user_query(self, user_query: str, session_id: str, chat_history: List[Dict[str, str]], model: str) -> Dict[str, Any]:
        self.logger.info(f"AgentService processing query for session {session_id}: {user_query}")
        agent_log = []

        # 1. Client Interaction Agent (Gemini 1.5 Flash)
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
        # The system prompt is now part of the model initialization for the native Gemini model.
        # We only need to send the HumanMessage.
        client_interaction_prompt_messages = [
            HumanMessage(content=f"Chat History:\n{history_str}\n\nUser's current query: {user_query}\n\nBased on this, what is the core recruitment task for the recruiter agent?")
        ]
        structured_request_for_recruiter = self._call_llm_with_tracking(
            self.client_interaction_llm, 
            client_interaction_prompt_messages, 
            CLIENT_INTERACTION_MODEL_GEMINI, 
            "gemini_flash", 
            "client_interaction"
        )
        agent_log.append(f"ClientInteractionAgent (Gemini Flash) refined query to: {structured_request_for_recruiter[:200]}...")

        # 2. Objective Recruiter Agent (Llama3 via RAGService)
        # 2. Objective Recruiter Agent (User-selected model via RAGService)
        # Use the model selected by the user for the recruiter agent
        recruiter_model = model if model else RECRUITER_MODEL_LLAMA3
        self.logger.info(f"Passing to Recruiter Agent (RAGService with {recruiter_model}) with request: {structured_request_for_recruiter}")
        recruiter_findings = self.rag_service.answer_query(
            query=structured_request_for_recruiter,
            session_id=session_id,
            model=recruiter_model,
            chat_history=[] # Recruiter gets focused task, not full history again unless designed so
        )
        agent_log.append(f"ObjectiveRecruiterAgent (Llama3 via RAGService) found: {recruiter_findings['answer'][:200]}...")

        # 3. Makers Platform Manager Agent (Qwen-32B)
        platform_manager_prompt_messages = [
            SystemMessage(content="You are a platform manager. Review the recruiter's findings. Ensure they align with platform policies (e.g., fairness, diversity, promoting new talent if applicable) and quality standards. You can approve, or suggest modifications to the recruiter's answer. For now, focus on quality and clarity."),
            HumanMessage(content=f"Recruiter's proposed answer for the client (based on query '{structured_request_for_recruiter}'):\n\n{recruiter_findings['answer']}\n\nReview this. Is it good to send to the client? If so, repeat it. If not, refine it or state concerns.")
        ]
        final_answer_from_manager = self._call_llm_with_tracking(
            self.platform_manager_llm, 
            platform_manager_prompt_messages, 
            PLATFORM_MANAGER_MODEL_MIXTRAL, 
            "mixtral_8x7b", 
            "platform_manager"
        )
        agent_log.append(f"PlatformManagerAgent (Mixtral) reviewed and finalized answer: {final_answer_from_manager[:200]}...")

        return {
            "answer": final_answer_from_manager,
            "sources": recruiter_findings['sources'], # Sources from Recruiter/RAG
            "assistant_message_id": recruiter_findings.get('assistant_message_id'), # From RAGService DB log
            "agent_log": agent_log,
            "refined_query_for_recommendations": structured_request_for_recruiter
        }

    def get_api_stats(self) -> Dict[str, Any]:
        # Combine stats from AgentService's own calls and RAGService's calls
        rag_stats = self.rag_service.get_api_stats()
        combined_stats = {
            "api_call_counts": {**self.api_call_counts, **rag_stats.get('api_call_counts', {})},
            "api_total_latency": {**self.api_total_latency, **rag_stats.get('api_total_latency', {})},
            "api_total_cost": {**self.api_total_cost, **rag_stats.get('api_total_cost', {})},
        }
        return combined_stats
