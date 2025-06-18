import json
import os
import re
import traceback
from typing import TypedDict, Annotated, List, Optional, Tuple
import operator
from dotenv import load_dotenv

# Langchain and Langgraph imports
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate # Added import
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI # Added for Gemini Pro
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage # Corrected AIMessage import and consolidated others

# Import services
from .database_service import DatabaseService
from .rag_service import RAGService # RAGService now returns token usage
from .recommendation_service import RecommendationService

load_dotenv()

# --- Pricing Constants (mirrored from rag_service.py for direct use here if needed) ---
GEMINI_FLASH_INPUT_COST_PER_1K_TOKENS = 0.00035
GEMINI_FLASH_OUTPUT_COST_PER_1K_TOKENS = 0.00105
EMBEDDING_COST_PER_1K_TOKENS = 0.0001
# Groq Llama3 70b pricing
GROQ_LLAMA3_70B_INPUT_COST_PER_1K_TOKENS = 0.00059 
GROQ_LLAMA3_70B_OUTPUT_COST_PER_1K_TOKENS = 0.00079
# Groq Qwen 32b pricing (example, adjust if different)
GROQ_QWEN_32B_INPUT_COST_PER_1K_TOKENS = 0.0002 
GROQ_QWEN_32B_OUTPUT_COST_PER_1K_TOKENS = 0.0008
# Gemini 1.5 Pro pricing (example, verify if switching back)
GEMINI_1_5_PRO_INPUT_COST_PER_1K_TOKENS = 0.0035
GEMINI_1_5_PRO_OUTPUT_COST_PER_1K_TOKENS = 0.0105


# --- Agent State ---
class AgentState(TypedDict):
    query: str
    chat_history: list
    research_findings: dict
    agent_outcomes: Annotated[List[str], operator.add]
    final_response: str
    escalation_topic: Optional[str]
    api_token_usage: dict  # Added for token/cost tracking. Stores detailed token counts and total cost.


# --- Service Initialization ---
try:
    db_service = DatabaseService()
    rag_service = RAGService(db_service=db_service)
    rs_service = RecommendationService()
    print("All services initialized successfully.")
except Exception as e:
    print(f"Error initializing services: {e}")
    db_service, rag_service, rs_service = None, None, None


# --- Costing Helper ---
def _update_token_usage(current_usage: dict, new_rag_usage: Optional[dict]) -> dict:
    """Updates the session's total token usage and cost with data from a RAG service call."""
    if not new_rag_usage:
        return current_usage

    updated_usage = current_usage.copy()

    # Get values from the new usage dict using the correct keys
    embedding_tokens = new_rag_usage.get('embedding_tokens', 0)
    input_tokens = new_rag_usage.get('llm_input_tokens', 0)
    output_tokens = new_rag_usage.get('llm_output_tokens', 0)
    model_name = new_rag_usage.get('model_name', '')
    call_cost = new_rag_usage.get('cost', 0.0)

    # Update token counts for detailed tracking
    updated_usage['embedding_tokens'] += embedding_tokens
    
    if 'gemini' in model_name:
        updated_usage['gemini_input_tokens'] += input_tokens
        updated_usage['gemini_output_tokens'] += output_tokens
    # RAG service returns model names like 'groq/llama3-8b-8192'
    elif 'groq' in model_name:
        updated_usage['groq_input_tokens'] += input_tokens
        updated_usage['groq_output_tokens'] += output_tokens
    
    # Add the pre-calculated cost from the RAG service directly to the total
    updated_usage['total_cost'] += call_cost

    print(f"DEBUG: RAG/LLM call cost: ${call_cost:.6f}. Session total cost: ${updated_usage['total_cost']:.6f}")
    return updated_usage


# --- Tools ---
@tool
def web_search_tool(query: str) -> str:
    """Searches the web for the given query. Use for general knowledge or recent events."""
    print(f"---TOOL: Web Search--- \nQuery: {query}")
    return f"Placeholder web search results for '{query}'."

@tool
def rag_and_freelancer_search_tool(query: str) -> dict:
    """Searches the internal knowledge base, freelancer database, and articles. Returns findings and embedding token usage."""
    print(f"---TOOL: RAG & Freelancer Search--- \nQuery: {query}")
    findings = {
        "freelancers": [],
        "articles": [],
        "kb_chunks": [],
        "token_usage": {'embedding_input_tokens': 0} # Initialize for this tool's specific usage
    }

    if rs_service:
        try:
            recommendations = rs_service.get_recommendations_for_query(query)
            findings["freelancers"] = recommendations.get("freelancers", [])
            findings["articles"] = recommendations.get("articles", [])
        except Exception as e:
            print(f"Error calling RecommendationService: {e}")

    if rag_service:
        try:
            # search_knowledge_base now returns (chunks, token_count)
            chunks, tokens = rag_service.search_knowledge_base(query, k=3, session_id="tool-rag-search-session")
            findings["kb_chunks"] = [chunk['content'] for chunk in chunks] # Store only content for simplicity in agent
            findings["token_usage"]['embedding_input_tokens'] = tokens
            print(f"RAG service (tool) found {len(chunks)} chunks, using {tokens} embedding tokens.")
        except Exception as e:
            print(f"Error calling RAGService.search_knowledge_base in tool: {e}")

    return findings


# --- Helper Function for Budget Parsing ---
def _parse_budget_from_query(query: str) -> Optional[Tuple[str, float, Optional[float]]]:
    """Parses a query to extract budget information."""
    query_lower = query.lower()
    price_match = re.search(r'\$?(\d+(\.\d{1,2})?)\b', query_lower)
    if not price_match: return None
    price = float(price_match.group(1))
    
    if "less than or equal to" in query_lower or "at most" in query_lower or "<=" in query: return ("less_equal", price, None)
    if "less than" in query_lower or "under" in query_lower or "<" in query: return ("less", price, None)
    if "more than or equal to" in query_lower or "at least" in query_lower or ">=" in query: return ("greater_equal", price, None)
    if "more than" in query_lower or "over" in query_lower or ">" in query: return ("greater", price, None)
    if "exactly" in query_lower or "equal to" in query_lower: return ("equal", price, None)
    if "around" in query_lower or "about" in query_lower: return ("range", price * 0.9, price * 1.1)
    # Fallback for just a price mentioned, implying max budget
    if price_match and not any(c in query_lower for c in ["more", "over", "greater", "least", "minimum", "above"]): return ("less_equal", price, None)
    return None


# --- Agent Logics ---

# Research Agent
# Note: The research_llm itself (llama3-70b) is not directly used in the bypass logic.
# If it were, its token usage would also need to be captured via Langchain callbacks or similar.
research_llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")
research_tools = [rag_and_freelancer_search_tool, web_search_tool]
# research_agent_runnable = research_llm.bind_tools(research_tools) # Not used in bypass

def research_agent_node(state: AgentState):
    print("---AGENT: Research--- ")
    outcomes = []
    # Safely get the token usage dictionary, providing a default if it's missing.
    current_token_usage = state.get('api_token_usage', {
        'embedding_tokens': 0,
        'gemini_input_tokens': 0,
        'gemini_output_tokens': 0,
        'groq_input_tokens': 0,
        'groq_output_tokens': 0,
        'total_cost': 0.0
    })
    
    try:
        print(f"DEBUG: Research Agent bypassing LLM, using direct tool call for query: {state['query']}")
        query_lower = state['query'].lower()
        
        if any(term in query_lower for term in ['freelancer', 'specialist', 'expert', 'developer', 'designer', 'consultant', 'makers', 'platform', 'payment', 'fee', 'rate', 'budget', 'cost']):
            # Invoke the tool and get its result, which includes its specific token_usage
            tool_result = rag_and_freelancer_search_tool.invoke(state['query'])
            outcomes.append("Research Agent used RAG & Freelancer Search tool.")
            # Update overall token usage with the tool's embedding token usage
            current_token_usage = _update_token_usage(current_token_usage, tool_result.get("token_usage"))
            research_data_for_state = tool_result # Pass full tool result to research_findings
        else: 
            web_search_results = web_search_tool.invoke(state['query'])
            research_data_for_state = {"web_search_results": web_search_results, "token_usage": {}} # No specific RAG tokens for web search
            outcomes.append("Research Agent used Web Search tool.")
            
        outcomes.append("Research completed successfully.")
        return {"agent_outcomes": outcomes, "research_findings": research_data_for_state, "api_token_usage": current_token_usage}
            
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"\n\nCRITICAL ERROR IN RESEARCH AGENT: {e}\n{error_trace}\n\n")
        return {"agent_outcomes": outcomes + [f"Research Agent failed: {str(e)}"], 
                "research_findings": {"error": f"Error during research: {str(e)}", "token_usage": {}},
                "api_token_usage": current_token_usage}

# Customer-Facing Agent
customer_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)

def customer_facing_agent_node(state: AgentState):
    """Generates the final response for the user using structured findings and RAGService."""
    print("---AGENT: Customer-Facing--- ")
    response_text = "I encountered an issue processing your request. Please try again."
    outcomes = ["Customer-Facing Agent received the query."]
    current_token_usage = state['api_token_usage']
    
    query = state.get("query", "")
    chat_history = state.get("chat_history", [])
    research_findings = state.get("research_findings", {}) 

    freelancers_found = research_findings.get("freelancers", [])
    articles_found = research_findings.get("articles", [])
    # kb_chunks from research_findings are just strings, not the full RAG output with sources
    kb_chunks_content_from_research = research_findings.get("kb_chunks", []) 

    outcomes.append(f"Query analysis: User asked '{query}'.")
    outcomes.append(f"Research findings: {len(freelancers_found)} freelancers, {len(articles_found)} articles, {len(kb_chunks_content_from_research)} KB chunk contents.")

    sensitive_query_terms = ["margin", "commission", "makers' fee", "shakers' fee", "makers fee", "shakers fee", "platform fee", "makers takes", "shakers takes"]
    is_sensitive_query = any(term in query.lower() for term in sensitive_query_terms)
    escalation_topic_value = None
    rag_call_token_usage = None # To store token usage from RAGService.answer_query

    if is_sensitive_query:
        outcomes.append("Sensitive query (e.g., platform margin) identified.")
        escalation_topic_value = "platform_margin"
        if kb_chunks_content_from_research: # If research found KB content, try RAG
            try:
                rag_result_dict = rag_service.answer_query(
                    session_id="customer-agent-sensitive-rag-session",
                    current_query=query,
                    chat_history=chat_history
                )
                response_text = rag_result_dict.get("answer", "I am looking into that for you. This type of query might require specific information.")
                rag_call_token_usage = rag_result_dict.get("token_usage")
                if rag_result_dict.get("sources"):
                    response_text += f"\n\nSources: {', '.join([src.get('document_name', 'N/A') for src in rag_result_dict.get('sources')])}"
                outcomes.append("RAG service provided initial info for sensitive query.")
            except Exception as e:
                print(f"Error in Customer Agent calling RAGService for sensitive query: {e}")
                response_text = "I am looking into that for you. This type of query might require specific information."
                outcomes.append(f"Error calling RAGService for sensitive query: {str(e)}")
        elif articles_found:
            response_text = "I found some general articles that might be related. For specific details on platform fees, this will be clarified shortly by our support team if needed."
        else:
            response_text = "I understand you have a question about platform specifics. I'll ensure this is addressed accurately."
    else:
        budget_info = _parse_budget_from_query(query)
        if budget_info:
            comp_type, val1, val2 = budget_info
            outcomes.append(f"Budget constraint identified: {comp_type} ${val1}" + (f" to ${val2}" if val2 else ""))
            matching_freelancers = []
            if freelancers_found:
                # freelancers_found from research_findings is a list of freelancer dicts directly
                # No longer wrapped in rec.get("freelancer", {})
                for freelancer in freelancers_found:
                    rate = freelancer.get("hourly_rate_usd")
                    # The 'freelancer' object here is directly from the freelancer_database.json structure
                    # It should have keys like 'name', 'specialization', 'hourly_rate_usd'
                    rate_str = freelancer.get("hourly_rate_usd") # This is usually a string like "80"
                    try:
                        rate = float(rate_str)
                    except (ValueError, TypeError):
                        # If rate is missing or not a number, skip this freelancer for budget filtering
                        continue
                    
                    if comp_type == "less" and rate < val1: matching_freelancers.append(freelancer)
                    elif comp_type == "less_equal" and rate <= val1: matching_freelancers.append(freelancer)
                    elif comp_type == "greater" and rate > val1: matching_freelancers.append(freelancer)
                    elif comp_type == "greater_equal" and rate >= val1: matching_freelancers.append(freelancer)
                    elif comp_type == "equal" and rate == val1: matching_freelancers.append(freelancer)
                    elif comp_type == "range" and val1 <= rate <= val2: matching_freelancers.append(freelancer)
            
            # Initialize response_text. If RAG provides a general answer, that will be used as a base.
            # Otherwise, we start fresh for budget-specific findings.
            base_rag_answer = ""
            try:
                # Attempt to get a RAG answer even if budget is primary focus, for context.
                rag_result_dict_budget_path = rag_service.answer_query(
                    session_id="customer-agent-budget-rag-session",
                    current_query=query, # Use original query for RAG
                    chat_history=chat_history
                )
                base_rag_answer = rag_result_dict_budget_path.get("answer", "")
                rag_call_token_usage = rag_result_dict_budget_path.get("token_usage") # Capture token usage
                rag_sources_budget_path = rag_result_dict_budget_path.get("sources")
                if base_rag_answer and rag_sources_budget_path:
                    source_doc_names = sorted(list(set([src.get('document_name', 'Unknown Document') for src in rag_sources_budget_path if src.get('document_name')])))
                    if source_doc_names:
                        base_rag_answer += "\n\n(This information is based on: " + ", ".join(source_doc_names) + ")"
                outcomes.append("RAG service provided contextual answer for budget query.")
            except Exception as e:
                print(f"Error calling RAGService in budget path: {e}")
                outcomes.append(f"RAGService call failed in budget path: {str(e)}")

            response_text = base_rag_answer + "\n\n" if base_rag_answer else ""

            if matching_freelancers:
                budget_criteria_str = f"{comp_type.replace('_', ' ')} ${val1:,.2f}"
                if val2: budget_criteria_str += f" to ${val2:,.2f}"
                response_text += f"Based on your budget criteria ({budget_criteria_str}), I found {len(matching_freelancers)} freelancer(s):"
                
                for i, freelancer in enumerate(matching_freelancers[:3]): # Show top 3 directly
                    name = freelancer.get('name', 'N/A')
                    specialization = freelancer.get('specialization', 'N/A') # Assuming freelancer dict has this
                    rate = freelancer.get('hourly_rate_usd', 'N/A')
                    response_text += f"\n- **{name}**: Specializes in {specialization}, Rate: ${rate}/hr"
                
                if len(matching_freelancers) > 3:
                    response_text += "\n(More matching freelancer profiles are in the 'Recommended Talent' sidebar.)"
                elif len(matching_freelancers) > 0:
                    response_text += "\n(Full profiles for these freelancers are in the 'Recommended Talent' sidebar.)"
                outcomes.append(f"Listed {len(matching_freelancers)} freelancers matching budget.")
            else:
                response_text += f"I couldn't find any freelancers matching your specified budget ({comp_type.replace('_', ' ')} ${val1}"
                if val2: response_text += f" to ${val2}"
                response_text += "). You might want to adjust your budget or criteria."
                outcomes.append("No freelancers found matching budget criteria.")
            
            # Append articles if found, regardless of budget match for freelancers
            if articles_found:
                response_text += "\n\nI also found some articles that might be generally helpful:"
                for i, article in enumerate(articles_found[:2]): # Show top 2
                    title = article.get('title', 'N/A')
                    doc_name = article.get('document_name', 'N/A')
                    preview = article.get('content', '')[:150] + '...' # First 150 chars as preview
                    response_text += f"\n- **{title}** (Source: {doc_name})\n  *Preview: {preview}*"
                if len(articles_found) > 2:
                    response_text += "\nFor more details on these and other articles, check the 'Recommended Resources' in the sidebar."
                elif len(articles_found) > 0:
                     response_text += "\nDetails for these articles are in the 'Recommended Resources' sidebar."
                outcomes.append("Appended general articles to budget response.")

            # If neither freelancers nor a RAG answer, and no articles, provide a fallback.
            if not matching_freelancers and not base_rag_answer and not articles_found:
                 response_text = f"I couldn't find any freelancers matching your budget criteria. I also didn't find specific information or articles for your query: '{query}'. You might want to try rephrasing or broadening your search."
                 outcomes.append("No budget freelancers, RAG answer, or articles found.")

        elif any(term in query.lower() for term in ['freelancer', 'specialist', 'expert']):
            if freelancers_found:
                response_text = f"I've found {len(freelancers_found)} specialist(s). Check the sidebar for details."
            else:
                response_text = f"I couldn't find specialists for '{query}'. Try rephrasing."

        elif kb_chunks_content_from_research:
            outcomes.append("Relevant KB chunks found by research. Generating interpretive answer using RAG.")
            try:
                rag_result_dict = rag_service.answer_query(
                    session_id="customer-agent-general-rag-session",
                    current_query=query,
                    chat_history=chat_history,
                    model='gemini-1.5-flash-latest' # Or determine model dynamically
                )
                response_text = rag_result_dict.get("answer", "I found some relevant information but couldn't generate a summary.")
                rag_call_token_usage = rag_result_dict.get("token_usage")
                rag_sources = rag_result_dict.get("sources")
                if rag_sources:
                    source_doc_names = sorted(list(set([src.get('document_name', 'Unknown Document') for src in rag_sources if src.get('document_name')])))
                    if source_doc_names:
                        response_text += "\n\n(This information is based on: " + ", ".join(source_doc_names) + ")"

                if articles_found:
                    response_text += "\n\nI also found some articles that might be helpful:"
                    for i, article in enumerate(articles_found[:2]): # Show top 2
                        title = article.get('title', 'N/A')
                        doc_name = article.get('document_name', 'N/A')
                        preview = article.get('content', '')[:150] + '...' # First 150 chars as preview
                        response_text += f"\n- **{title}** (Source: {doc_name})\n  *Preview: {preview}*"
                    if len(articles_found) > 2:
                        response_text += "\nFor more details on these and other articles, check the 'Recommended Resources' in the sidebar."
                    elif len(articles_found) > 0:
                         response_text += "\nDetails for these articles are in the 'Recommended Resources' sidebar."

                if freelancers_found:
                    response_text += "\n\nRegarding freelancers, I found these individuals who might be a fit:"
                    for i, freelancer in enumerate(freelancers_found[:3]): # Show top 3 directly
                        name = freelancer.get('name', 'N/A')
                        specialization = freelancer.get('specialization', 'N/A')
                        rate = freelancer.get('rate_usd_hr', 'N/A')
                        response_text += f"\n- **{name}**: Specializes in {specialization}, Rate: ${rate}/hr (if available)"
                    if len(freelancers_found) > 3:
                        response_text += "\nMore freelancer profiles are available in the 'Recommended Talent' section in the sidebar."
                    elif len(freelancers_found) > 0:
                        response_text += "\nFull profiles for these freelancers are in the 'Recommended Talent' sidebar."

                outcomes.append("RAG service generated an answer, potentially supplemented with articles and freelancer mentions.")
            except Exception as e:
                print(f"Error in Customer Agent calling RAGService for general query: {e}")
                response_text = "I found some information but had trouble summarizing it for you."
                outcomes.append(f"Error calling RAGService for general query: {str(e)}")
                
        elif articles_found or freelancers_found:
            response_text = f"Based on your query '{query}', I found the following:"
            if articles_found:
                response_text += f"\n\n**Relevant Articles ({len(articles_found)} found):**"
                for i, article in enumerate(articles_found[:3]): # Show top 3
                    title = article.get('title', 'N/A')
                    doc_name = article.get('document_name', 'N/A')
                    content_snippet = article.get('content', '')[:200] + '...' # First 200 chars
                    response_text += f"\n- **{title}** (from *{doc_name}*)\n  *{content_snippet}*"
                if len(articles_found) > 3:
                    response_text += "\n(More articles in the 'Recommended Resources' sidebar.)"
                elif len(articles_found) > 0:
                    response_text += "\n(Full articles in the 'Recommended Resources' sidebar.)"
            
            if freelancers_found:
                response_text += f"\n\n**Potential Freelancers ({len(freelancers_found)} found):**"
                # Check for budget context from the state if available (e.g., state.get('parsed_budget'))
                # This part requires knowing how budget is stored in AgentState if parsed earlier
                # For now, just list them generally
                for i, freelancer in enumerate(freelancers_found[:3]):
                    name = freelancer.get('name', 'N/A')
                    specialization = freelancer.get('specialization', 'N/A')
                    rate = freelancer.get('rate_usd_hr', 'N/A')
                    response_text += f"\n- **{name}**: Specializes in {specialization}, Rate: ${rate}/hr (if available)"
                if len(freelancers_found) > 3:
                    response_text += "\n(More freelancer profiles in the 'Recommended Talent' sidebar.)"
                elif len(freelancers_found) > 0:
                    response_text += "\n(Full profiles in the 'Recommended Talent' sidebar.)"
            
            outcomes.append("Provided relevant articles and/or freelancers as primary response.")
        else:
            escalation_topic_value = "general_inquiry_unresolved"
            response_text = "I couldn't find specific information for your query at the moment. For general inquiries or if you need further assistance, please feel free to contact our support team at support@makers.com. They'll be happy to help!"
            outcomes.append("No specific KB/article/freelancer info found. Provided general assistance message.")

    if rag_call_token_usage:
        print(f"DEBUG: customer_facing_agent_node received token usage: {rag_call_token_usage}")
        current_token_usage = _update_token_usage(current_token_usage, rag_call_token_usage)

    outcomes.append("Response generation complete.")
    print(f"Customer-Facing Agent Response: {response_text}")
    return {"final_response": response_text, "agent_outcomes": outcomes, "escalation_topic": escalation_topic_value, "api_token_usage": current_token_usage}


# Manager Agent
# Note: If manager_llm (qwen) is used, its token usage would also need to be captured.
# For now, it's providing canned responses, so no LLM call is made in this version.
# Manager Agent (Singleton)
# Upgraded to Gemini 1.5 Pro for more advanced reasoning and response direction.
manager_llm = ChatGroq(temperature=0.3, model_name="llama3-70b-8192")  # Reverted to Groq Llama3-70b
manager_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a senior manager AI overseeing a customer interaction. Your primary role is to ensure the quality, accuracy, and appropriateness of the final response. Review the query, chat history, research findings, and the Customer-Facing Agent's proposed response. \nKey Responsibilities:\n1. **Sensitive Topics**: If the query involves sensitive topics (e.g., company profit margins, confidential financial details, direct negative comparisons with named competitors), you MUST override or heavily guide the response. For sensitive financial queries, state that such details are confidential. For competitor comparisons, provide a neutral, factual statement about Makers' platform features without disparaging competitors. If asked for information not in the knowledge base (e.g. 'who are your investors?'), state that this information is not publicly available through this channel.\n2. **Accuracy & Completeness**: If the Customer Agent's response is inaccurate, incomplete, or misses the nuance of the query, provide a refined answer or clear, actionable instructions for the Customer Agent to reformulate its response. Ensure all relevant information (freelancers, articles, KB sources) is presented clearly if available.\n3. **Fallback & Escalation**: If the Customer Agent cannot find any information and the query is reasonable, provide fallback guidance (e.g., 'For specific account issues, please contact support@makers.com.') or suggest how the user might rephrase their query for better results. If the query is truly unanswerable or outside the platform's scope, state that clearly and politely.\n4. **Qualitative Comparisons**: For questions like 'which of these freelancers is best?', guide the Customer Agent to provide a balanced comparison based on available data (skills, experience, rate if relevant) rather than making a definitive judgment, and encourage the user to review full profiles.\n5. **Clarity and Conciseness**: Ensure the final response is clear, concise, and directly addresses the user's query.\n6. **Output Format**: Your response should *only* contain the text intended for the end-user. Do NOT include any prefatory phrases (e.g., 'I will now provide a refined answer:'), meta-commentary, or concluding summaries about your own response. Such internal notes should be part of your reasoning process but not in the final output text."),
    ("human", "User Query: {query}\nFull Chat History:\n{chat_history}\nRelevant Research Findings (from RAG, tools, etc.):\n{research_findings}\nCustomer Agent's Proposed Response (if any):\n{proposed_response}\nPrevious Agent Outcomes/Actions:\n{agent_outcomes}\n\nYour refined answer or direct instructions to the Customer Agent:")
])
manager_agent_runnable = manager_prompt_template | manager_llm

def manager_agent_node(state: AgentState):
    print("---AGENT: Manager--- ")
    query = state['query']
    chat_history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state['chat_history']])
    research_findings_str = str(state.get('research_findings', ''))
    # The 'final_response' from the customer_facing_agent_node is the 'proposed_response' for the manager
    proposed_customer_response = state.get("final_response", "Customer agent did not propose a response.")
    previous_agent_outcomes = state.get("agent_outcomes", [])
    current_token_usage = state.get('api_token_usage', {}).copy() # Ensure we have a mutable copy

    manager_input = {
        "query": query,
        "chat_history": chat_history_str,
        "research_findings": research_findings_str,
        "proposed_response": proposed_customer_response,
        "agent_outcomes": previous_agent_outcomes
    }

    print(f"DEBUG: Manager input: {manager_input}")
    
    try:
        # Invoke the manager LLM
        # The response object from ChatGoogleGenerativeAI typically includes usage_metadata
        manager_response_obj = manager_agent_runnable.invoke(manager_input)
        
        if hasattr(manager_response_obj, 'content'):
            final_response = manager_response_obj.content
        else:
            # Fallback if the response object structure is different (e.g. if it's just a string)
            final_response = str(manager_response_obj)

        # Clean the response from the manager agent
        final_response = final_response.strip()
        # Regex to remove everything up to and including the first colon on a line, plus any following whitespace and optional newline.
        prefix_pattern = r'^.*?:\s*\n?'
        final_response = re.sub(prefix_pattern, '', final_response, flags=re.IGNORECASE).strip()
        
        input_tokens = 0
        output_tokens = 0
        cost = 0.0

        if hasattr(manager_response_obj, 'response_metadata') and 'token_usage' in manager_response_obj.response_metadata:
            # LangChain's standard way of returning token usage for some models
            usage = manager_response_obj.response_metadata['token_usage']
            input_tokens = usage.get('prompt_token_count', 0) 
            output_tokens = usage.get('candidates_token_count', 0)
        elif hasattr(manager_response_obj, 'usage_metadata'):
            # Gemini specific way via ChatGoogleGenerativeAI
            input_tokens = manager_response_obj.usage_metadata.get('prompt_token_count',0)
            output_tokens = manager_response_obj.usage_metadata.get('candidates_token_count',0)
        else:
            # Fallback: Estimate based on text length (crude)
            print("Warning: Could not find token usage in manager_response_obj. Estimating crudely.")
            input_text_for_llm = manager_prompt_template.format(**manager_input)
            input_tokens = len(input_text_for_llm) // 4 
            output_tokens = len(final_response) // 4

        cost = ((input_tokens / 1000) * GROQ_LLAMA3_70B_INPUT_COST_PER_1K_TOKENS) + \
               ((output_tokens / 1000) * GROQ_LLAMA3_70B_OUTPUT_COST_PER_1K_TOKENS)
        
        print(f"DEBUG: Manager LLM (Groq Llama3-70b) usage: Input Tokens: {input_tokens}, Output Tokens: {output_tokens}, Cost: ${cost:.6f}")

        current_token_usage['groq_input_tokens'] = current_token_usage.get('groq_input_tokens', 0) + input_tokens
        current_token_usage['groq_output_tokens'] = current_token_usage.get('groq_output_tokens', 0) + output_tokens
        current_token_usage['total_cost'] = current_token_usage.get('total_cost', 0.0) + cost
        
        agent_outcomes = previous_agent_outcomes + [f"Manager Agent (Groq Llama3-70b) reviewed and provided response. Cost: ${cost:.6f}"]
        print(f"Manager Agent (Groq Llama3-70b) Output: {final_response}")

    except Exception as e:
        print(f"CRITICAL ERROR in Manager Agent: {e}\n{traceback.format_exc()}")
        final_response = "I encountered an issue while processing your request with the managerial review. Please try again."
        agent_outcomes = previous_agent_outcomes + [f"Manager Agent failed: {str(e)}"]

    return {"final_response": final_response, "agent_outcomes": agent_outcomes, "api_token_usage": current_token_usage}

# --- Graph Construction ---
workflow = StateGraph(AgentState)
workflow.add_node("research_agent", research_agent_node)
workflow.add_node("customer_facing_agent", customer_facing_agent_node)
workflow.add_node("manager_agent", manager_agent_node)

workflow.set_entry_point("research_agent")
workflow.add_edge("research_agent", "customer_facing_agent")
workflow.add_edge("customer_facing_agent", "manager_agent")
workflow.add_edge("manager_agent", END)

# Compile the graph
try:
    agent_graph = workflow.compile()
    print("Agent graph compiled successfully.")
except Exception as e:
    print(f"Error compiling agent graph: {e}")
    agent_graph = None

# --- Main Application Logic ---
def run_agent_graph(query: str, chat_history: Optional[List[dict]] = None):
    if not agent_graph:
        return {"error": "Agent graph not compiled."}
    if chat_history is None:
        chat_history = []
        
    initial_state = AgentState(
        query=query, 
        chat_history=chat_history, 
        research_findings={},
        agent_outcomes=[], 
        final_response="",
        escalation_topic=None,
        api_token_usage={
            'embedding_tokens': 0, 
            'gemini_input_tokens': 0, 'gemini_output_tokens': 0,
            'groq_input_tokens': 0, 'groq_output_tokens': 0,
            'total_cost': 0.0
        }
    )
    
    print(f"\n--- Running Agent Graph for Query: '{query}' ---")
    print(f"Initial Chat History: {chat_history}")
    print(f"Initial Token Usage: {initial_state['api_token_usage']}")

    final_state = agent_graph.invoke(initial_state)
    
    print("\n--- Agent Graph Execution Complete ---")
    print(f"Final Response: {final_state.get('final_response')}")
    print(f"Final Token Usage: {final_state.get('api_token_usage')}")
    print("Agent Outcomes:")
    for outcome in final_state.get('agent_outcomes', []):
        print(f"- {outcome}")
    
    return {
        "final_response": final_state.get("final_response"),
        "agent_outcomes": final_state.get("agent_outcomes"),
        "updated_chat_history": chat_history + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": final_state.get("final_response")}
        ],
        "structured_recommendations": final_state.get("research_findings", {}).get("freelancers", []),
        "structured_articles": final_state.get("research_findings", {}).get("articles", []),
        "api_token_usage": final_state.get("api_token_usage", {}) 
    }

if __name__ == '__main__':
    print("Running example usage...")
    if not all([db_service, rag_service, rs_service, agent_graph]):
        print("Services or agent graph not available. Exiting example.")
    else:
        history = []
        # Test 1: Freelancer search with budget (triggers RAG tool for embeddings)
        result1 = run_agent_graph("Find me a python developer for less than $70 an hour", history)
        history = result1.get("updated_chat_history", [])
        print(f"API Usage after Test 1: {result1.get('api_token_usage')}")
        print("-" * 50)

        # Test 2: General platform question (triggers RAG tool for embeddings + RAG answer_query for LLM)
        result2 = run_agent_graph("How do payments work on Makers?", history)
        history = result2.get("updated_chat_history", [])
        print(f"API Usage after Test 2: {result2.get('api_token_usage')}")
        print("-" * 50)

        # Test 3: Sensitive query (triggers RAG answer_query if KB chunks found)
        result3 = run_agent_graph("What is the platform fee for Makers?", history)
        history = result3.get("updated_chat_history", [])
        print(f"API Usage after Test 3: {result3.get('api_token_usage')}")
        print("-" * 50)

        # Test 4: Web search (no RAG/LLM calls in this path currently)
        # result4 = run_agent_graph("What's the latest news on AI?", history)
        # history = result4.get("updated_chat_history", [])
        # print(f"API Usage after Test 4: {result4.get('api_token_usage')}")
        # print("-" * 50)

        # print("-" * 50)