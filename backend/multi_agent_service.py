import os
import re
from typing import TypedDict, Annotated, List, Optional, Tuple
import operator
from dotenv import load_dotenv

# Langchain and Langgraph imports
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Import services
from .database_service import DatabaseService
from .rag_service import RAGService
from .recommendation_service import RecommendationService

# --- Agent State ---
class AgentState(TypedDict):
    query: str
    chat_history: list
    research_findings: dict  # Changed from str to dict
    # Expected keys: 'freelancers': List[dict], 'articles': List[dict], 'kb_chunks': List[str]
    agent_outcomes: Annotated[List[str], operator.add]
    final_response: str
    escalation_topic: Optional[str] # Added for escalating sensitive queries

# --- Service Initialization ---
try:
    db_service = DatabaseService()
    rag_service = RAGService(db_service=db_service)
    rs_service = RecommendationService() # Recommendation Service
    print("All services initialized successfully.")
except Exception as e:
    print(f"Error initializing services: {e}")
    db_service, rag_service, rs_service = None, None, None

# --- Tools ---
@tool
def web_search_tool(query: str) -> str:
    """Searches the web for the given query. Use for general knowledge or recent events."""
    print(f"---TOOL: Web Search--- \nQuery: {query}")
    # This is a placeholder. In a real scenario, you'd use an API like Tavily or SerpAPI.
    return f"Placeholder web search results for '{query}'."

@tool
def rag_and_freelancer_search_tool(query: str) -> dict:
    """Searches the internal knowledge base for platform documentation, the freelancer database for specialists, and retrieves relevant KB articles. Use this for any questions about the 'Makers' platform, its features, or when asked to find, recommend, or search for freelancers."""
    print(f"---TOOL: RAG & Freelancer Search--- \nQuery: {query}")
    
    findings = {
        "freelancers": [],
        "articles": [],
        "kb_chunks": []
    }

    # Get freelancer and article recommendations
    if rs_service:
        try:
            recommendations = rs_service.get_recommendations_for_query(query)
            findings["freelancers"] = recommendations.get("freelancers", [])
            findings["articles"] = recommendations.get("articles", [])
            print(f"Recommendation service found {len(findings['freelancers'])} freelancers and {len(findings['articles'])} articles.")
        except Exception as e:
            print(f"Error calling RecommendationService: {e}")
    else:
        print("RecommendationService (rs_service) not available.")

    # Get relevant knowledge base chunks for RAG
    if rag_service:
        try:
            kb_chunks = rag_service.search_knowledge_base(query, k=3, session_id="tool-rag-search-session")
            findings["kb_chunks"] = kb_chunks # List of strings (document chunks)
            print(f"RAG service found {len(findings['kb_chunks'])} knowledge base chunks.")
        except Exception as e:
            print(f"Error calling RAGService.search_knowledge_base: {e}")
    else:
        print("RAGService (rag_service) not available.")

    return findings

# --- Helper Function for Budget Parsing ---
def _parse_budget_from_query(query: str) -> Optional[Tuple[str, float, Optional[float]]]:
    """
    Parses a query to extract budget information.
    Returns a tuple: (comparison_type, value1, value2_for_range) or None.
    Comparison types: "less", "less_equal", "greater", "greater_equal", "equal", "range".
    """
    query_lower = query.lower()
    price_match = re.search(r'\$?(\d+(\.\d{1,2})?)\b', query_lower) # Finds numbers like 50, 50.5, $50, $50.50
    
    if not price_match:
        return None
    
    price = float(price_match.group(1))
    
    # Order by specificity to avoid premature matching
    if "less than or equal to" in query_lower or "at most" in query_lower or "<=" in query:
        return ("less_equal", price, None)
    if "less than" in query_lower or "under" in query_lower or "<" in query :
        return ("less", price, None)
    if "more than or equal to" in query_lower or "at least" in query_lower or ">=" in query:
        return ("greater_equal", price, None)
    if "more than" in query_lower or "over" in query_lower or ">" in query:
        return ("greater", price, None)
    if "exactly" in query_lower or "equal to" in query_lower or (query_lower.startswith("is $") and price_match): # "is $50"
         return ("equal", price, None)
    if "around" in query_lower or "about" in query_lower:
        lower_bound = price * 0.9 # Example: 10% range
        upper_bound = price * 1.1
        return ("range", lower_bound, upper_bound)
    
    # If a price is mentioned, and no other comparators like "more", "over", "greater", "least" are present,
    # assume the user implies "less than or equal to" as a common default for budget constraints.
    # This is a heuristic.
    if price_match and not any(comp in query_lower for comp in ["more", "over", "greater", "least", "minimum", "above"]):
         return ("less_equal", price, None)

    return None # No clear budget constraint parsed

# --- Agent Logics ---

# Research Agent
research_llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")
research_tools = [rag_and_freelancer_search_tool, web_search_tool]
research_agent_runnable = research_llm.bind_tools(research_tools)

def research_agent_node(state: AgentState):
    print("---AGENT: Research--- ")
    outcomes = []
    try:
        # For now, bypass the actual LLM-based research agent logic
        # Directly use the rag_and_freelancer_search_tool or web_search_tool based on query
        print(f"DEBUG: Research Agent bypassing LLM, using direct tool call for query: {state['query']}")
        query_lower = state['query'].lower()
        
        # Prioritize freelancer/platform queries for the specialized tool
        if any(term in query_lower for term in ['freelancer', 'specialist', 'expert', 'developer', 'designer', 'consultant', 'makers', 'platform', 'payment', 'fee', 'rate', 'budget', 'cost']):
            result = rag_and_freelancer_search_tool(state['query'])
            outcomes.append("Research Agent used RAG & Freelancer Search tool.")
        else: # Fallback to general web search
            result = {"web_search_results": web_search_tool(state['query'])} # Ensure dict structure
            outcomes.append("Research Agent used Web Search tool.")
            
        outcomes.append("Research completed successfully.")
        return {"agent_outcomes": outcomes, "research_findings": result}
            
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"\n\nCRITICAL ERROR IN RESEARCH AGENT: {e}\n{error_trace}\n\n")
        return {"agent_outcomes": outcomes + [f"Research Agent failed: {str(e)}"], 
                "research_findings": {"error": f"Error during research: {str(e)}"}}

# Customer-Facing Agent
customer_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)

def customer_facing_agent_node(state: AgentState):
    """Generates the final response for the user using structured findings."""
    print("---AGENT: Customer-Facing--- ")
    response_text = "I encountered an issue processing your request. Please try again."
    outcomes = ["Customer-Facing Agent received the query."]
    
    query = state.get("query", "")
    chat_history = state.get("chat_history", [])
    research_findings = state.get("research_findings", {}) # Now a dict

    freelancers_found = research_findings.get("freelancers", [])
    articles_found = research_findings.get("articles", [])
    kb_chunks_found = research_findings.get("kb_chunks", []) # List of strings

    outcomes.append(f"Query analysis: User asked '{query}'.")
    outcomes.append(f"Research findings: {len(freelancers_found)} freelancers, {len(articles_found)} articles, {len(kb_chunks_found)} KB chunks.")

    sensitive_query_terms = ["margin", "commission", "makers' fee", "shakers' fee", "makers fee", "shakers fee", "platform fee", "makers takes", "shakers takes"]
    is_sensitive_query = any(term in query.lower() for term in sensitive_query_terms)
    escalation_topic_value = None

    if is_sensitive_query:
        outcomes.append("Sensitive query (e.g., platform margin) identified.")
        escalation_topic_value = "platform_margin"
        # Try to find general info if available, but primary response will be from manager
        if kb_chunks_found:
            try:
                rag_result = rag_service.answer_query(
                    session_id="customer-agent-sensitive-rag-session",
                    current_query=query,
                    chat_history=chat_history
                )
                response_text = rag_result.get("answer", "I am looking into that for you. This type of query might require specific information.")
                if rag_result.get("sources"):
                    response_text += f"\n\nSources: {', '.join([src.get('document_name', 'N/A') for src in rag_result.get('sources')])}"
                outcomes.append("RAG service provided initial info for sensitive query.")
            except Exception as e:
                print(f"Error in Customer Agent calling RAGService for sensitive query: {e}")
                response_text = "I am looking into that for you. This type of query might require specific information."
                outcomes.append(f"Error calling RAGService for sensitive query: {str(e)}")
        elif articles_found:
            response_text = "I found some general articles that might be related. For specific details on platform fees, this will be clarified shortly by our support team if needed."
            outcomes.append("Found articles potentially related to sensitive query.")
        else:
            response_text = "I understand you have a question about platform specifics. I'll ensure this is addressed accurately."
        # The manager agent will ultimately provide the final response for escalated topics.
    else:
        budget_info = _parse_budget_from_query(query)
        if budget_info:
            comp_type, val1, val2 = budget_info
            outcomes.append(f"Budget constraint identified: {comp_type} ${val1}" + (f" to ${val2}" if val2 else ""))
            
            matching_freelancers = []
            if freelancers_found:
                for rec in freelancers_found:
                    freelancer = rec.get("freelancer", {})
                    rate = freelancer.get("hourly_rate_usd")
                    if rate is None: continue

                    if comp_type == "less" and rate < val1: matching_freelancers.append(freelancer)
                    elif comp_type == "less_equal" and rate <= val1: matching_freelancers.append(freelancer)
                    elif comp_type == "greater" and rate > val1: matching_freelancers.append(freelancer)
                    elif comp_type == "greater_equal" and rate >= val1: matching_freelancers.append(freelancer)
                    elif comp_type == "equal" and rate == val1: matching_freelancers.append(freelancer)
                    elif comp_type == "range" and val1 <= rate <= val2: matching_freelancers.append(freelancer)
            
            if matching_freelancers:
                response_text = f"Okay, I found {len(matching_freelancers)} freelancer(s) matching your budget criteria ({comp_type.replace('_', ' ')} ${val1}"
                if val2: response_text += f" to ${val2}"
                response_text += "):\n"
                for i, fl in enumerate(matching_freelancers[:3]): # Show top 3
                    response_text += f"- {fl.get('name')} at ${fl.get('hourly_rate_usd')}/hour\n"
                if len(matching_freelancers) > 3:
                    response_text += f"See the sidebar for all {len(matching_freelancers)} matching freelancers."
                else:
                     response_text += "You can find their full profiles in the sidebar recommendations."
                outcomes.append(f"Found {len(matching_freelancers)} freelancers within budget.")
            else:
                response_text = f"I couldn't find any freelancers matching your budget of {comp_type.replace('_', ' ')} ${val1}"
                if val2: response_text += f" to ${val2}"
                response_text += ". You might consider adjusting your budget, or I can show you all available freelancers for this skill. What would you like to do?"
                outcomes.append("No freelancers found within the specified budget.")

        elif any(term in query.lower() for term in ['freelancer', 'specialist', 'expert', 'developer', 'designer', 'consultant']):
            if freelancers_found:
                response_text = f"I've found {len(freelancers_found)} specialist(s) who might be a good fit. Check the sidebar for detailed recommendations, including their skills, rates, and match percentages."
            else:
                response_text = f"I couldn't find any specialists matching '{query}' right now. You could try rephrasing your needs or broadening your search."
            outcomes.append("Processed general freelancer query.")

        elif kb_chunks_found: # Prioritize using RAG if KB chunks are found
            outcomes.append("Relevant KB chunks found. Generating interpretive answer using RAG.")
            try:
                rag_result = rag_service.answer_query(
                    session_id="customer-agent-rag-session",
                    current_query=query,
                    chat_history=chat_history
                )
                response_text = rag_result.get("answer", "I found some relevant information, but I'm having a bit of trouble summarizing it. Please check the recommended articles if available.")
                if rag_result.get("sources"):
                     response_text += f"\n\nSources: {', '.join([src.get('document_name', 'N/A') for src in rag_result.get('sources')])}"
                outcomes.append("RAG service generated an answer from KB chunks.")
            except Exception as e:
                print(f"Error in Customer Agent calling RAGService: {e}")
                response_text = "I found some relevant information, but encountered an issue trying to explain it. You might find details in the recommended articles."
                outcomes.append(f"Error calling RAGService: {str(e)}")
                
        elif articles_found:
            response_text = f"I found {len(articles_found)} article(s) that might help with '{query}':\n"
            for i, art_info in enumerate(articles_found[:2]): # Show top 2
                response_text += f"- {art_info.get('article_name', 'N/A').replace('.md', '').replace('_', ' ')}\n"
            response_text += "You can find these in the 'Recommended Resources' section."
            outcomes.append("Found relevant articles.")
            
        else: # Default fallback
            response_text = f"I've processed your query: '{query}'. I couldn't find specific freelancers or detailed articles for this, but I can try a general search if you'd like."
            outcomes.append("No specific freelancers, articles, or KB chunks found for the query. Default response.")

    outcomes.append("Response generation complete.")
    print(f"Customer-Facing Agent Response: {response_text}")
    # Pass escalation_topic to the next state, defaulting to None if not set by sensitive query logic
    return {"final_response": response_text, "agent_outcomes": outcomes, "escalation_topic": escalation_topic_value}


# Manager Agent (Placeholder)
manager_llm = ChatGroq(temperature=0, model_name="qwen-32b-chat") # Example Qwen model

def manager_agent_node(state: AgentState):
    print("---AGENT: Manager--- ")
    final_response = state.get("final_response", "No final response generated.")
    agent_outcomes = state.get("agent_outcomes", []) # Get existing outcomes
    escalation_topic = state.get("escalation_topic")

    if escalation_topic == "platform_margin":
        final_response = (
            "For specific details about our platform's fee structure and margins, "
            "please refer to our official Terms of Service document or contact our dedicated support team. "
            "They can provide you with the most accurate and up-to-date information."
        )
        # Append to outcomes, don't overwrite
        agent_outcomes.append("Manager Agent provided standard response for platform margin query.")
    else:
        agent_outcomes.append("Manager Agent approved the response from Customer-Facing Agent.")
    
    return {"final_response": final_response, "agent_outcomes": agent_outcomes}

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

# --- Main Application Logic (FastAPI or other framework would go here) ---
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
        escalation_topic=None # Initialize escalation_topic
    )
    
    # Log initial state
    print(f"\n--- Running Agent Graph for Query: '{query}' ---")
    print(f"Initial Chat History: {chat_history}")

    final_state = agent_graph.invoke(initial_state)
    
    print("\n--- Agent Graph Execution Complete ---")
    print(f"Final Response: {final_state.get('final_response')}")
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
        # Include structured data if needed by frontend for display
        "structured_recommendations": final_state.get("research_findings", {}).get("freelancers", []),
        "structured_articles": final_state.get("research_findings", {}).get("articles", [])
    }

if __name__ == '__main__':
    # Example Usage
    print("Running example usage...")
    # Ensure services are up
    if not all([db_service, rag_service, rs_service, agent_graph]):
        print("Services or agent graph not available. Exiting example.")
    else:
        history = []
        
        # Test 1: General platform question
        # result1 = run_agent_graph("How do payments work on Makers?", history)
        # history = result1.get("updated_chat_history", [])
        # print("-" * 50)

        # Test 2: Freelancer search with budget
        result2 = run_agent_graph("Find me a python developer for less than $70 an hour", history)
        history = result2.get("updated_chat_history", [])
        print("-" * 50)

        # Test 3: Follow-up, more specific budget
        # result3 = run_agent_graph("Okay, what about under $50/hr for python?", history)
        # history = result3.get("updated_chat_history", [])
        # print("-" * 50)

        # Test 4: General freelancer search
        # result4 = run_agent_graph("I need a good Javascript expert", history)
        # history = result4.get("updated_chat_history", [])
        # print("-" * 50)