import os
import json
from typing import TypedDict, List, Dict, Any, Optional, Annotated, Literal
from dotenv import load_dotenv

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq

# Your module imports
from computations import TeslaFinancialAnalyzer
from web import GoogleChatbot
from general import general_query

load_dotenv()

class AgentState(TypedDict):
    """Enhanced state with proper message handling and routing control."""
    messages: Annotated[List[BaseMessage], add_messages]
    pdf_path: Optional[str]
    task_completed: bool  # NEW: Track if task is complete
    iterations: int       # NEW: Track iteration count

# Enhanced tool wrapper functions with better error handling
def tesla_financial_analysis(query: str, pdf_path: str = None) -> str:
    """Tesla financial analysis wrapper with enhanced error handling."""
    try:
        analyzer = TeslaFinancialAnalyzer()
        if pdf_path and os.path.exists(pdf_path):
            analyzer.extract_financial_data_from_pdf(pdf_path)
            analyzer.calculate_portfolio_metrics()
        return analyzer.query_llm(query)
    except Exception as e:
        return f"Tesla analysis completed with error: {str(e)}. Please ask a different financial question."

def web_search_analysis(query: str) -> str:
    """Web search analysis wrapper with enhanced error handling."""
    try:
        bot = GoogleChatbot()
        result = bot.ask(query)
        return f"Search completed: {result}"
    except Exception as e:
        return f"Web search completed with error: {str(e)}. Please try a different search query."

def general_document_analysis(query: str) -> str:
    """General document analysis wrapper with enhanced error handling."""
    try:
        result = general_query()  # Updated to work with your general.py
        return f"Document analysis completed: {result}"
    except Exception as e:
        return f"Document analysis completed with error: {str(e)}. Please try a different query."

# Enhanced LangGraph tools with completion indicators
@tool
def tesla_tool(query: str, pdf_path: Optional[str] = None) -> str:
    """Tesla 10-K financial analysis and portfolio metrics calculation. This tool provides complete financial analysis."""
    return tesla_financial_analysis(query, pdf_path)

@tool
def web_tool(query: str) -> str:
    """Real-time web search for current information and news. This tool provides complete search results."""
    return web_search_analysis(query)

@tool
def general_tool(query: str) -> str:
    """General document analysis using RAG. This tool provides complete document analysis."""
    return general_document_analysis(query)

class FixedLangGraphOrchestrator:
    """Fixed LangGraph orchestrator with proper stop conditions."""
    
    def __init__(self, groq_api_key: str = None):
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY must be provided or set as environment variable")
        
        # Use ChatGroq instead of direct Groq client for better LangChain integration
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            api_key=self.groq_api_key
        )
        
        self.tools = [tesla_tool, web_tool, general_tool]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.graph = self._create_graph()
    
    def _create_graph(self) -> StateGraph:
        """Create the fixed LangGraph workflow with proper routing."""
        
        def agent_node(state: AgentState) -> AgentState:
            """Enhanced agent node with proper stop conditions."""
            messages = state["messages"]
            iterations = state.get("iterations", 0)
            
            # Increment iteration count
            iterations += 1
            
            # Stop condition: Max iterations reached
            if iterations > 10:
                return {
                    "messages": [AIMessage(content="I've reached the maximum number of iterations. Please ask a new question.")],
                    "task_completed": True,
                    "iterations": iterations
                }
            
            # Enhanced system message with clear stopping instructions
            system_message = """You are an intelligent assistant that can help with three types of queries:

1. **Tesla Financial Analysis** - Use tesla_tool for Tesla 10-K analysis, portfolio metrics, financial calculations
2. **Web Search** - Use web_tool for current news, real-time information, stock prices  
3. **Document Analysis** - Use general_tool for document analysis and RAG queries

IMPORTANT STOPPING RULES:
- If you have already called a tool and received a complete answer, DO NOT call tools again
- Provide a final response based on the tool results
- Only call tools if you need specific information you don't already have
- If the user asks a simple greeting or conversational question, respond directly without tools

Based on the conversation, decide whether to:
1. Call a tool if you need specific information
2. Provide a final answer if you have sufficient information"""

            # Prepare messages for the LLM
            conversation_messages = [HumanMessage(content=system_message)] + messages
            
            try:
                # Call the LLM with tools
                response = self.llm_with_tools.invoke(conversation_messages)
                
                # Update state
                updated_state = {
                    "messages": [response],
                    "iterations": iterations,
                    "task_completed": False
                }
                
                # Check if this is a final response (no tool calls)
                if not response.tool_calls:
                    updated_state["task_completed"] = True
                
                return updated_state
                
            except Exception as e:
                error_response = AIMessage(content=f"I encountered an error: {str(e)}. Please try asking your question differently.")
                return {
                    "messages": [error_response],
                    "task_completed": True,
                    "iterations": iterations
                }
        
        def router(state: AgentState) -> Literal["tools", "__end__"]:
            """Enhanced router with multiple stop conditions."""
            
            # Stop condition 1: Task marked as completed
            if state.get("task_completed", False):
                return "__end__"
            
            # Stop condition 2: Too many iterations
            if state.get("iterations", 0) > 10:
                return "__end__"
            
            # Stop condition 3: Check last message for tool calls
            messages = state["messages"]
            if messages:
                last_message = messages[-1]
                if isinstance(last_message, AIMessage):
                    if last_message.tool_calls:
                        return "tools"
                    else:
                        return "__end__"
            
            return "__end__"
        
        # Build the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", ToolNode(self.tools))
        
        # Add edges with proper routing
        workflow.add_edge(START, "agent")
        
        # FIXED: Use explicit routing map instead of default tools_condition
        workflow.add_conditional_edges(
            "agent",
            router,
            {
                "tools": "tools",
                "__end__": END
            }
        )
        
        # Always return to agent after tool execution
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()
    
    def chat(self, query: str, pdf_path: str = None) -> str:
        """Process a single query with fixed recursion handling."""
        try:
            # Create initial state with proper initialization
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "pdf_path": pdf_path,
                "task_completed": False,
                "iterations": 0
            }
            
            # Invoke with recursion limit and config
            config = {"recursion_limit": 15}  # Increased but still safe limit
            
            # Get final state
            final_state = self.graph.invoke(initial_state, config=config)
            
            # Extract the final response
            messages = final_state.get("messages", [])
            if messages:
                last_message = messages[-1]
                if isinstance(last_message, (AIMessage, ToolMessage)):
                    return last_message.content
            
            return "I'm sorry, I couldn't process your request properly."
            
        except Exception as e:
            return f"Error processing query: {str(e)}"
    
    def interactive_chat(self):
        """Start an interactive chat session with enhanced error handling."""
        print("🤖 Fixed LangGraph Multi-Tool Assistant")
        print("=" * 50)
        print("I can help you with:")
        print("• Tesla financial analysis")
        print("• Real-time web search")
        print("• Document analysis")
        print("Type 'quit' to exit")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Thanks for using the assistant!")
                    break
                
                if not user_input:
                    print("Please enter a query.")
                    continue
                
                # Check if user wants to provide PDF path for Tesla analysis
                pdf_path = None
                if any(keyword in user_input.lower() for keyword in ['tesla', 'financial', 'sharpe', 'portfolio']):
                    pdf_input = input("📄 PDF path for Tesla analysis (optional, press Enter to skip): ").strip()
                    if pdf_input and os.path.exists(pdf_input):
                        pdf_path = pdf_input
                
                print("🤖 Processing your query...")
                response = self.chat(user_input, pdf_path)
                
                print(f"\n🤖 Assistant:\n{response}\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Thanks for using the assistant!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")

def main():
    """Main function with enhanced error handling."""
    print("🚀 Starting Fixed LangGraph Assistant...")
    
    try:
        # Check for required environment variables
        if not os.getenv("GROQ_API_KEY"):
            print("❌ Error: GROQ_API_KEY environment variable not found!")
            return
        
        # Initialize the fixed orchestrator
        orchestrator = FixedLangGraphOrchestrator()
        print("✅ Assistant initialized successfully!")
        
        # Start interactive chat
        orchestrator.interactive_chat()
        
    except Exception as e:
        print(f"❌ Error initializing assistant: {str(e)}")

if __name__ == "__main__":
    main()
