import os
import json
import re
from typing import List, TypedDict, Annotated, Sequence, Any

from dotenv import load_dotenv
import fitz  # PyMuPDF for PDF parsing
import requests
import numpy as np
from save_result import append_to_response

from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

# -------------------------------------------------------------------
# ENVIRONMENT & MODEL INITIALIZATION
# -------------------------------------------------------------------
load_dotenv()
MODEL_NAME       = os.getenv("MODEL_NAME", "qwen/qwen3-32b")
SERPER_API_KEY   = os.getenv("SERPER_API_KEY")
PDF_DIR          = os.getenv("PDF_DIR", "PDFs/")
ALL_DOCS_JSON    = os.getenv("ALL_DOCS_JSON", "all_docs.json")
CHROMA_DB_PATH   = os.getenv("CHROMA_DB_PATH", "chromaDB/saved/")
COLLECTION_NAME  = os.getenv("COLLECTION_NAME", "RAG_DOCS")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

# -------------------------------------------------------------------
# DOCUMENT STORAGE UTILITIES
# -------------------------------------------------------------------

def load_docs(filepath: str = ALL_DOCS_JSON) -> List[Document]:
    """
    Load pre-chunked documents from a JSON file for RAG retrieval.

    Args:
        filepath (str): Path to the JSON file containing stored Document objects.

    Returns:
        List[Document]: A list of Document instances with page_content and metadata.
    """
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        items = json.load(f)
    return [Document(page_content=i["page_content"], metadata=i["metadata"]) for i in items]


def init_chroma() -> Chroma:
    """
    Initialize or load an existing Chroma vector store for document embeddings.

    Returns:
        Chroma: A Chroma object pointing to the persisted vector database.
    """
    return Chroma(
        persist_directory=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME,
        embedding_function=HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    )

# -------------------------------------------------------------------
# TOOL DEFINITIONS
# -------------------------------------------------------------------

@tool
def hybrid_search(query: str) -> List[dict]:
    """
    Hybrid retrieval combining BM25 and vector search (ChromaDB) over local PDFs.

    Args:
        query (str): Natural language query string from the user.

    Returns:
        List[dict]: Top-matching chunks with 'text' and associated metadata.
    """
    chroma_store = init_chroma()
    docs = load_docs()
    if chroma_store._collection.count() == 0 and docs:
        chroma_store.add_documents(docs)

    bm25_ret = BM25Retriever.from_texts(
        [d.page_content for d in docs],
        metadatas=[d.metadata for d in docs],
        k=5
    )
    vec_ret = chroma_store.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10})
    ensemble = EnsembleRetriever(retrievers=[bm25_ret, vec_ret], weights=[0.5, 0.5])

    results = ensemble.invoke(query)
    # print("Hybrid --> ", results)

    return [{"text": d.page_content, **d.metadata} for d in results]

@tool
def google_search(query: str, num: int = 20) -> dict:
    """
    Web search via Serper API for real-time news and factual queries.

    Args:
        query (str): Search string to send to the API.
        num (int): Number of top results to retrieve (default: 5).
        country: 

    Returns:
        dict: Parsed JSON response from Serper with search results.
    """
    url = "https://google.serper.dev/search"

    payload = json.dumps({
     "q": query,
  "gl": "in",
  "num": num,
  "tbs": "qdr:w"
})
    headers = {
    'X-API-KEY': SERPER_API_KEY,
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    return response.text

@tool
def wiki_lookup(title: str, language: str = "en") -> dict:
    """
    Fetch full Wikipedia page content using the MediaWiki Action API.

    Args:
        title (str): Title of the Wikipedia page.
        language (str): Language code (e.g., 'en', 'hi', 'fr').

    Returns:
        dict: Dictionary with page existence, title, extract (intro), content (wikitext), and URL.
    """
    api_url = f"https://{language}.wikipedia.org/w/api.php"
    session = requests.Session()

    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts|info",
        "exintro": True,
        "explaintext": True,
        "titles": title,
        "inprop": "url",
    }

    try:
        response = session.get(url=api_url, params=params)
        # print("Wiki--> ", response)
        response.raise_for_status()
        data = response.json()

        pages = data.get("query", {}).get("pages", {})
        page = next(iter(pages.values()))

        if "missing" in page:
            return {
                "exists": False,
                "error": f"PageError: The page titled '{title}' does not exist."
            }

        return {
            "exists": True,
            "page_id": page.get("pageid"),
            "title": page.get("title"),
            "summary": page.get("extract"),
            "content_url": page.get("fullurl")
        }

    except requests.RequestException as e:
        return {
            "exists": False,
            "error": f"RequestError: {str(e)}"
        }
# ----- Financial Metrics Tools ----- #
@tool
def company_overview(symbol: str) -> dict:
    """
    Fetch company info & key financial metrics for a ticker via Alpha Vantage.

    Args:
        symbol (str): Stock ticker, e.g. "IBM" or "AAPL".

    Returns:
        dict: Overview fields such as Name, Exchange, MarketCap,
              P/E & PEG ratios, Dividends, Margins, Growth rates,
              Analyst targets, Valuation ratios, and 52‑week highs/lows.
    """
    # ensure .env has ALPHAVANTAGE_API_KEY=<your_key>
  
 
    if not ALPHAVANTAGE_API_KEY:
        raise ValueError("Set ALPHAVANTAGE_API_KEY in your environment before calling this tool.")
    
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "OVERVIEW",
        "symbol": symbol,
        "apikey": ALPHAVANTAGE_API_KEY,
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    return data


@tool
def sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """
    Calculate the Sharpe Ratio for a return series.

    Formula: mean(returns - rf) / std(returns - rf)

    Args:
        returns (List[float]): Portfolio return time series.
        risk_free_rate (float): Risk-free rate baseline (default: 0.0).

    Returns:
        float: Computed Sharpe ratio.

    Raises:
        ValueError: If insufficient data or zero volatility.
    """
    arr = np.array(returns, dtype=float)
    excess = arr - risk_free_rate
    if arr.size < 2 or np.std(excess, ddof=1) == 0:
        raise ValueError("Insufficient data or zero volatility for Sharpe Ratio.")
    return float(np.mean(excess) / np.std(excess, ddof=1))

@tool
def batting_average(port: List[float], bench: List[float]) -> float:
    """
    Compute the batting average: fraction of periods where portfolio beats benchmark.

    Args:
        port (List[float]): Portfolio return series.
        bench (List[float]): Benchmark return series of equal length.

    Returns:
        float: Proportion of periods where port > bench.

    Raises:
        ValueError: If series lengths differ or are empty.
    """
    p = np.array(port)
    b = np.array(bench)
    if p.size != b.size or p.size == 0:
        raise ValueError("Return series must be equal-length non-empty arrays.")
    return float(np.sum(p > b) / p.size)

@tool
def capture_ratios(port: List[float], bench: List[float]) -> dict:
    """
    Compute up- and down-market capture ratios.

    Args:
        port (List[float]): Portfolio return series.
        bench (List[float]): Benchmark return series of equal length.

    Returns:
        dict: Contains 'up_capture' and 'down_capture' ratios.

    Raises:
        ValueError: If series lengths differ or are empty.
    """
    p = np.array(port)
    b = np.array(bench)
    if p.size != b.size or p.size == 0:
        raise ValueError("Return series must be equal-length non-empty arrays.")
    up = p[b > 0].sum() / b[b > 0].sum() if np.any(b > 0) else None
    down = p[b < 0].sum() / b[b < 0].sum() if np.any(b < 0) else None
    return {"up_capture": up, "down_capture": down}

@tool
def tracking_error(port: List[float], bench: List[float]) -> float:
    """
    Calculate the tracking error: standard deviation of active returns (port - bench).

    Args:
        port (List[float]): Portfolio return series.
        bench (List[float]): Benchmark return series of equal length.

    Returns:
        float: Tracking error.

    Raises:
        ValueError: If fewer than two observations.
    """
    diff = np.array(port) - np.array(bench)
    if diff.size < 2:
        raise ValueError("Need at least two observations for tracking error.")
    return float(np.std(diff, ddof=1))

@tool
def max_drawdown(returns: List[float]) -> float:
    """
    Compute the maximum drawdown for a return series.

    Args:
        returns (List[float]): Portfolio return time series.

    Returns:
        float: Maximum peak-to-trough drawdown.

    Raises:
        ValueError: If the return series is empty.
    """
    r = np.array(returns)
    if r.size == 0:
        raise ValueError("Empty return series.")
    wealth = np.cumprod(1 + r)
    peak = np.maximum.accumulate(wealth)
    return float(((wealth - peak) / peak).min())

# -------------------------------------------------------------------
# AGENT GRAPH DEFINITION & PROMPTS
# -------------------------------------------------------------------
class AgentState(TypedDict):
    """
    State dictionary storing chat messages and any user-specific data.

    Fields:
        messages: (Sequence[BaseMessage]): Conversation history for the agent 1.
        user_data (Any): Optional storage for parsed user inputs or context.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    query: str
    user_data: Any

# Instantiate graph
graph = StateGraph(AgentState)
llm_query_redirector = ChatGroq(model=MODEL_NAME).bind_tools(
    [hybrid_search, google_search, wiki_lookup, company_overview, sharpe_ratio,
     batting_average, capture_ratios, tracking_error, max_drawdown]
)
def entry_agent(state: AgentState) -> AgentState:
    """
    Entry node: Classify user intent to select the appropriate retrieval or calculation tool.
    """
    system_prompt = SystemMessage(
    content=(
        "You are a Retrieval‑Augmented Generation (RAG) orchestrator.\n"
        "Based on the user's message, decide which tool to invoke:\n"
        "  1. FinancialMetrics: if the user provided a numeric return series or asks about ratios, metrics, stock analysis or permformance.\n"
        "  2. WikiLookup: Only if the query contains 'wiki' or requests historical/contextual background.\n"
        "  3. WebSearch: Only if the query contains 'latest', 'current', 'news', or factual real-time events.\n"
        "  4. HybridSearch: fallback for general document retrieval from local PDFs. These PDFs contain  detailed financial reports and security exchange filings of companies.\n\n"
        "After your internal reasoning, reply with exactly one of the following as the message content:\n"
        "  • 'Calling FinancialMetrics'\n"
        "  • 'Searching Wikipedia'\n"
        "  • 'Searching Web'\n"
        "  • 'Doing Hybrid Search'\n"
    )
)

    llm_response = llm_query_redirector.invoke([system_prompt] + state["messages"])
    # print('LLM 1 --> ',llm_response)
    
    return {"messages": [llm_response], "user_data": state.get("user_data")}
    # return {"messages": [AIMessage(content=llm_response.content, kwargs=llm_response.additional_kwargs)], "user_data": state.get("user_data")}

def generate_stateful_query(
    state: AgentState,
    temperature: float = 0.8
) -> str:
    """
    Generate one optimized financial search query, using the full history
    of raw message dicts, each having 'content' and 'additional_kwargs'.

    Args:
      state: AgentState containing
        - state["messages"]: List[dict] with keys 'content' and 'additional_kwargs'
        - state["user_data"]["past_queries"]: List[str] of prior queries
      temperature, max_tokens: LLM generation params

    Returns:
      A single, high‑precision, high‑recall financial search query.
    """
    user_data = state.setdefault("user_data", {})
    history = user_data.setdefault("past_queries", [])

    # Build a flattened context string from the last 5 messages
    ctx_entries = []
    for msg in state["messages"][-5:]:
        line = f"{msg['type'].upper()}: {msg.get('content', '') or '<no content>'}"
        ak = msg.get("additional_kwargs", {})

        if "reasoning_content" in ak:
            line += f"\n[reasoning] {ak['reasoning_content'].strip()}"

        if "tool_calls" in ak and ak["tool_calls"]:
            line += f"\n[tool_calls] {json.dumps(ak['tool_calls'], ensure_ascii=False)}"

        ctx_entries.append(line)

    context_str = "\n\n".join(ctx_entries)

    system = SystemMessage(
        content=(
            "You are a Financial RAG assistant. Based on the full conversation context "
            "below (including any silent reasoning and tool calls), produce exactly one "
            "optimized search query for financial data. "
            "Correct any conceptual errors in the user's request before expanding."
        )
    )
    human = HumanMessage(
        content=(
            f"CONTEXT:\n{context_str}\n\n"
            "Now generate a single, high‑recall and high‑precision financial search query "
            "by adding relevant market terms, tickers/ISINs, and financial jargon."
        )
    )

    llm = ChatGroq(model=MODEL_NAME, temperature=temperature)
    response = llm.invoke([system, human])

    query = response.content.strip()
    history.append(query)
    return query


def answer_agent(state: AgentState) -> AgentState:
    """
    Final node: Integrates tool results into a concise, accurate response using the same bound tools.

    """
    final_prompt = SystemMessage(
        content=(
            "You are a knowledgeable assistant. Use the tool outputs and conversation history to answer the user's query thoroughly. "
            "- Provide clear explanations, structured when appropriate. "
            "- If data is incomplete, acknowledge the gap. "
            "- Always be concise and accurate."
        )
    )
    llm = ChatGroq(model=MODEL_NAME)
    llm_response = llm.invoke([final_prompt] + state["messages"])
    # print('LLM 2 --> ',llm_response)

    return {"messages": [llm_response], "user_data": state.get("user_data")}
    # return {"messages": [AIMessage(content=llm_response.content, kwargs=llm_response.additional_kwargs)], "user_data": state.get("user_data")}

# Register and wire nodes
graph.add_node('EntryAgent', entry_agent)
graph.add_node('HybridNode', ToolNode([hybrid_search]))
graph.add_node('WebNode', ToolNode([google_search, wiki_lookup]))
graph.add_node('FinNode', ToolNode([company_overview,sharpe_ratio, batting_average,
                                     capture_ratios, tracking_error, max_drawdown]))
graph.add_node('AnswerAgent', answer_agent)

def route(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    calls   = getattr(last_msg, "additional_kwargs", {}).get("tool_calls", [])
    if calls:
        tool_name = calls[0]["function"]["name"]
        # catch every finance‐related tool in FinNode
        if tool_name in (
            "company_overview",
            "sharpe_ratio",
            "batting_average",
            "capture_ratios",
            "tracking_error",
            "max_drawdown"
        ):
            return "FinNode"
        if tool_name in ("google_search", "wiki_lookup"):
            return "WebNode"
        if tool_name in ("hybrid_search",):
            return "HybridNode"
    # fallback
    return "HybridNode"

# Graph wiring
graph.set_entry_point('EntryAgent')
graph.add_conditional_edges('EntryAgent', route, {
    'HybridNode': 'HybridNode',
    'WebNode': 'WebNode',
    'FinNode': 'FinNode'
})
for node in ['HybridNode', 'WebNode', 'FinNode']:
    graph.add_edge(node, 'AnswerAgent')
graph.add_conditional_edges('AnswerAgent', lambda _: 'end', {'end': END})

app = graph.compile()

if __name__ == '__main__':
    
    user_input = HumanMessage(content='''What is the EVToRevenue and ShareVolume of IBM''')
    state = {'messages': [user_input], 'user_data': None}
    response = app.invoke(state)
    print(response['messages'][-1].content, end='\v')
    append_to_response(response['messages'])
    # print(app.get_graph().draw_ascii())
