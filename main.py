import os
import json
import re
from typing import List, TypedDict, Annotated, Sequence, Any

from dotenv import load_dotenv
import fitz  # PyMuPDF for PDF parsing
import requests
import wikipedia
import numpy as np

from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

# -------------------------------------------------------------------
# ENVIRONMENT & MODEL INITIALIZATION
# -------------------------------------------------------------------
load_dotenv()
MODEL_NAME       = "qwen/qwen3-32b"
SERPER_API_KEY   = os.getenv("SERPER_API_KEY")
PDF_DIR          = os.getenv("PDF_DIR", "PDFs/")
ALL_DOCS_JSON    = os.getenv("ALL_DOCS_JSON", "all_docs.json")
CHROMA_DB_PATH   = os.getenv("CHROMA_DB_PATH", "chromaDB/saved/")
COLLECTION_NAME  = os.getenv("COLLECTION_NAME", "RAG_DOCS")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")


# -------------------------------------------------------------------
# DOCUMENT STORAGE UTILITIES
# -------------------------------------------------------------------

def load_docs(filepath: str = ALL_DOCS_JSON) -> List[Document]:
    """
    Load pre-chunked documents from a JSON file for RAG retrieval.

    Args:
        filepath: Path to the JSON file containing stored Document objects.

    Returns:
        A list of Document instances with page_content and metadata.
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
        A Chroma object pointing to the persisted vector database.
    """
    return Chroma(
        persist_directory=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME,
        embedding_function=HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    )

# -------------------------------------------------------------------
# TOOL DEFINITIONS WITH DOCSTRINGS
# -------------------------------------------------------------------

@tool
def hybrid_search(query: str) -> List[dict]:
    """
    Perform a hybrid retrieval combining BM25 and vector search (ChromaDB) on local PDFs.

    Args:
        query: User's natural language query string.

    Returns:
        A list of dicts with 'text' and metadata for top-matching document chunks.
    """
    # Load or index documents
    chroma_store = init_chroma()
    docs = load_docs()
    if chroma_store._collection.count() == 0 and docs:
        chroma_store.add_documents(docs)

    # Initialize retrievers
    bm25 = BM25Retriever.from_texts(
        [d.page_content for d in docs],
        metadatas=[d.metadata for d in docs],
        k=5
    )
    vec_ret = chroma_store.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10})
    ensemble = EnsembleRetriever(retrievers=[bm25, vec_ret], weights=[1, 1])

    # Query ensemble
    results = ensemble.invoke(query)
    return [{"text": d.page_content, **d.metadata} for d in results]


@tool
def google_search(query: str, num: int = 5) -> dict:
    """
    Perform a web search via the Serper API for current news or factual queries.

    Args:
        query: The search string to send to Google.
        num: Number of top results to retrieve.

    Returns:
        The raw JSON response from the Serper API.
    """
    url = "https://google.serper.dev/search"
    payload = {"q": query, "num": num}
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()


@tool
def wiki_lookup(title: str) -> dict:
    """
    Fetch a Wikipedia page summary and content for a given title.

    Args:
        title: Exact title of the Wikipedia article.

    Returns:
        A dict containing existence, summary, content, URL, or error message.
    """
    wikipedia.set_lang("en")
    try:
        page = wikipedia.page(title)
        return {
            "exists": True,
            "title": page.title,
            "summary": wikipedia.summary(title, sentences=3),
            "content": page.content,
            "url": page.url
        }
    except Exception as e:
        return {"exists": False, "error": str(e)}


# ----- Financial Metrics Tools ----- #
@tool
def sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """
    Compute the Sharpe Ratio: (mean(return - rf)) / std(return - rf).

    Args:
        returns: Time-series of portfolio returns.
        risk_free_rate: Baseline return to subtract from portfolio.

    Returns:
        The Sharpe ratio (float).
    """
    arr = np.array(returns, dtype=float)
    excess = arr - risk_free_rate
    if arr.size < 2 or np.std(excess, ddof=1) == 0:
        raise ValueError("Insufficient data or zero volatility for Sharpe Ratio.")
    return float(np.mean(excess) / np.std(excess, ddof=1))


@tool
def batting_average(port: List[float], bench: List[float]) -> float:
    """
    Calculate fraction of periods where portfolio > benchmark.
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
    """
    p = np.array(port);
    b = np.array(bench)
    if p.size != b.size or p.size == 0:
        raise ValueError("Return series must be equal-length non-empty arrays.")
    up = p[b > 0].sum() / b[b > 0].sum() if np.any(b > 0) else None
    down = p[b < 0].sum() / b[b < 0].sum() if np.any(b < 0) else None
    return {"up_capture": up, "down_capture": down}


@tool
def tracking_error(port: List[float], bench: List[float]) -> float:
    """
    Calculate std deviation of (port - bench).
    """
    diff = np.array(port) - np.array(bench)
    if diff.size < 2:
        raise ValueError("Need at least two observations for tracking error.")
    return float(np.std(diff, ddof=1))


@tool
def max_drawdown(returns: List[float]) -> float:
    """
    Compute max peak-to-trough drawdown.
    """
    r = np.array(returns)
    if r.size == 0:
        raise ValueError("Empty return series.")
    wealth = np.cumprod(1 + r)
    peak = np.maximum.accumulate(wealth)
    return float(((wealth - peak) / peak).min())

# -------------------------------------------------------------------
# AGENT GRAPH DEFINITION
# -------------------------------------------------------------------
class AgentState(TypedDict):
    """
    State dict storing chat messages and any parsed user data.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_data: Any


graph = StateGraph(AgentState)


def entry_agent(state: AgentState) -> AgentState:
    """
    Initial router that inspects user intent and decides which tool node to invoke.
    - Financial metrics if arrays provided
    - Wikipedia lookup if 'wiki' keyword
    - Web search for 'search' or 'latest'
    - Default to hybrid document search
    """
    prompt = SystemMessage(
        content=(
            "You are a RAG orchestrator. Determine the user's intent precisely. ``financial`` "
            "if they supply returns and benchmark arrays; ``wiki`` if they need an encyclopedia lookup; "
            "``web`` for general news or factual search; otherwise default to document retrieval."
        )
    )
    # Initialize ChatGroq LLM client
    llm = ChatGroq(model=MODEL_NAME).bind_tools([hybrid_search, google_search, wiki_lookup, sharpe_ratio,batting_average])
    llm_response = llm.invoke([prompt] + state["messages"])
    return {"messages": [llm_response], "user_data": state.get("user_data")}


def answer_agent(state: AgentState) -> AgentState:
    """
    Final node: synthesizes tool outputs into a concise, user-facing response.
    """
    prompt = SystemMessage(
        content="Now take the tool's output and craft a clear, friendly answer for the user."
    )
    llm = ChatGroq(model=MODEL_NAME)
    llm_response = llm.invoke([prompt] + state["messages"])
    return {"messages": [llm_response], "user_data": state.get("user_data")}

# Register nodes
graph.add_node('EntryAgent', entry_agent)
graph.add_node('HybridNode', ToolNode([hybrid_search]))
graph.add_node('WebNode', ToolNode([google_search, wiki_lookup]))
graph.add_node('FinNode', ToolNode([sharpe_ratio, batting_average, capture_ratios, tracking_error, max_drawdown]))
graph.add_node('AnswerAgent', answer_agent)

def route(state: AgentState) -> str:
    text = state['messages'][-1].content.lower()
    if 'returns' in text and 'benchmark' in text:
        return 'FinNode'
    if 'wiki' in text:
        return 'WebNode'
    if 'search' in text or 'latest' in text:
        return 'WebNode'
    return 'HybridNode'

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
    # Example for human-like message formatting
    user_input = HumanMessage(content='latest share price of tesla')
    state = {'messages': [user_input], 'user_data': None}
    response = app.invoke(state)
    print(response['messages'][-1].content)
print(app.get_graph().draw_ascii())  
