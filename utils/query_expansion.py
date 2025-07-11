from typing import List, Optional
from langchain_groq import ChatGroq

# Global history lists
prev_inputs: List[str] = []
prev_outputs: List[str] = []


def generate_stateful_query(
    user_query: str,
    llm: Optional[ChatGroq] = None,
    model_name: str = "qwen-32b",
    temperature: float = 0.7
) -> str:
    """
    Generate a single optimized search query each time, maintaining history of the last
    inputs and generated queries. Automatically correct any conceptual or logical errors
    in the user's query before expansion. Returns the new expanded query string.
    """
    global prev_inputs, prev_outputs

    # Initialize LLM if not provided
    if llm is None:
        llm = ChatGroq(model=model_name, temperature=temperature)

    # Define system-level prompt for financial RAG assistant with error correction
    system_prompt = (
        "You are a specialized Financial Retrieval-Augmented Generation (RAG) assistant. "
        "Your goal is to craft precise, high-recall search queries over financial datasets, "
        "incorporating terminology, market context, and relevant instrument identifiers. "
        "If the user's query contains any conceptual or logical errors, first correct them silently "
        "before generating the optimized search query."
    )

    # Build history context for last three interactions
    last_inputs = prev_inputs[-3:]
    last_outputs = prev_outputs[-3:]
    history_str = ""
    if last_inputs and last_outputs:
        pairs = [f"Q: {q}\nA: {a}" for q, a in zip(last_inputs, last_outputs)]
        history_str = "Here are the last interactions:\n" + "\n".join(pairs) + "\n"

    # Construct full prompt with context and user query
    prompt = (
        system_prompt + "\n" +
        history_str +
        f"New user query: '{user_query}'. Generate a single optimized financial search query"
        " that maximizes both recall and precision by adding relevant market-specific terms, "
        "financial jargon, and asset identifiers (e.g., tickers, ISINs)."
    )

    # Invoke LLM
    raw_output = llm(prompt)
    new_query = raw_output.strip()

    # Update history
    prev_inputs.append(user_query)
    prev_outputs.append(new_query)

    return new_query
