import json
import os
from typing import Any, List, Union
from datetime import datetime, timezone, timedelta
from langchain_core.messages import BaseMessage

def _unwrap(item: Any) -> Any:
    """
    Recursively convert BaseMessage objects to dicts via model_dump(),
    and leave other types (primitives, lists, dicts) intact.
    """
    if isinstance(item, BaseMessage):
        return item.model_dump()
    elif isinstance(item, dict):
        return {k: _unwrap(v) for k, v in item.items()}
    elif isinstance(item, list):
        return [_unwrap(v) for v in item]
    else:
        return item

def append_to_response(
    new_items: List[Union[dict, BaseMessage, Any]],
    filename: str = "response.json"
) -> None:
    """
    Append a list of items to a JSON array in `filename`, tagging each with a 'timestamp'.
    Supports dicts, lists, primitives, and LangChain Message objects (BaseMessage).
    """
    # Indian timezone
    IST = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(IST).isoformat()

    # Load existing data (or start fresh list)
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError(f"{filename} does not contain a JSON list.")
            except (json.JSONDecodeError, ValueError):
                data = []
    else:
        data = []

    # Process and append each new item
    for raw in new_items:
        # First unwrap any nested BaseMessage / lists / dicts
        item_dict = _unwrap(raw)

        # Must end up as a dict or primitive
        if not isinstance(item_dict, dict):
            # wrap primitives under a generic key
            item_dict = {"value": item_dict}

        # add timestamp if missing
        item_dict.setdefault("timestamp", now)
        data.append(item_dict)

    # Write back
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
