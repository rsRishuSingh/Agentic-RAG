import json
import os
from typing import List, Union
from datetime import datetime, timezone, timedelta

def append_to_response(new_items: List[Union[dict, object]], filename: str = "response.json") -> None:
    """
    Append a list of message-like objects to a JSON array in `filename`, tagging each with a 'timestamp'.
    Supports both dicts and LangChain Message objects like HumanMessage or AIMessage.
    """
    # Indian timezone
    IST = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(IST).isoformat()

    # Load existing data
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

    for item in new_items:
        if hasattr(item, "model_dump"):
            item_dict = item.model_dump()
        elif isinstance(item, dict):
            item_dict = item
        else:
            raise TypeError(f"Unsupported type: {type(item)}")

        item_dict.setdefault("timestamp", now)
        data.append(item_dict)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
