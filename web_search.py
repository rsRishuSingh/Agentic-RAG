import requests
import wikipediaapi

SERPER_API_KEY = "8351c8d666a70eaf483cc0f2a0c120440aaa0b96"

def serper_search(query: str, num: int = 5, api_key: str = SERPER_API_KEY) -> dict:
    """
    Perform a Google search using the Serper API.
    
    Args:
        query (str): The search query.
        num (int): Number of results to return. Default is 5.
        api_key (str): Your Serper API key.
    
    Returns:
        dict: Parsed JSON response from Serper.
    """
    url = "https://google.serper.dev/search"
    payload = {
        "q": query,
        "num": num
    }
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()

def get_wikipedia_page(title: str, user_agent: str = "AgenticAI/1.0 (contact@example.com)") -> dict:
    """
    Fetch Wikipedia page content using wikipediaapi.
    
    Args:
        title (str): Title of the Wikipedia page.
        user_agent (str): Custom User-Agent string for the requests.
    
    Returns:
        dict: Dictionary with page existence, summary, and full text.
    """
    
    wiki = wikipediaapi.Wikipedia(user_agent=user_agent, language="en")
    page = wiki.page(title)
    return {
        "exists": page.exists(),
        "title": page.title,
        "summary": page.summary,
        "content": page.text
    }

# Example usage:
query = 'Share price of tesla'
results = serper_search(query)
print(results)
wiki_data = get_wikipedia_page("tesla")
print(wiki_data)
