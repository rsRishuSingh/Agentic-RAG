import requests
import wikipedia
from wikipedia.exceptions import DisambiguationError, PageError

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

def get_wikipedia_page(title: str, language: str = "en") -> dict:
    """
    Fetch Wikipedia page content using the `wikipedia` package.

    Args:
        title (str): Title of the Wikipedia page.
        language (str): Language code (e.g., 'en', 'hi', 'fr').

    Returns:
        dict: Dictionary with page existence, title, summary, content, and URL.
              If page does not exist or is ambiguous, returns an error message.
    """
    wikipedia.set_lang(language)
    
    try:
        page = wikipedia.page(title)
        return {
            "exists": True,
            "title": page.title,
            "summary": wikipedia.summary(title, sentences=2),
            "content": page.content,
            "url": page.url
        }
    except DisambiguationError as e:
        return {
            "exists": False,
            "error": f"DisambiguationError: '{title}' refers to multiple pages.",
            "options": e.options
        }
    except PageError:
        return {
            "exists": False,
            "error": f"PageError: The page titled '{title}' does not exist."
        }
    except Exception as e:
        return {
            "exists": False,
            "error": f"UnexpectedError: {str(e)}"
        }


# Example usage:
query = 'Share price of tesla'
results = serper_search(query)
print(results)
wiki_data = get_wikipedia_page("Tesla")
print(wiki_data)
