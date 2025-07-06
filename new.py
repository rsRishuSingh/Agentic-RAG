import requests

def get_wikipedia_page_mw(title: str, language: str = "en") -> dict:
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
wiki_data = get_wikipedia_page_mw("A. P. J. Abdul Kalam")
print(wiki_data)
