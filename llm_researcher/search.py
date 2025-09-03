import requests
import os
import json


def tavily_search(query: str, headers=None, topic="general", max_results=1) -> list:
    """Simplified function to search using the Tavily API.

    Args:
        query (str): The search query.
        headers (dict, optional): Additional headers. Defaults to None.
        topic (str, optional): The topic to search in. Defaults to "general".
        max_results (int, optional): Maximum number of results to return. Defaults to 7.

    Returns:
        list: A list of search results.
    """
    base_url = "https://api.tavily.com/search"
    headers = headers or {"Content-Type": "application/json"}
    api_key = headers.get("tavily_api_key") or os.environ.get("TAVILY_API_KEY")

    if not api_key:
        raise Exception(
            "Tavily API key not found. Please set the TAVILY_API_KEY environment variable."
        )

    data = {
        "query": query,
        "search_depth": "advanced",
        "topic": topic,
        "days": 2,
        "max_results": max_results,
        "api_key": api_key,
        "use_cache": True,
    }

    try:
        response = requests.post(
            base_url, data=json.dumps(data), headers=headers, timeout=100
        )
        response.raise_for_status()  # Raises an HTTPError for bad responses
        results = response.json().get("results", [])
        if not results:
            raise Exception("No results found with Tavily API search.")
        return [{"href": obj["url"], "body": obj["content"]} for obj in results]
    except Exception as e:
        print(f"Error: {e}. Failed fetching sources. Resulting in empty response.")
        return []
