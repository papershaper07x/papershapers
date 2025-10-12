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
    # Prepare headers: allow caller to pass headers dict; ensure JSON content-type.
    headers = dict(headers or {})
    headers.setdefault("Content-Type", "application/json")

    # Determine API key from headers or environment
    api_key = headers.pop("tavily_api_key", None) or os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise Exception(
            "Tavily API key not found. Please set the TAVILY_API_KEY environment variable or pass it in headers['tavily_api_key']."
        )

    # Prefer to pass the API key in an Authorization header or x-api-key
    headers.setdefault("Authorization", f"Bearer {api_key}")
    headers.setdefault("x-api-key", api_key)

    payload = {
        "query": query,
        "search_depth": "advanced",
        "topic": topic,
        "days": 2,
        "max_results": int(max_results),
        "use_cache": True,
    }

    try:
        # Use requests' json parameter so requests sets the Content-Type correctly
        response = requests.post(base_url, json=payload, headers=headers, timeout=30)
        try:
            response.raise_for_status()  # Raises an HTTPError for bad responses
        except requests.HTTPError as http_err:
            # Log response body to help debug 400/422 errors
            print(f"Tavily API HTTPError {response.status_code}: {response.text}")
            raise

        resp_json = response.json()
        results = resp_json.get("results") or resp_json.get("data") or []
        if not results:
            # Log response for debugging when no results returned
            print(f"Tavily API returned empty results: {resp_json}")
            return []

        # Normalize expected keys to {href, body}
        normalized = []
        for obj in results:
            url = obj.get("url") or obj.get("href") or obj.get("link")
            content = obj.get("content") or obj.get("body") or obj.get("snippet")
            if url or content:
                normalized.append({"href": url, "body": content})
        return normalized
    except Exception as e:
        # Provide more debug info than before so callers can see why 400 happened
        print(f"Error: {e}. Failed fetching sources. Resulting in empty response.")
        return []
