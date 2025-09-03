import requests
from bs4 import BeautifulSoup

def validate_url(url):
    try:
        response = requests.head(url, allow_redirects=True,timeout=10)
        return response.status_code == 200
    except requests.RequestException:
        return False
def get_prompt_tokens(response):
    # For Google Gemini responses
    return response.result.usage_metadata.prompt_token_count

def get_completion_tokens(response):
    # For Google Gemini responses
    return response.result.usage_metadata.candidates_token_count
def scrape_webpage(link, session=None):
    """
    Scrapes content from a webpage, removing scripts and styles,
    and returns a tuple containing:
      - the cleaned content (str)
      - the page title (str)
      - an empty list for image URLs (list)
    This function always returns three values to avoid unpacking errors.
    """
    if session is None:
        session = requests.Session()

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/58.0.3029.110 Safari/537.3"
        )
    }

    try:
        if validate_url(link):
            response = session.get(link, headers=headers, timeout=20)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser', from_encoding=response.encoding)

            title = soup.title.string if soup.title else 'No title found'
            # If needed, add logic here to extract image URLs; for now, we return an empty list.
            all_content = soup.get_text()
            unique_lines = set()
            ordered_unique_lines = []

            for line in all_content.splitlines():
                stripped_line = line.strip()
                if stripped_line and len(stripped_line.split()) > 3 and stripped_line not in unique_lines:
                    unique_lines.add(stripped_line)
                    ordered_unique_lines.append(stripped_line)

            content = '\n'.join(ordered_unique_lines)
            print('-' * 50)
            print(content)

            # Always return three values: content, title, and an empty list for images.
            return content, title, []
        else:
            return "", "", []
    except requests.exceptions.RequestException as e:
        print(f"Error! : {e}")
        return "", "", []
