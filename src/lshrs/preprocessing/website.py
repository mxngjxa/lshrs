# function to specifically preprocess raw website data that still contains
# HTML tags, formatting issues, etc.

from typing import List
from urllib.request import urlopen

from bs4 import BeautifulSoup


def get_website_content(url: str) -> str:
    """
    Function to fetch the content of a webpage given its URL.

    :param url: The URL of the webpage to fetch.
    :return: The raw HTML content of the webpage as a string.
    """
    try:
        response = urlopen(url)
        html_content = response.read().decode("utf-8")
        return html_content

    except Exception:
        # print(f"Error fetching content from {url}: {e}")
        return ""


def process_website_content(
    html_content: str,
    target_tags: List[str] = [
        "p",
        "h1",
        "h2",
        "h3",
        "li",
        "article",
        "section",
        "blockquote",
    ],
) -> str:
    """
    Extracts and cleans text content from specific HTML tags in raw website HTML.

    :param html_content: Raw HTML content of the webpage.
    :param target_tags: List of HTML tag names to extract text from
      (e.g., ["p", "h1"]).

    :return: A cleaned string containing only text from the specified tags.
    """
    soup = BeautifulSoup(html_content, features="html.parser")

    # Remove unwanted elements like scripts and styles
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Collect text from specified target tags
    extracted_texts = []
    for tag_name in target_tags:
        for tag in soup.find_all(tag_name):
            text = tag.get_text(separator=" ", strip=True)
            if text:
                extracted_texts.append(text)

    # Join and clean the extracted text
    cleaned_text = " ".join(extracted_texts)
    return cleaned_text
