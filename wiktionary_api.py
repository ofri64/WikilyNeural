import requests
import json
import re

API_URL = "http://en.wiktionary.org/w/api.php"


class WiktionaryAPI(object):
    def __init__(self, api_url: str = API_URL):
        self.api_url = api_url
        self.request_params = {
            'prop': 'extracts|revisions', 'explaintext': '', 'rvprop': 'ids',
            'format': 'json', 'action': 'query'
        }

    @staticmethod
    def _extract_page_content(json_content: dict) -> str:
        pages_dict = json_content["query"]["pages"]
        first_page = next(iter(pages_dict.values()))
        page_content = ""  # default value in case of a missing page
        if "extract" in first_page:
            page_content = first_page["extract"]

        return page_content

    @staticmethod
    def _extract_page_level_sections(content: str, level: int = 2) -> list:
        sections = []
        num_prefix_tokens = level + 1
        pattern = f"[^=]={{{num_prefix_tokens}}} +.* +={{{num_prefix_tokens}}}[^=]"
        level_sections_iter = re.finditer(pattern, content)
        for section_match in level_sections_iter:
            section_title = section_match[0][1:-1][num_prefix_tokens+1:-num_prefix_tokens-1]
            sections.append(section_title)
        return sections

    def request_page(self, page_title: str) -> dict:
        request_params = self.request_params
        request_params["titles"] = page_title

        response = requests.get(self.api_url, params=request_params)
        content = response.content
        json_content = json.loads(content)

        return json_content

    def get_page_tags(self, page_title: str) -> tuple:
        json_content = self.request_page(page_title)
        page_content = self._extract_page_content(json_content)
        inner_level_sections = self._extract_page_level_sections(page_content, level=3)
        return page_title, inner_level_sections
