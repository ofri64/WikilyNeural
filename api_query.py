from wiktionary_api import WiktionaryAPI

api = WiktionaryAPI()
text = api.get_page_tags("bird cherry")
print(text[1])
