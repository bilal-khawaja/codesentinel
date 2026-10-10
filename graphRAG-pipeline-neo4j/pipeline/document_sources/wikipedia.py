import logging
import wikipedia
wikipedia.set_user_agent("llm-graph-builder/1.0")

from requests.exceptions import JSONDecodeError

from .document import Document

logger = logging.getLogger(__name__)


def get_documents_from_wikipedia(wiki_query: str, language: str):
    """Loads documents from Wikipedia for a given query and language."""
    try:
        results = wikipedia.search(wiki_query.replace('_', ' '), results=1, suggestion=False)
        docs = []
        try:
            page = wikipedia.page(results[0], auto_suggest=False, redirect=True, preload=False)
            docs.append(Document(page_content=page.content[:100000], metadata={"title": page.title, "source": page.url}))
        except wikipedia.exceptions.DisambiguationError as e:
            logger.warning("Disambiguation error for query '%s': %s", wiki_query, str(e))
        file_name = wiki_query.strip()
        logger.info("Total Pages from Wikipedia = %d", len(docs))
        return file_name, docs
    except JSONDecodeError as exc:
        raise Exception("Wikipedia returned an invalid response, possibly due to rate limiting.") from exc
    except Exception as exc:
        raise Exception(f"{str(exc)} Failed To Process Wikipedia Query") from exc
