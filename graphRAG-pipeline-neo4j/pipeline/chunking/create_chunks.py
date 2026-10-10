import logging
import re

import tiktoken

from ..graphdb.db import get_value_from_env
from ..document_sources.document import Document

logging.basicConfig(format="%(asctime)s - %(message)s", level="INFO")


class _TokenTextSplitter:
    """Minimal replacement for langchain_text_splitters.TokenTextSplitter:
    splits page_content into fixed-size, overlapping token windows using the
    tokenizer associated with the configured embedding model."""

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        model_name: str = "text-embedding-3-small",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            logging.warning("Unknown tokenizer model '%s'; using cl100k_base", model_name)
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def split_text(self, text: str):
        tokens = self.encoding.encode(text, disallowed_special=())
        if not tokens:
            return []
        step = max(self.chunk_size - self.chunk_overlap, 1)
        chunks = []
        start = 0
        while start < len(tokens):
            window = tokens[start:start + self.chunk_size]
            chunks.append(self.encoding.decode(window))
            if start + self.chunk_size >= len(tokens):
                break
            start += step
        return chunks

    def split_documents(self, documents):
        result = []
        for document in documents:
            for piece in self.split_text(document.page_content):
                result.append(Document(page_content=piece, metadata=dict(document.metadata)))
        return result


class CreateChunksofDocument:
    """Splits a list of documents (pages) into smaller token-sized chunks."""

    def __init__(self, pages: list):
        self.pages = pages

    def split_file_into_chunks(
        self,
        token_chunk_size: int,
        chunk_overlap: int,
        email: str = None,
        embedding_model: str = "text-embedding-3-small",
    ):
        logging.info("Split file into smaller chunks")
        text_splitter = _TokenTextSplitter(
            chunk_size=token_chunk_size,
            chunk_overlap=chunk_overlap,
            model_name=embedding_model,
        )
        max_token_chunk_size = get_value_from_env("MAX_TOKEN_CHUNK_SIZE", 10000, "int")
        chunk_to_be_created = int(max_token_chunk_size / token_chunk_size)
        normalized_email = (email or "").strip().lower() or None
        is_neo4j_user = bool(normalized_email and normalized_email.endswith("@neo4j.com"))

        if not self.pages:
            return[]
            
        first_metadata = self.pages[0].metadata
        chunks = []
        if 'page' in first_metadata or "page_number" in first_metadata:

            for i, document in enumerate(self.pages):
                page_number = document.metadata.get('page') or document.metadata.get('page_number') or (i + 1)
                for chunk in text_splitter.split_documents([document]):
                    updated_metadata = dict(chunk.metadata)
                    updated_metadata['page_number'] = page_number
                    clean_content = re.sub(r'\s+', ' ', chunk.page_content).strip()
                    if clean_content:  # Only add chunk if it has non-whitespace content
                        chunks.append(Document(page_content=clean_content, metadata=updated_metadata))

        else:
            logging.info("No page metadata found, proceeding with standard document chunking")
            raw_chunks = text_splitter.split_documents(self.pages)
            for chunk in raw_chunks:
                    clean_content = re.sub(r"\s+", " ", chunk.page_content).strip()
                    if clean_content:
                        chunks.append(Document(page_content=clean_content, metadata=dict(chunk.metadata)))

        logging.info("Total chunks created: %d", len(chunks))

        if not is_neo4j_user and len(chunks) > chunk_to_be_created:
            chunks = chunks[:chunk_to_be_created]
            logging.info('Non Neo4j user - limiting chunks to %d', chunk_to_be_created)

        return chunks
