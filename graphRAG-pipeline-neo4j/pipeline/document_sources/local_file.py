import logging
from pathlib import Path
import chardet

from .document import Document


def detect_encoding(file_path):
    """
    Detects the file encoding to avoid UnicodeDecodeError.

    Args:
        file_path (str or Path): Path to the file.

    Returns:
        str: Detected encoding (default "utf-8" if not found).
    """
    with open(file_path, 'rb') as f:
        raw_data = f.read(4096)
        result = chardet.detect(raw_data)
        return result['encoding'] or "utf-8"


def _load_pdf(file_path) -> list:
    """Loads a PDF page-by-page using PyMuPDF directly (replacement for
    langchain_community.document_loaders.PyMuPDFLoader)."""
    import fitz  # PyMuPDF

    pages = []
    with fitz.open(str(file_path)) as pdf:
        for page_number, page in enumerate(pdf):
            text = page.get_text()
            pages.append(Document(
                page_content=text,
                metadata={"source": str(file_path), "page": page_number},
            ))
    return pages


def _load_unstructured_elements(file_path):
    """Loads a file via unstructured's auto-partitioner (replacement for
    langchain_community.document_loaders.UnstructuredFileLoader(mode="elements")).
    Returns a list of Document, one per element, with metadata mirroring what
    UnstructuredFileLoader would have produced (source/page_number/filename/filetype
    and, where present, category for PageBreak detection)."""
    from unstructured.partition.auto import partition

    elements = partition(filename=str(file_path))
    docs = []
    for el in elements:
        meta = el.metadata.to_dict() if hasattr(el.metadata, "to_dict") else dict(el.metadata.__dict__)
        metadata = {
            "source": str(file_path),
            "filename": meta.get("filename", Path(file_path).name),
            "filetype": meta.get("filetype", ""),
        }
        if meta.get("page_number") is not None:
            metadata["page_number"] = meta["page_number"]
        category = getattr(el, "category", None)
        if category is not None:
            metadata["category"] = category
        docs.append(Document(page_content=str(el), metadata=metadata))
    return docs


def load_document_content(file_path):
    """
    Loads document content from a file, handling PDFs and text encoding.

    Args:
        file_path (str or Path): Path to the file.

    Returns:
        tuple: (loader_fn, encoding_flag)
            loader_fn: zero-arg callable returning a list[Document].
            encoding_flag (bool): True if non-UTF-8 encoding was used for .txt files.
    """
    file_extension = Path(file_path).suffix.lower()
    encoding_flag = False
    if file_extension == '.pdf':
        return (lambda: _load_pdf(file_path)), encoding_flag
    if file_extension == ".txt":
        encoding = detect_encoding(file_path)
        logging.info("Detected encoding for text file: %s", encoding)
        if encoding.lower() == "utf-8":
            return (lambda: _load_unstructured_elements(file_path)), encoding_flag
        with open(file_path, encoding=encoding, errors="replace") as f:
            content = f.read()
        documents = [Document(page_content=content, metadata={"source": str(file_path)})]
        encoding_flag = True
        return (lambda: documents), encoding_flag
    return (lambda: _load_unstructured_elements(file_path)), encoding_flag


def get_documents_from_file_by_path(file_path, file_name):
    """
    Loads documents from a file by its path and returns file name, pages, and extension.

    Args:
        file_path (str or Path): Path to the file.
        file_name (str): Name of the file.

    Returns:
        tuple: (file_name, pages, file_extension)

    Raises:
        Exception: If file does not exist or reading fails.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logging.info('File %s does not exist', file_name)
        raise Exception(f'File {file_name} does not exist')
    logging.info('file %s processing', file_name)
    try:
        loader_fn, encoding_flag = load_document_content(file_path)
        file_extension = file_path.suffix.lower()
        if file_extension == ".pdf" or (file_extension == ".txt" and encoding_flag):
            pages = loader_fn()
        else:
            unstructured_pages = loader_fn()
            pages = get_pages_with_page_numbers(unstructured_pages)
    except Exception as exc:
        raise Exception(f'Error while reading the file content or metadata, {exc}')
    return file_name, pages, file_extension


def get_pages_with_page_numbers(unstructured_pages):
    """
    Groups unstructured pages into logical pages with page numbers and metadata.

    Args:
        unstructured_pages (list): List of Document objects.

    Returns:
        list: List of Document objects with page numbers and metadata.
    """
    pages = []
    page_number = 1
    page_content = ''
    metadata = {}
    for idx, page in enumerate(unstructured_pages):
        if 'page_number' in page.metadata:
            if page.metadata['page_number'] == page_number:
                page_content += page.page_content
                metadata = {
                    'source': page.metadata['source'],
                    'page_number': page_number,
                    'filename': page.metadata['filename'],
                    'filetype': page.metadata['filetype']
                }
            if page.metadata['page_number'] > page_number:
                page_number += 1
                pages.append(Document(page_content=page_content))
                page_content = ''
            if page == unstructured_pages[-1]:
                pages.append(Document(page_content=page_content))
        elif page.metadata.get('category') == 'PageBreak' and page != unstructured_pages[0]:
            page_number += 1
            pages.append(Document(page_content=page_content, metadata=metadata))
            page_content = ''
            metadata = {}
        else:
            page_content += page.page_content
            metadata_with_custom_page_number = {
                'source': page.metadata['source'],
                'page_number': 1,
                'filename': page.metadata['filename'],
                'filetype': page.metadata['filetype']
            }
            if page == unstructured_pages[-1]:
                pages.append(Document(page_content=page_content, metadata=metadata_with_custom_page_number))
    return pages
