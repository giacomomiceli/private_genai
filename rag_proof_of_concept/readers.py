"""
Document Processing Module

This module provides functionality for loading and processing documents from various sources. It
includes functions for loading documents from directories, splitting text into manageable chunks,
and handling different file formats.

Key components:
    - load_directory: Function to load all text and PDF files from a specified directory
    - DirectoryLoader: Class for loading documents from a directory
    - TextLoader: Class for loading text documents
    - PyMuPDFLoader: Class for loading PDF documents
    - RecursiveCharacterTextSplitter: Class for splitting text into chunks based on character count

Typical usage example:
    from documents import load_directory 
    from langchain.schema.document import Document

    documents = load_directory('/path/to/directory')
    
    for doc in documents:
        print(doc.content)

Note:
    This module requires the 'langchain', 'langchain-community', and 'langchain-text-splitters'
    packages to be installed.
"""

# External imports
import logging

import re
import yaml
from collections import defaultdict, Counter
from tqdm import tqdm

# Pluralization, number to word conversion for building prompts
import inflect

from langchain.schema.document import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyMuPDFLoader

from weaviate.client import WeaviateClient

# Text splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter

from openai import BadRequestError

from typing import List, Dict

# Internal imports
from .utils import ChunkGranularity, LLMClient
from .vdb import clear_vector_store, load_to_vector_store

__all__ = [
    'load_directory',
    'load_txt_directory',
    'load_pdf_directory',
    'split_documents'
]

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------------------
# Constants
DOCUMENT_SPLIT_SEPARATORS = ['\\n\\n', '([.!?](\\s+|\\n))']

# --------------------------------------------------------------------------------------------------
# Functions for cleaning the content of documents.

def _extract_head(text: str, num_sentences: int = 1) -> str:
    pattern = r'[^.!?\n]*[.!?\n]?'
    matches = re.findall(pattern, text)
    extracted_sentences = ' '.join(matches[:num_sentences]).strip()
    return extracted_sentences

def _count_headers(header_candidates):
    # Initialize a dictionary to store headers for each source
    headers_dict = defaultdict(list)
    
    # Initialize a dictionary to count occurrences of each header for each source
    header_count = defaultdict(Counter)
    
    # Count occurrences of each header for each source
    for header, source in header_candidates:
        if header:  # Ignore empty headers
            header_count[source][header] += 1
    
    # Filter headers that repeat at least once
    for source, counter in header_count.items():
        for header, count in counter.items():
            if count > 1:
                headers_dict[source].append(header)
    
    return headers_dict

def _clean_pdf_documents(docs: list[Document]) -> None:
    header_candidates = [[_extract_head(x.page_content, 1), x.metadata['source']] for x in docs]
    headers = _count_headers(header_candidates)

    for doc in docs:
        for header in headers[doc.metadata['source']]:
            doc.page_content = doc.page_content.replace(header, '')

def _clean_txt_documents(docs: list[Document]) -> None:
    for doc in docs:
        doc.metadata['page'] = 0
        doc.metadata['total_pages'] = 1
        doc.metadata['granularity'] = ChunkGranularity.RAW.value

def _clean_chunks(chunks: list[Document]) -> None:
    for chunk in chunks:
        x = chunk.page_content.replace('\n', ' ').replace('\t', ' ').strip()
        x = re.sub(r'\s{2,}', ' ', x)
        x = re.sub(r'(\.\s*){2,}\s', '. ', x)
        x = re.sub(r'(\.\s*){2,}', '.', x)
        x = x.lstrip('.').strip()
        chunk.page_content = x
        chunk.metadata['granularity'] = ChunkGranularity.RAW.value


#---------------------------------------------------------------------------------------------------
# Functions for loading and splitting documents.

def load_directory(path: str) -> list[Document]:
    """
    Load all text files and PDF files found in the specified directory and its subdirectories.

    Args:
        path (str): The path to the directory containing the files to load.
    
    Returns:
        list[Document]: A list of Document objects representing the loaded files.
    """
    
    logger.info(f"Loading all text files found in {path} and its subdirectories")
    txt_documents = load_txt_directory(path)
    logger.info(f"Loading all pdf files found in {path} and its subdirectories")
    pdf_documents = load_pdf_directory(path)
    combined_documents = txt_documents + pdf_documents
    return combined_documents


def load_txt_directory(path: str) -> list[Document]:
    """
    Load text files from a directory and its subdirectories into a list of Document objects.
    
    Args:
        path (str): The path to the directory containing the text files to load.

    Returns:
        list[Document]: A list of Document objects representing the loaded text files.
    """
    document_loader = DirectoryLoader(
        path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={'autodetect_encoding': True},
        show_progress=True
    )
    docs = document_loader.load()
    _clean_txt_documents(docs)
    return docs


def load_pdf_directory(path: str) -> list[Document]:
    """
    Load PDF documents from a directory and its subdirectories into a list of Document objects.

    Args:
        path (str): The path to the directory containing the PDF documents to load.
    
    Returns:
        list[Document]: A list of loaded Document objects.
    """
    document_loader = DirectoryLoader(
        path,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=True
    )
    docs = document_loader.load()
    _clean_pdf_documents(docs)
    return docs


def split_documents(documents: list[Document], chunk_size: int) -> list[Document]:
    """
    Splits a list of documents into smaller chunks.

    Args:
        documents (list[Document]): A list of documents to be split.

    Returns:
        list[Document]: A list of smaller document chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=True,
        separators=DOCUMENT_SPLIT_SEPARATORS
    )

    chunks = text_splitter.split_documents(documents)
    _clean_chunks(chunks)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    return chunks


# --------------------------------------------------------------------------------------------------
# Functions for summarizing documents.

def get_chunk_summary_prompt(text: str, inflect_engine: inflect.engine, n_sentences) -> str:

    prompt = [
        {"role": "system", "content": f"""Summarize the text below in approximately {inflect_engine.number_to_words(n_sentences)} {inflect_engine.plural("sentence", n_sentences)}. Do not use bullet points.
Your task is to focus on the ideas and information, and not to describe the document. Never use phrases like 'The document is', 'The text discusses', 'this section elaborates' or any other similar metalanguage."""},
        {"role": "user", "content": f"Text: {text}"}

    ]

    return prompt

def summarize(
        chunks: list[Document], 
        granularity: ChunkGranularity, 
        n_sentences: int, 
        llm: LLMClient, 
        progress_bar: bool = True) -> list[Document]:
    
    if granularity not in ChunkGranularity:
        raise ValueError(f"Invalid summary granularity: {granularity}")

    inflect_engine = inflect.engine()

    def process_chunk(chunk):
        chunk = chunk.copy()
        chunk_summary_prompt = get_chunk_summary_prompt(
            text = chunk.page_content,
            inflect_engine = inflect_engine,
            n_sentences = n_sentences)
        
        if not llm.validate_prompt_size(chunk_summary_prompt):
            raise RuntimeError(f"Prompt too large for LLM context window")

        chunk.page_content = llm.generate(chunk_summary_prompt, max_new_tokens = 128*n_sentences)
        chunk.metadata = chunk.metadata.copy()
        chunk.metadata['granularity'] = granularity.value
        return(chunk)
    
    if progress_bar:
        chunks = tqdm(chunks, desc = "Summarizing")
    chunk_summaries = [process_chunk(chunk) for chunk in chunks]
    
    return chunk_summaries


def get_chunk_batch_summary_prompt(text: str, n_items: str, inflect_engine: inflect.engine, n_sentences: int) -> str:
        prompt = [
            {"role": "system", "content": f"""Summarize each item in the YAML data separately using approximately {inflect_engine.number_to_words(n_sentences)} {inflect_engine.plural("sentence", n_sentences)}. Directly output the summaries in the same YAML format using exactly the schema
{"\n".join([f"- summary: \"{inflect_engine.number_to_words(n_sentences)} {inflect_engine.plural("sentence", n_sentences)} long summary of item {i+1} without bullet points\"" for i in range(n_items)])}

Note that there are {n_items} items in the data, so you should output exactly {n_items} summaries.

Your task is to focus on the ideas and information, and not to describe the document. Never use phrases like 'The document is', 'The text discusses', 'this section elaborates' or any other similar metalanguage."""},
            {"role": "user", "content": f"YAML data: {text}"}
        ]

        return prompt


def batch_summarize(
        chunks: list[Document], 
        batch_size: int, 
        granularity: ChunkGranularity, 
        n_sentences: int, 
        llm: LLMClient, 
        progress_bar: bool = True) -> list[Document]:
    
    if batch_size < 2:
        return summarize(chunks, granularity, n_sentences, llm, progress_bar)

    def group_chunks(chunks, batch_size):
        return [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]    

    def process_chunk(chunk, summary):
        chunk = chunk.copy()
        chunk.page_content = summary
        chunk.metadata = chunk.metadata.copy()
        chunk.metadata['granularity'] = "test"
        return(chunk)

    def summarize_group(group):
        chunk_batch = yaml.safe_dump(
            [{'item': re.sub('\n+', ' ', chunk.page_content)} for chunk in group],
            allow_unicode=True,
            default_flow_style=False,
            width=2**16)

        chunk_summary_prompt = get_chunk_batch_summary_prompt(
            text = chunk_batch, 
            n_items = len(group),
            inflect_engine = inflect_engine,
            n_sentences = n_sentences)

        if not llm.validate_prompt_size(chunk_summary_prompt):
            raise RuntimeError(f"Prompt too large for LLM context window")

        response = llm.generate(prompt = chunk_summary_prompt, max_new_tokens = 128*len(group)*n_sentences)

        response = re.sub(r"```\s*yaml\s*|```\s*", "", response)
        try:
            split_response = yaml.safe_load(response)
        except Exception as e:
            try:
                response = re.sub(r'-.*?:', '- summary:', response)
                split_response = yaml.safe_load(response)
            except Exception as e:
                raise RuntimeError(f"Batch summarization failed (invalid YAML response: {response}). The LLM may have been overwhelmed. Try reducing the batch size.")
        
        if len(group) != len(split_response):
            raise RuntimeError(f"Batch summarization failed (length mismatch: {len(group)} chunks vs {len(split_response)} summaries). The LLM may have been overwhelmed. Try reducing the batch size.")

        return [process_chunk(chunk, summary['summary']) for chunk, summary in zip(group, split_response)]

    def process_group(group):
        try:
            logger.debug(f"Attempting to summarize group of size {len(group)}")
            if len(group) > 1:
                batch_summaries = summarize_group(group)
            else:
                batch_summaries = summarize(group, granularity, n_sentences, llm, progress_bar=False)
            return batch_summaries
        except Exception as e:
            logger.warning(f"Error in group of size {len(group)}, splitting further: {str(e)}")
            if len(group) > 1:
                mid = len(group) // 2
                first_half = group[:mid]
                second_half = group[mid:]
                return process_group(first_half) + process_group(second_half)
            else:
                logger.error("Group of size 1 failed, fatal error:" + str(e))
                # If the group cannot be split further, raise the exception
                raise e

    inflect_engine = inflect.engine()

    chunk_groups = group_chunks(chunks, batch_size)

    chunk_summaries = []

    if progress_bar:
        chunk_groups = tqdm(chunk_groups, desc = "Summarizing")

    for group in chunk_groups:
        batch_summaries = process_group(group)
        chunk_summaries.extend(batch_summaries)
    return chunk_summaries


def summarize_chunks(
        chunks: list[Document], 
        n_sentences: int, 
        llm: LLMClient, 
        batch_size: int = 1,
        progress_bar: bool = True) -> list[Document]:
    if batch_size == 1:
        return summarize(chunks, ChunkGranularity.CHUNK_SUMMARY, n_sentences, llm, progress_bar=progress_bar)
    else :
        return batch_summarize(chunks, batch_size, ChunkGranularity.CHUNK_SUMMARY, n_sentences, llm, progress_bar=progress_bar)


def summarize_pages(
        pages: list[Document], 
        n_sentences: int, 
        llm: LLMClient, 
        batch_size: int = 1,
        progress_bar: bool = True) -> list[Document]:
    if batch_size == 1:
        return summarize(pages, ChunkGranularity.PAGE_SUMMARY, n_sentences, llm, progress_bar=progress_bar)
    else:
        return batch_summarize(pages, batch_size, ChunkGranularity.PAGE_SUMMARY, n_sentences, llm, progress_bar=progress_bar)


def recursive_splitter_summarize(
        doc: Document, 
        granularity: ChunkGranularity, 
        n_sentences: int, 
        llm: LLMClient) -> Document:
    
    """
    Recursively splits a document into smaller chunks and summarizes each chunk as soon as it fits 
    the LLM context. Split chunks are merged back together and summarized as a whole, to ensure a 
    final summary length of n_sentences.

    Args:
        doc (Document): The document to be summarized.
        granularity (ChunkGranularity): The granularity to record in the metadata of the returned Document.
        n_sentences (int): The number of sentences to use in the summary.
        llm (LLMClient): The language model used for summarization.
    Returns:
        Document: The summarized document.
    """

    try:
        # Remarks:
        # - summarize takes a list.
        # - the function calling recursive_splitter_summarize is expected to use a progress bar, 
        #   so we disable it here to avoid nested progress bars.
        return summarize([doc], granularity, n_sentences, llm, progress_bar = False)[0]
    except (BadRequestError, RuntimeError) as e:
        # Those errors occur when the context is too large for the LLM.
        # We split the document in two parts. Since we split at end-of-sentence markers, we set the
        # chunk size to 10% more than half the document length to reduce the chances that we get
        # more than 2 splits.
        logger.warning(f"Document size exceeded LLM context in summarization attempt, splitting document.")

        mid = len(doc.page_content)*1.1 // 2
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=mid,
            chunk_overlap=0,
            length_function=len,
            is_separator_regex=True,
            separators=DOCUMENT_SPLIT_SEPARATORS
        )

        splits = text_splitter.split_documents([doc])

        split_summaries = [recursive_splitter_summarize(x, 
                                                        granularity, 
                                                        n_sentences, 
                                                        llm) for x in splits]
        
        # Merge the summaries
        split_summaries[0].page_content = ' '.join([x.page_content for x in split_summaries])

        return summarize([split_summaries[0]], 
                         granularity, 
                         n_sentences, 
                         llm, 
                         progress_bar = False)[0]


def summarize_documents(
        page_summaries: List[Document],
        n_sentences: int,
        llm: LLMClient,
        progress_bar: bool = True) -> List[Document]:

    """
    Summarizes documents based on summaries of their pages. The purpose of this function is to
    summarize large documents using a small context LLM.

    Args:
        page_summaries (List[Document]): A list of Document objects containing page summaries. llm
        llm (LLMClient): An instance of the LLMClient model used for summarization.
        encoding (Encoding): The encoding used by the language model.
        progress_bar (bool): Flag indicating whether to display a progress bar (default: False).
    Returns:
        List[Document]: A list of Document objects representing the summarized documents.

    The function first groups the input summaries by their source and page metadata. It then
    concatenates the page summaries for each source (i.e. document) into a single Document object.
    Next, those Document objects are summarized into 5 sentence summaries. The approach is
    recursive, in case the concatenated page summaries exceed the LLM context, the text is split
    into two parts and the summarization is attempted again.
    """

    # Internal helper function to concatenate the page summaries after grouping page_summaries by
    # source and page.
    def concatenate_summaries(page_data: Dict[str, Dict[str, List[Document]]]) -> Document:   
        # See the comment below (just above the grouped_summaries definition) for an explanation of
        # the argument type hints for page_data.
        document = next(iter(page_data.values()))[0].copy()
        document.page_content = '\n'.join([summary[0].page_content for summary in page_data.values()])
        return document

    # We use defaultdict(lambda: defaultdict(list)) to create a nested dictionary structure.
    # defaultdict is an extension of dict which creates default types for new keys, allowing for
    # more concise code.
    # The outer defaultdict automatically creates an inner defaultdict for any new key.
    # The inner defaultdict automatically creates an empty list for any new key.
    grouped_summaries = defaultdict(lambda: defaultdict(list))
    for summary in page_summaries:
        source = summary.metadata['source']
        page = summary.metadata['page']
        grouped_summaries[source][page].append(summary)

    concatenated_documents = [concatenate_summaries(page_data) for page_data in grouped_summaries.values()]

    if progress_bar:
        concatenated_documents = tqdm(concatenated_documents, desc = "Summarizing documents")

    return [recursive_splitter_summarize(doc, 
                                         ChunkGranularity.DOCUMENT_SUMMARY,
                                         n_sentences,
                                         llm) for doc in concatenated_documents]


def create_collection(
        client: WeaviateClient, 
        collection: str, 
        data_path: str, 
        chunk_size: int, 
        n_sentences: int, 
        llm: LLMClient,
        include_chunk_summaries: bool = False,
        document_summaries_from_page_summaries: bool = False,
        batch_size: int = 1):

    pages = load_directory(data_path)

    chunks = split_documents(pages, chunk_size)

    if include_chunk_summaries:
        chunk_summaries = summarize_chunks(chunks, n_sentences, llm, batch_size=batch_size)
    else:
        chunk_summaries = []

    page_summaries = summarize_pages(pages, n_sentences, llm, batch_size=batch_size)

    document_summaries = summarize_documents(
        page_summaries if document_summaries_from_page_summaries else pages,
        n_sentences,
        llm)

    clear_vector_store(client, collection)

    load_to_vector_store(
        chunks + chunk_summaries + page_summaries + document_summaries,
        client,
        collection)
