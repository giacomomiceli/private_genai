# External imports
import logging
import re
from urllib.parse import quote

from fuzzywuzzy import process

from langchain.schema.document import Document

from typing import List, Dict, Tuple, Any

# Internal imports
from ..utils import HEX_ID_LENGTH


logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------------------
# Functions for cleaning and formatting references

def _parse_page_number(metadata: Dict[str, Any]) -> str:
    """
    Parses the page metadata to return a formatted string indicating the current page number 
    and total pages.

    Args:
        metadata (Dict[str, Any]): A dictionary containing metadata about the page, 
                                    specifically 'page' (current page index) and 
                                    'total_pages' (total number of pages).
    Returns:
        str: A string formatted as 'current_page/total_pages', where current_page 
             is the page number adjusted for 1-based indexing.
    Raises:
        Exception: Logs an error if there is an issue accessing the metadata.
    """

    try:    
        # About the +1: python counts from 0, but pages are conventionally counted from 1.
        return str(int(metadata['page']+1)) + '/' + str(int(metadata['total_pages']))
    except Exception as e:
        logger.error(f"An error occured in _parse_page_number: {str(e)}")
        return 1/1


def _get_file_url(file_path: str, page: str) -> str:
    """
    Generates a URL for accessing a file on a local server.
    Args:
        file_path (str): The path to the file on the local server.
        page (str): The specific page to access within the file.
    Returns:
        str: A URL string that points to the file on the local server, 
             formatted to include the specified page.
    """

    # Assuming the files are served from a local server
    file_subpath = re.sub(r'\\+', '/', file_path)
    full_url = f"http://localhost:8000/{file_subpath}#page={page}"
    return quote(full_url, safe='/:#=')


def get_uuid_to_ref_map(vdb_response: Dict[str, Tuple[str]]) -> dict:
    """
    Generates a mapping of UUIDs to document references from a given vector store output.
    Args:
        vdb_response (list): A list of Document objects.
    Returns:
        dict: A dictionary where keys are UUIDs and values are formatted document references.
    The document reference is formatted as a Markdown link, including the file name and page number.
    """

    uuid_to_doc_map = {}
    for x in vdb_response:
        uuid = x[0].metadata['uuid']
        file_name = re.split(r'[\\/]', x[0].metadata['source'])[-1]
        page = _parse_page_number(x[0].metadata).split('/')[0]
        url = _get_file_url(x[0].metadata['source'], page)
        refnum_template = f"[{{number}}]({url})"
        reference = f"[{file_name}, p. {page}]({url})"
        uuid_to_doc_map[uuid] = tuple([reference, refnum_template])
    return uuid_to_doc_map


# {36} on the full UUID

CONTEXT_ID_PATTERN = re.compile(r'((\[|\(|\{)*(UUID(S)*:*\s*)*' 
                                + rf'\**[0-9a-fA-F-]{{{HEX_ID_LENGTH}}}\**,*\s*' 
                                + r'(\]|\)|\})*)', re.IGNORECASE)

CONTEXT_ID_SUBPATTERN = re.compile(rf'[0-9a-fA-F-]{{{HEX_ID_LENGTH}}}')

def format_references(text: str, context: List[Document], id_dict: Dict[str, str]) -> str:
    """
    Formats references in the given text by replacing context ids with their corresponding document 
    references.
    Parameters:
        text (str): The input text containing UUIDs to be replaced.
        context (List[Document])): The vector store output from which to retrieve the mapping of 
            UUID to references.
        id_dict (Dict[str, str]): A dictionary mapping context ids to vector store UUIDs.
    Returns:
        str: The text with UUIDs replaced by their corresponding document references.
    """

    uuid_to_doc_map = get_uuid_to_ref_map(context)
    # Find all UUIDs in the text
    ref_ids = [x[0] for x in CONTEXT_ID_PATTERN.findall(text)]
    logger.info(f"Found the following UUIDs in the text: {ref_ids}")
    
    ref_list = []

    for ref_id in ref_ids:
        if len(ref_id) > 0:
            match = CONTEXT_ID_SUBPATTERN.search(ref_id)
            if match:
                extracted_id = match.group(0)
                # Use fuzzy matching to find the best match in the dictionary
                best_match = process.extractOne(extracted_id, id_dict.keys(), score_cutoff=90)
                if best_match:
                    reference, refnum_template = uuid_to_doc_map[id_dict[best_match[0]]]
                    text = text.replace(
                        ref_id, 
                        '['+refnum_template.format(number = len(ref_list)+1)+']')
                    ref_list.append(f'[{len(ref_list)+1}] ' + reference)
    
    if len(ref_list) > 0:
        text += "  \n  \n---  \n**References:**  \n  \n"+ '  \n'.join([f'{ref}' for ref in ref_list])
# ---
# **References:** 

# """+ '  \n'.join([f'{ref}' for ref in ref_list])

    return text


def strip_references(text: str) -> str:
    """
    Removes UUID references from the given text.
    This function searches for patterns that match context ids and removes them from the input text.
    The purpose of the function is to clean up the text for updating the conversation history.

    Args:
        text (str): The input string from which UUID references will be removed.
    Returns:
        str: The cleaned text with UUID references removed.
    """

    # Substitute all matches with an empty string
    cleaned_text = CONTEXT_ID_PATTERN.sub('', text)
    return cleaned_text
