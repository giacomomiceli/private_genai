"""
Vector Data Base (VDB) Tools Module

This module provides functions for interacting with vector databases, specifically Weaviate. It
includes functions for connecting to Weaviate instances, loading documents into vector stores, and
managing vector store indices.

Components:
    - get_weaviate_client: Function to obtain a Weaviate client instance
    - load_to_vector_store: Function to load documents into a Weaviate vector store
    - clear_vector_store: Function to clear a specific index in the vector store
    - search_vector_store: Function to search for documents in a Weaviate vector store
    - update_vector_store: Function to update documents in a Weaviate vector store
    - delete_from_vector_store: Function to delete documents from a Weaviate vector store

Typical usage example:
    from vdb import get_weaviate_client, load_to_vector_store, clear_vector_store 
    from langchain.schema.document import Document

    client = get_weaviate_client(local=True) documents = [Document(content="Sample document
    content")] load_to_vector_store(documents, client, "sample_index")
    
    # Clear the vector store index clear_vector_store(client, "sample_index")

Note:
    This module requires the 'weaviate-client', 'langchain', 'langchain-weaviate' and
    'langchain-openai' libraries to be installed. 
    If you intend to use a Weaviate cloud instance (get_weaviate_client(local=False)), Ensure the 
    environment variables WEAVIATE_URL and WEAVIATE_API_KEY are set when connecting to a Weaviate 
    Cloud Service (WCS) instance.
"""

# Basic imports
import os
import logging
import yaml

# Class imports
from langchain.schema.document import Document

# Vendor solution specific imports
import weaviate
from weaviate.client import WeaviateClient
import weaviate.classes as wvc
from weaviate.collections.classes.filters import _FilterValue
from weaviate.collections.classes.internal import QueryReturn
from weaviate.collections.collection import Collection

from langchain_openai import OpenAIEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore

from typing import List, Dict, Any, Tuple, Iterator

logger = logging.getLogger(__name__)

def get_weaviate_client(local: bool = True) -> WeaviateClient:
    """
    Get a Weaviate client instance.

    This function returns a Weaviate client instance based on the specified mode. If `local` is
    True, it connects to a local Weaviate instance. Otherwise, it connects to a Weaviate Cloud
    Service (WCS) instance using environment variables WEAVIATE_URL for the cluster URL and
    WEAVIATE_API_KEY for the API key. The user must take care of setting those environment variables
    before calling this function.

    Args:
        local (bool): Flag indicating whether to connect to a local instance (default: True).

    Returns:
        WeaviateClient: A Weaviate client instance.

    Raises:
        ValueError: If `local` is False and the required environment variables `WEAVIATE_URL` or
            `WEAVIATE_API_KEY` are not set.
    """
    if local:
        return weaviate.connect_to_local()
    else:
        weaviate_url = os.getenv('WEAVIATE_URL')
        weaviate_api_key = os.getenv('WEAVIATE_API_KEY')
        if not weaviate_url:
            raise ValueError("Environment variable WEAVIATE_URL must be set when local is False.")
        if not weaviate_api_key:
            raise ValueError("Environment variable WEAVIATE_API_KEY must be set when local is False.")
        return weaviate.connect_to_wcs(
            cluster_url=os.getenv('WEAVIATE_URL'),
            auth_credentials=weaviate.auth.AuthApiKey(os.getenv('WEAVIATE_API_KEY'))
        )
    
def close_client(client: WeaviateClient) -> None:
    """
    Closes the Weaviate client.

    Args:
        client (WeaviateClient): the Weaviate client to be closed.
    """
    logger.info("Closing Weaviate client...")
    client.close()
    
def get_vector_store(
        client: WeaviateClient, 
        index_name: str) -> WeaviateVectorStore:
    """
    Returns a WeaviateVectorStore object initialized with the given client, index name,
    OpenAIEmbeddings, and text key.
    
    Args:
        client (WeaviateClient): the Weaviate client to use. index_name (str): the name of the
        Weaviate collection.

    Returns:
        WeaviateVectorStore: The initialized WeaviateVectorStore object.
        
    Raises:
        WeaviateClosedClientError: If the client is closed.
    """
    return WeaviateVectorStore(
        client=client, 
        index_name=index_name, 
        embedding=OpenAIEmbeddings(), 
        text_key="text")

def load_to_vector_store(
        documents: list[Document], 
        client: WeaviateClient,
        index_name: str) -> WeaviateVectorStore:
    """
    Load the given documents into a Weaviate vector store.

    Args:
        documents (list[Document]): A list of documents to be loaded into the vector store.
        client (WeaviateClient): The Weaviate client used to interact with the 
            Weaviate server.
        index_name (str): The name of the index in the vector store.

    Returns:
        WeaviateVectorStore: The loaded Weaviate vector store.
    """
    return WeaviateVectorStore.from_documents(
        documents,
        embedding=OpenAIEmbeddings(),
        client=client,
        index_name=index_name)

def clear_vector_store(client: WeaviateClient, index_name: str) -> None:
    """
    Clears the vector store for the specified index.

    Args:
        client (WeaviateClient): The client object used to interact with the vector 
            store.
        index_name (str): The name of the index to clear.

    Returns:
        None
    """
    client.collections.delete(index_name)


class FilterSpecs:
    def __init__(self, specs: Dict[str, Tuple[Any, bool]]):
        """
        The key is the name of the weaviate object property to filter on, and the value is a list containing the required value of the property, and a bool which determines whether the filter is `equal` (if True) or `not_equal` (if False).
        """
        self.filters: Dict[str, Tuple[Any, bool]] = specs

    def __iter__(self) -> Iterator[Tuple[str, Any, bool]]:
        for key, (value, equal) in self.filters.items():
            yield key, value, equal


def get_weaviate_filter(filters: FilterSpecs) -> _FilterValue:
    """
    Create Weaviate filters based on the given dictionary of filter specifications. Each entry in
    the dictionary specifies a filter, and the filters are combined using the logical AND.

    Args:
        filters (FilterSpecs): Specifications of the filters.

    Returns:
        _FilterValue: The Weaviate filters created based on the given dictionary.
    """

    if (filters is None):
        return None
    else:
        weaviate_filters = []
        for key, value, equal in filters:
            if equal:
                weaviate_filters.append(wvc.query.Filter.by_property(key).equal(value))
            else:
                weaviate_filters.append(wvc.query.Filter.by_property(key).not_equal(value))
        return wvc.query.Filter.all_of(weaviate_filters)


def search_vector_store(
        vector_store: WeaviateVectorStore, 
        query: str, 
        k: int, 
        keyword_search_weight: float = 0, 
        filters: FilterSpecs = None) -> list[Document]:
    """
    Searches the vector store using the specified query and returns a list of documents.
    Args:
        vector_store (WeaviateVectorStore): The vector store to search in.
        query (str): The query string to search for.
        k (int): The maximal number of documents to retrieve.
        keyword_search_weight (float, optional): The weight of the keyword search. Defaults to 0. 
            It is passed to vector_store.similarity_search_with_score() as parameter alpha.
        filters (Dict[str, List[str, bool]]): A dictionary containing specifications of the 
            filters. The key is the name of the weaviate object property to filter on, and the value 
            is a list containing the required value of the property, and a bool which determines 
            whether the filter is `equal` (if True) or `not_equal` (if False).
    Returns:
        list[Document]: A list of documents matching the search query.
    """

    return vector_store.similarity_search_with_score(
        query = query,
        k = k,
        alpha = keyword_search_weight,
        filters = get_weaviate_filter(filters),
        return_uuids = True
    )


def weaviate_query_return_to_langchain_documents(result: QueryReturn) -> List[Document]:
    def process_item(item):
        md = item.properties.copy()
        page_content = md.pop('text')
        md['uuid'] = str(item.uuid)
        return [Document(page_content, metadata = md), 1]
    
    return [process_item(item) for item in result.objects]


def fetch_from_collection(
        collection: Collection,
        limit: int = None,
        filter_specs: FilterSpecs = None) -> List[Document]:

    try:
        return weaviate_query_return_to_langchain_documents(
            collection.query.fetch_objects(
                limit = limit,
                filters = get_weaviate_filter(filter_specs)))
    except Exception as e:
        print(f"Error fetching from collection {collection}: {e}")
        return []


def initialize_use_case_vs(client: WeaviateClient) -> WeaviateVectorStore:
    logger.info("Initializing the use case vector store...")
    use_cases_folder = 'use_cases'
    use_cases_content = []
    for filename in os.listdir(use_cases_folder):
        if filename.endswith('.yaml'):
            with open(os.path.join(use_cases_folder, filename), 'r') as file:
                use_case = yaml.safe_load(file)
                use_cases_content.append(use_case)

    use_cases = [
        Document(
            page_content=avatar,
            metadata = {
                'use_case': use_case['label'],
                'use_case_description': use_case['description']
            }) for use_case in use_cases_content for avatar in use_case['avatars'] ]


    clear_vector_store(client, "RAG_use_cases")

    return load_to_vector_store(
        use_cases,
        client,
        "RAG_use_cases")