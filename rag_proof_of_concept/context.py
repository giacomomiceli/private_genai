# External imports
import logging
import numpy as np

from langchain.schema.document import Document

from langchain_weaviate.vectorstores import WeaviateVectorStore

from typing import List, Dict, Union

# Internal imports
from .utils import generate_hex_ids, ChunkGranularity, LLMClient
from .vdb import search_vector_store, fetch_from_collection, FilterSpecs

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------------------
# Functions for managing LLM input token usage

def get_mean_context_length(context: List[Document]) -> float:
    return np.mean([len(str(x)) for x in context])


def format_context_element_sizes(
        chunk_size: float,
        summary_size: float) -> Dict[ChunkGranularity, float]:
    return {
        ChunkGranularity.RAW: chunk_size,
        ChunkGranularity.CHUNK_SUMMARY: summary_size,
        ChunkGranularity.PAGE_SUMMARY: summary_size,
        ChunkGranularity.DOCUMENT_SUMMARY: summary_size
    }


def get_context_element_sizes(knowledge_connection: WeaviateVectorStore) -> Dict[ChunkGranularity, float]:
    raw_sample = fetch_from_collection(
        knowledge_connection,
        500,
        FilterSpecs({"granularity": [ChunkGranularity.RAW.value, True]}))
    
    raw_mean = np.mean([len(x[0].page_content) for x in raw_sample])

    summary_sample = fetch_from_collection(
        knowledge_connection,
        500,
        FilterSpecs({"granularity": [ChunkGranularity.RAW.value, False]}))

    summary_mean = np.mean([len(x[0].page_content) for x in summary_sample])

    return format_context_element_sizes(raw_mean, summary_mean)

# --------------------------------------------------------------------------------------------------
# Functions for preprocessing context

def format_context(vdb_response: List[Document], include_granularity: bool = False) -> list[dict]:
    """
    Formats the context extracted from the vector store for use in a prompt.
    Args:
        vdb_response (list): A list of Document objects.
        include_granularity (bool): A flag indicating whether to include granularity information in 
            the output. Defaults to False.
    Returns:
        list[dict]: A list of dictionaries, each containing a unique UUID, content, and optionally 
        granularity information.
    """

    unique_uuids = set()
    result = []

    short_ids = generate_hex_ids(len(vdb_response))
  
    id_dict = {x[0].metadata['uuid']: shortid for x, shortid in zip(vdb_response, short_ids)}
    reverse_id_dict = {value: key for key, value in id_dict.items()}

    for x in vdb_response:
        uuid = x[0].metadata['uuid']
        if uuid not in unique_uuids:
            unique_uuids.add(uuid)
            if include_granularity:
                result.append({
                    'uuid': id_dict[uuid],
                    'content': x[0].page_content,
                    'granularity': x[0].metadata['granularity']
                })
            else:
                result.append({
                    'uuid': id_dict[uuid],
                    'content': x[0].page_content
                })

    return result, reverse_id_dict


# --------------------------------------------------------------------------------------------------
# Functions for retrieving context

def get_context_sample(knowledge_vs: WeaviateVectorStore, query: str):
    try:
        return [result
                for granularity in ChunkGranularity
                for result in search_vector_store(
                    vector_store = knowledge_vs,
                    query=query, 
                    k=1,
                    keyword_search_weight = 0.5,
                    filters=FilterSpecs({"granularity": [granularity.value, True]}))]
    except Exception as e:
        logger.error(f"An error occured in get_context_sample: {str(e)}")
        return []
    

def get_context_elements(
        knowledge_vs: WeaviateVectorStore, 
        query: str, 
        n: Dict[ChunkGranularity, int], 
        keyword_search_weight: float = 0):
    try:
        return [result
                for granularity in n.keys()
                for result in search_vector_store(
                    vector_store=knowledge_vs,
                    query=query, 
                    k=n[granularity],
                    keyword_search_weight = keyword_search_weight,
                    filters=FilterSpecs({"granularity": [granularity.value, True]}))]
    except Exception as e:
        logger.error(f"An error occured in get_context_elements: {str(e)}")
        return []


def get_chunks(
        knowledge_vs: WeaviateVectorStore, 
        query: str, 
        n: int, 
        keyword_search_weight: float = 0):
    try:
        return search_vector_store(
            vector_store=knowledge_vs,
            query=query, 
            k=n,
            keyword_search_weight = keyword_search_weight,
            filters=FilterSpecs({"granularity": [ChunkGranularity.RAW.value, True]}))
    except Exception as e:
        logger.error(f"An error occured in get_chunks: {str(e)}")
        return []
    

def get_chunk_summaries(
        knowledge_vs: WeaviateVectorStore, 
        query: str, 
        n: int, 
        keyword_search_weight: float = 0):
    try:
        return search_vector_store(
            vector_store=knowledge_vs,
            query=query, 
            k=n,
            keyword_search_weight = keyword_search_weight,
            filters=FilterSpecs({"granularity": [ChunkGranularity.CHUNK_SUMMARY.value, True]}))
    except Exception as e:
        logger.error(f"An error occured in get_chunk_summaries: {str(e)}")
        return []
    

def get_page_summaries(
        knowledge_vs: WeaviateVectorStore, 
        query: str, 
        n: int, 
        keyword_search_weight: float = 0):
    try:
        return search_vector_store(
            vector_store=knowledge_vs,
            query=query, 
            k=n,
            keyword_search_weight = keyword_search_weight,
            filters=FilterSpecs({"granularity": [ChunkGranularity.PAGE_SUMMARY.value, True]}))
    except Exception as e:
        logger.error(f"An error occured in get_page_summaries: {str(e)}")
        return []
    

def get_document_summaries(
        knowledge_vs: WeaviateVectorStore, 
        query: str, 
        n: int, 
        keyword_search_weight: float = 0):
    try:
        return search_vector_store(
            vector_store=knowledge_vs,
            query=query, 
            k=n,
            keyword_search_weight = keyword_search_weight,
            filters=FilterSpecs({"granularity": [ChunkGranularity.DOCUMENT_SUMMARY.value, True]}))
    except Exception as e:
        logger.error(f"An error occured in get_document_summaries: {str(e)}")
        return []
    

def estimate_chunk_count(n_tokens: int, context_length: int, llm: LLMClient) -> float:
    return round(llm.token_count_to_str_length(n_tokens) / context_length)


def get_context(
        prompt: Union[str, List[Dict[str, str]]],
        context_query: str,
        tokens_target: int, 
        expected_chunk_size: Dict[ChunkGranularity, float],  
        knowledge_vs: WeaviateVectorStore, 
        llm: LLMClient, 
        keyword_search_weight: float = 0,
        granularity_allocation: Dict[ChunkGranularity, float] = {ChunkGranularity.RAW: 1}):
    
    # Check if the sum of granularity_allocation values equals 1
    allocation_values = np.array(list(granularity_allocation.values()))
    allocation_sum = allocation_values.sum()

    if np.abs(allocation_sum-1) > 0.001:
        logger.warning(f"Granularity allocation values sum to {allocation_sum}, normalizing them.")
        granularity_allocation = {granularity: value / allocation_sum for granularity, value in granularity_allocation.items()}

    weighted_chunk_size = sum(
        granularity_allocation[granularity] * expected_chunk_size[granularity] 
            for granularity in granularity_allocation.keys())

    #try:
    n_chunks_0 = estimate_chunk_count(tokens_target, weighted_chunk_size, llm)
    
    logger.info(f"Initial estimated chunk count: {n_chunks_0}")

    context = get_context_elements(
        knowledge_vs=knowledge_vs, 
        query=context_query,
        n={granularity: round(value * n_chunks_0 * 2) for granularity, value in granularity_allocation.items()},
        keyword_search_weight = keyword_search_weight)
    
    formatted_context, _ = format_context(context)

    mean_context_length = get_mean_context_length(formatted_context)

    n_chunks_1 = estimate_chunk_count(tokens_target, mean_context_length, llm)
    logger.info(f"Refined chunk count estimate: {n_chunks_1}")

    if n_chunks_1 > n_chunks_0 * 1.8:
        context = get_context_elements(
            knowledge_vs=knowledge_vs, 
            query=context_query,
            n={granularity: round(value * n_chunks_1 * 2) for granularity, value in granularity_allocation.items()},
            keyword_search_weight = keyword_search_weight)

    measured_tokens, _ = llm.measure_prompt(
        [{**message, 
            'content': message['content'].format(context=formatted_context[0:n_chunks_1])} 
            if 'content' in message else message
            for message in prompt])
            
    logger.info(f"Resulting tokens: {measured_tokens}")

    n_chunks_final = round(tokens_target / measured_tokens * n_chunks_1)
    logger.info(f"Final chunk count: {n_chunks_final}")

    final_tokens, _ = llm.measure_prompt(
        [{**message, 
            'content': message['content'].format(context=formatted_context[0:n_chunks_final])} 
            if 'content' in message else message
            for message in prompt])

    logger.info(f"Final resulting tokens: {final_tokens} (target: {tokens_target})")

    return context[0:n_chunks_final]
    
    # except Exception as e:
    #     logger.error(f"An error occured in get_context: {str(e)}")
    #     return "An error occured in get_context: " + str(e)
