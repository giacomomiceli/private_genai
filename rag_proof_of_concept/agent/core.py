# External imports
from langchain_weaviate.vectorstores import WeaviateVectorStore
from weaviate.collections.collection import Collection
from streamlit.delta_generator import DeltaGenerator

from typing import List, Dict

import logging

# Internal imports
from ..utils import ChunkGranularity, LLMClient
from .pre import preprocess_query
from . import use_cases

logger = logging.getLogger(__name__)


def get_answer(
        query_text: str,
        tokens_target: int,
        expected_chunk_size: Dict[ChunkGranularity, float],
        knowledge_vs: WeaviateVectorStore,
        knowledge_connection: Collection,
        use_case_vs: WeaviateVectorStore,
        history: str,
        style: str,
        llm: LLMClient,
        msg_container: DeltaGenerator
) -> List[str]:
    # Preprocess the query to extract related concepts and rephrase the query in light of
    # the conversation history
    msg_container.write("Understanding the query...")
    query_assessment = preprocess_query(
        query_text=query_text, 
        history=history, 
        llm=llm,
        use_case_vs=use_case_vs)
    
    logger.info("Rephrased query: " + query_assessment['implied_query'])
    logger.info("Related concepts: " + query_assessment['related_concepts'])
    
    match query_assessment['query_type']:
        case 'count_documents':
            raw_answer, formatted_answer = use_cases.count_documents.get_count_documents_answer(
                query_text=query_assessment['implied_query'],
                context_query=query_assessment['related_concepts'],
                knowledge_vs=knowledge_vs,
                style=style,
                llm=llm,
                msg_container=msg_container)
        case 'collection_overview':
            raw_answer, formatted_answer = use_cases.collection_overview.get_collection_overview_answer(
                query_text=query_assessment['implied_query'],
                knowledge_connection=knowledge_connection,
                style=style,
                llm=llm,
                msg_container=msg_container)
        case 'self':
            raw_answer, formatted_answer = use_cases.self.get_self_answer(
                query_text=query_assessment['implied_query'],
                history=history,
                style=style,
                llm=llm,
                msg_container=msg_container
            )
        case _:
            raw_answer, formatted_answer = use_cases.generic.get_generic_answer(
                query_text=query_assessment['implied_query'],
                context_query=query_assessment['related_concepts'],
                tokens_target=tokens_target,
                expected_chunk_size=expected_chunk_size,
                knowledge_vs=knowledge_vs,
                history=history,
                style=style,
                llm=llm,
                msg_container=msg_container)

    return raw_answer, formatted_answer
