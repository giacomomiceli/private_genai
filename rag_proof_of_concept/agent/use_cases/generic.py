# External imports
from langchain_weaviate.vectorstores import WeaviateVectorStore

from streamlit.delta_generator import DeltaGenerator

from typing import List, Dict

import logging

# Internal imports
from ...readers import ChunkGranularity
from ..refs import format_references
from ...context import get_context, format_context, get_context_sample
from ...utils import LLMClient, get_formatted_current_date, read_json, HEX_ID_LENGTH


logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------------------
# Assess the query to select the appropriate generic use case procedure

def get_query_assessment_prompt(question: str, context: str) -> str:
    """
    Returns a prompt template for assessing a query to determine the procedure for answering.
    Returns:
        str: The prompt template.
    """
    
    prompt = [
        {"role": "system", "content": f"""Your task is to pre-assess the query. Answer in json format, including the keys:
- "context_relevant": "True" (if the context sample is related and helpful to answer the question) or "False" (if you have to rely on your general knowledge to answer).
- "granularity": The context granularity which is most helpful.
- "focus": "concepts" (if the query meaning is not sensitive to the exact word choice and could be easily rephrased with synonyms) or "keywords" (if the query focuses on individual keywords that do not have widely accepted synonyms)."""},
        {"role": "user", "content": f"""Query: {question}
Context: {context}"""}
    ]

    return prompt


def assess_query(
        query_text: str,
        context: str,
        llm: LLMClient) -> Dict[str, str]:
    
    prompt_query_assessment =  get_query_assessment_prompt(
        question=query_text,
        context=context)

    try:
        response = llm.generate(prompt=prompt_query_assessment, max_new_tokens=512)
        out = read_json(response)
        out['context_relevant'] = str(out['context_relevant']).capitalize()
        return out
    except Exception as e:
        logger.error(f"An error occured in assess_query: {str(e)}")
        return {'context_relevant': 'False',
                'granularity': ChunkGranularity.RAW.value,
                'focus': 'concepts'}

# --------------------------------------------------------------------------------------------------
# Generic non-RAG answer
def get_simple_answer_prompt(question: str, history: str, style: str) -> str:
    """
    Returns a prompt template for answering a query not related to any document in the vector
    store.
    Returns:
        str: The prompt template.
    """

    prompt = [
        {"role": "system", "content": f"""Today is {get_formatted_current_date()}. Answer the question below adopting a {style} style. Make sure it continues the conversation flow based on the provided conversation history. Try your best to give an answer but disclaim prominently that it is based on your general knowledge."""},

        {"role": "user", "content": f"""Question: {question}
Conversation history: {history}"""}
    ]

    return prompt


def get_simple_answer(
        query_text: str,
        history: str,
        style: str,
        llm: LLMClient) -> List[str]:

    prompt_generic_answer = get_simple_answer_prompt(
        question=query_text, 
        history=history, 
        style=style)

    try:
        response = llm.generate(prompt=prompt_generic_answer, max_new_tokens=512)
        
    except Exception as e:
        logger.error(f"An error occured in get_simple_answer: {str(e)}")
        response = f"An error occurred: {str(e)}"

    return response, response


# --------------------------------------------------------------------------------------------------
# RAG answers

def get_rag_answer_prompt(question: str, style: str, max_new_tokens: int, context: str = None) -> str:
    """
    Returns a prompt template for answering a question using RAG.

    Returns:
        str: The prompt template.
    """

    if not context:
        context = "{context}"

    prompt = [
        {"role": "system", "content": f"""Today is {get_formatted_current_date()}. Use my documents (which are ordered by decreasing relevance) and any additional knowledge you may have to answer the question. Correct any contradictions or erroneous knowledge implied by the question. 
         
For each point in your answer you must write the corresponding {HEX_ID_LENGTH} hexadecimal digits of the context uuid in brackets one at a time like this: [a1b2c3]. Do not write the word uuid and do not write any other tokens than the hexadecimal uuids inside the brackets.

Use Markdown to format your answer. Use a {style} writing style. Limit the size of your answer to {max_new_tokens} new tokens."""},

        {"role": "user", "content": f"""Question: {question}
Documents: {context}"""}]

    return prompt


def get_rag_answer(
        query_text: str, 
        context_query: str, 
        tokens_target: int, 
        expected_chunk_size: Dict[ChunkGranularity, float],
        knowledge_vs: WeaviateVectorStore,
        style: str,
        llm: LLMClient,
        keyword_search_weight: float = 0,
        context_granularity_allocation: Dict[ChunkGranularity, float] = {ChunkGranularity.RAW: 1}) -> List[str]:

    MAX_NEW_TOKENS = 1024

    context = get_context(
        prompt=get_rag_answer_prompt(question=query_text, style=style, max_new_tokens=round(1.25*MAX_NEW_TOKENS)),
        context_query=context_query,
        tokens_target=tokens_target, 
        expected_chunk_size=expected_chunk_size,  
        knowledge_vs=knowledge_vs, 
        llm=llm, 
        keyword_search_weight=keyword_search_weight,
        granularity_allocation=context_granularity_allocation)
    
    formatted_context, id_dict = format_context(context)

    prompt_answer = get_rag_answer_prompt(
        question=query_text, 
        style=style, 
        max_new_tokens=MAX_NEW_TOKENS,
        context=formatted_context)

    try:
        response = llm.generate(prompt=prompt_answer, max_new_tokens=MAX_NEW_TOKENS)
    except Exception as e:
        logger.error(f"An error occured in get_rag_answer: {str(e)}")
        response = f"An error occurred: {str(e)}" 
    
    return response, format_references(response, context, id_dict)
    
# --------------------------------------------------------------------------------------------------
# Overall procedure

def get_generic_answer(
        query_text: str,
        context_query: str,
        tokens_target: int,
        expected_chunk_size: Dict[ChunkGranularity, float],
        knowledge_vs: WeaviateVectorStore,
        history: str,
        style: str,
        llm: LLMClient,
        msg_container: DeltaGenerator
) -> List[str]:
    
    # Extract a context sample from the vector store based on the related concepts
    context_sample = get_context_sample(
        knowledge_vs=knowledge_vs, 
        query=context_query)

    formatted_context_sample, _ = format_context(context_sample, include_granularity=True)

    # Assess the query to determine the whether the context is relevant, the optimal context
    # granularity, and whether to use dense or sparse vector store search.
    assessment_outcome = assess_query(
        query_text=query_text, 
        context=formatted_context_sample, 
        llm=llm)

    logger.info("Context relevant: " + assessment_outcome['context_relevant'])
    logger.info("Optimal context granularity: " + assessment_outcome['granularity'])
    logger.info("Focus on: " + assessment_outcome['focus'])

    # Set the keyword search weight based on the focus of the query (concepts or keywords)
    if (assessment_outcome['focus'] == 'concepts'):
        keyword_search_weight = 0.1
    else:
        keyword_search_weight = 0.5

    if (assessment_outcome['context_relevant'] == 'False'):
        msg_container.write("Preparing an answer (question unrelated to user documents)...")
        raw_answer, formatted_answer = get_simple_answer(
            query_text=query_text, 
            history=history,
            style=style,
            llm=llm)
    # Else, the query is relevant to the user documents and we are in a RAG situation
    else: 
        msg_container.write("Preparing an answer...")     

        # Allocate the context granularity to retrieve from the vector store based on the assessment
        # outcome
        match assessment_outcome['granularity']:
            case ChunkGranularity.RAW.value:
                context_granularity_allocation = {
                    ChunkGranularity.RAW: 0.8,
                    ChunkGranularity.PAGE_SUMMARY: 0.1,
                    ChunkGranularity.DOCUMENT_SUMMARY: 0.1}
            case ChunkGranularity.PAGE_SUMMARY.value:
                context_granularity_allocation = {
                    ChunkGranularity.RAW: 0.3,
                    ChunkGranularity.PAGE_SUMMARY: 0.6,
                    ChunkGranularity.DOCUMENT_SUMMARY: 0.1}
            case ChunkGranularity.DOCUMENT_SUMMARY.value:
                context_granularity_allocation = {
                    ChunkGranularity.RAW: 0.2,
                    ChunkGranularity.PAGE_SUMMARY: 0.4,
                    ChunkGranularity.DOCUMENT_SUMMARY: 0.4}
            case _:
                context_granularity_allocation = {
                    ChunkGranularity.RAW: 0.33,
                    ChunkGranularity.PAGE_SUMMARY: 0.34,
                    ChunkGranularity.DOCUMENT_SUMMARY: 0.33}

        raw_answer, formatted_answer = get_rag_answer(
            query_text=query_text, 
            context_query=context_query, 
            tokens_target=tokens_target, 
            expected_chunk_size=expected_chunk_size,
            knowledge_vs=knowledge_vs,
            style=style,
            llm=llm,
            keyword_search_weight=keyword_search_weight,
            context_granularity_allocation=context_granularity_allocation)

    return [raw_answer, formatted_answer]