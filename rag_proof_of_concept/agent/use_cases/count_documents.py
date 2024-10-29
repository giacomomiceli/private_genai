# External Imports
import logging
from typing import List

from langchain_weaviate.vectorstores import WeaviateVectorStore

from streamlit.delta_generator import DeltaGenerator

# Pluralization, number to word conversion for building prompts
import inflect

# Internal Imports
from ...context import format_context, get_document_summaries
from ..refs import format_references
from ...utils import LLMClient, get_formatted_current_date, HEX_ID_LENGTH


logger = logging.getLogger(__name__)


def get_count_documents_prompt(question: str, context: str, style: str, num_docs: int) -> str:

    prompt = [
        {"role": "system", "content": f"""Today is {get_formatted_current_date()}. Your task is to count documents relating to a specific topic. Answer the query using the provided context. Give a concise description of the relevant documents. For each document print the corresponding {HEX_ID_LENGTH} hexadecimal digits of the context uuid in braces. Use a {style} writing style."""},

        {"role": "user", "content": f"""Query: {question}
Context: {context}
Number of documents in context: {num_docs}."""}]

    return prompt


def get_count_documents_answer(
        query_text: str, 
        context_query: str,
        knowledge_vs: WeaviateVectorStore,
        style: str,
        llm: LLMClient,
        msg_container: DeltaGenerator) -> List[str]:

    inflect_engine = inflect.engine()

    msg_container.write("Preparing an answer (counting documents)...") 

    relevant_docs = get_document_summaries(
        knowledge_vs = knowledge_vs, 
        query= context_query, 
        n = None, 
        keyword_search_weight = 0.5)

    relevant_docs = [doc for doc in relevant_docs if doc[1] > 0.3]

    formatted_context, id_dict = format_context(relevant_docs)

    prompt_answer = get_count_documents_prompt(
        question=query_text,
        context=formatted_context,
        style=style,
        num_docs=inflect_engine.number_to_words(len(relevant_docs)))

    try:
        response = llm.generate(prompt=prompt_answer, max_new_tokens=512)
    except Exception as e:
        logger.error(f"An error occured in get_rag_answer: {str(e)}")
        response = f"An error occurred: {str(e)}" 
    
    return response, format_references(response, relevant_docs, id_dict)
