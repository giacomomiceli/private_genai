# External Imports
import logging
from typing import List

from weaviate.collections.collection import Collection

from streamlit.delta_generator import DeltaGenerator

# Internal Imports
from ...utils import ChunkGranularity, LLMClient, get_formatted_current_date, HEX_ID_LENGTH
from ...context import format_context
from ..refs import format_references
from ...vdb import fetch_from_collection, FilterSpecs

logger = logging.getLogger(__name__)

def get_collection_overview_prompt(question: str, context: str, style: str) -> str:
    prompt = [
        {"role": "system", "content": f"""Today is {get_formatted_current_date()}. Your task is to provide an overview of all available knowledge. Answer the question using the provided context, which contains document summaries. You may cite the corresponding {HEX_ID_LENGTH} hexadecimal digits of the context uuid in brackets like this: [a1b2c3]. Do not write the word uuid and do not write any other tokens than the hexadecimal uuids inside the brackets. Use a {style} writing style."""},

        {"role": "user", "content": f"""Question: {question}
Context: {context}"""}]

    return prompt

def get_collection_overview_answer(
        query_text: str, 
        knowledge_connection: Collection,
        style: str,
        llm: LLMClient,
        msg_container: DeltaGenerator) -> List[str]:

    msg_container.write("Preparing an answer (knowledge base overview)...") 

    all_docs = fetch_from_collection(
        collection=knowledge_connection,
        filter_specs=FilterSpecs({"granularity": [ChunkGranularity.DOCUMENT_SUMMARY.value, True]}))

    formatted_context, id_dict = format_context(all_docs)

    prompt_answer = get_collection_overview_prompt(
        question=query_text,
        context=formatted_context,
        style=style)

    try:
        response = llm.generate(prompt=prompt_answer, max_new_tokens=512)
    except Exception as e:
        logger.error(f"An error occured in get_rag_answer: {str(e)}")
        response = f"An error occurred: {str(e)}" 
    
    return response, format_references(response, all_docs, id_dict)