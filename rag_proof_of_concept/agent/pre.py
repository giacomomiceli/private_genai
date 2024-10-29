# External imports
import logging

from langchain_weaviate.vectorstores import WeaviateVectorStore

from typing import List, Dict

# Internal imports
from ..utils import get_formatted_current_date, read_json, LLMClient

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------------------
# Functions for pre-processing and assessing queries

def get_query_preprocessing_prompt(question: str, history: str) -> List[Dict[str, str]]:

    prompt = [
        {"role": "system", "content": f"""Today is {get_formatted_current_date()}. You are an assistant who answers queries based on chunks retrieved from a document store. Answer in json format including the following keys:
-'implied_query' (Rephrase the query below in light of the conversation history without changing the voice, tense or mood. If it is a question, it should remain a question. If it is a declaration, it should remain a declaration. If it is an injunction, it should remain an injuction.).
-'query_type' (Provide a concise, abstract description of the purpose of the request. Special cases are 'Request for an overview of the document store', 'Request about a specific topic', 'Request to count documents relevant to a topic or keyword', 'Question or complaint about the performance or identity of the agent'.)
-'related_concepts (State concepts that are related to the query, just state them without using the word concepts or other metalanguage. You may not give any other answer)"""},

    {"role": "user", "content": f"""Query: {question}
Conversation history: {history}."""}]

    return prompt


def preprocess_query(
        query_text: str,
        history: str, 
        llm: LLMClient,
        use_case_vs: WeaviateVectorStore):
    
    prompt_query_preprocessing = get_query_preprocessing_prompt(
        question=query_text,
        history=history
    )

    try:
        response = llm.generate(prompt=prompt_query_preprocessing, max_new_tokens=128)
        preprocessed_query = read_json(response)

        if type(preprocessed_query['related_concepts']) == list:
            preprocessed_query['related_concepts'] = ", ".join(preprocessed_query['related_concepts'])

        preprocessed_query['query_type'] = identify_use_case(use_case_vs, preprocessed_query['query_type'])

        return preprocessed_query

    except Exception as e:
        logger.error(f"An error occured in preprocess_query: {str(e)}")
        return {'implied_query': query_text, 
                'related_concepts': "",
                'query_type': 'generic'}


def identify_use_case(
        use_case_vs: WeaviateVectorStore, 
        query: str) -> str: 

    try:
        best_match = use_case_vs.similarity_search_with_score(query = query, k = 1)
        logger.info(f"Query type description from preprocessing: {query}")
        logger.info(f"Matched use case avatar: {best_match[0][0].page_content}")
        logger.info(f"Use case: {best_match[0][0].metadata['use_case']}")
        logger.info(f"Best similarity score in use case identification: {best_match[0][1]}")
        if best_match:
            if best_match[0][1] > 0.85:
                return best_match[0][0].metadata['use_case']
            else:
                return 'generic'
        else:
            return 'generic'
    except Exception as e:
        logger.error(f"Error when identifying the agent use case: {str(e)}")
        return 'generic'