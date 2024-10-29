# External imports
import logging

# Internal imports
from ..utils import LLMClient

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------------------
# Functions for managing the conversation history

def get_context_summary_prompt(context: str) -> str:
    """
    Returns a prompt template for generating a compressed summary of a given context.

    Returns:
        str: The prompt.
    """

    prompt = [
        {"role": "system", "content": """Compress the context below as much as possible but keep the <User: question> <Assistant: answer> conversation flow avoiding any unmatched angle brackets. Prioritize preserving the meaning of the messages over their structure and format. In particular, do not use bullet points in the summary, only unstructured text.
The goal is use as few tokens as possible while still being able to reconstruct the original meaning. The compressed context will be concatenated with previous entries and provided to you in a future session to continue the conversation.
Print only the compressed summary of the context. Make sure to match all < and >. Include line breaks after every >."""},
{"role": "user", "content": f"""Context: {context}"""}]

    return prompt


def update_conversation_context(history: str, question: str, answer: str, llm: LLMClient) -> str:
    updated_context = history + '<User: ' + question + '>\n\n' + '<Assistant: ' + answer + '>\n\n'

    prompt_summary = get_context_summary_prompt(context=updated_context)

    try:
        compressed_context = llm.generate(prompt=prompt_summary, max_new_tokens=2048)
        
        return compressed_context if compressed_context else "No response from the model."
    except Exception as e:
        logger.error(f"An error occured in update_conversation_context: {str(e)}")
        return f"An error occurred: {str(e)}"