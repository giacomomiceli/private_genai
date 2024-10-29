# External imports
import logging

from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI

from streamlit.delta_generator import DeltaGenerator

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------------------
# Answer a question about the agent itself

def get_self_answer_prompt(question: str, history: str, style: str) -> str:
    """
    Returns a prompt template the agent to answer a question about itself.

    Returns:
        PromptTemplate: The prompt template.

    """

    prompt = [
        {"role": "system", "content": f"""You are a business assistant whose primary function is to assist with information retrieval from the user's secure document store. Do not make any further claims about your function. Answer the question adopting a {style} writing style and keeping your answer concise. Your answer must focus on yourself, you may not follow-up or try to improve on previous answers. Do not apologize."""},

        {"role": "user", "content": f"""Question: {question}
Conversation history: {history}."""}]

    return prompt


def get_self_answer(
        query_text: str,
        history: str,
        style: str,
        llm: ChatOpenAI,
        msg_container: DeltaGenerator):
    
    prompt_self_answer = get_self_answer_prompt(question=query_text, history=history, style=style)

    msg_container.write("Preparing an answer (about myself)...")

    try:
        response = llm.generate(prompt=prompt_self_answer, max_new_tokens=256)
    
    except Exception as e:
        logger.error(f"An error occured in get_self_answer: {str(e)}")
        response = f"An error occurred: {str(e)}"
    
    return response, response