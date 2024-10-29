import rag_proof_of_concept as ragpoc
from langchain_openai import ChatOpenAI

import tiktoken

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from rag_proof_of_concept.agent.refs import context_prep


import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = ragpoc.refs.load_config()

llm = ChatOpenAI(model=config['llm']['model'])
encoding = tiktoken.encoding_for_model(llm.model_name)

client = ragpoc.vdb.get_weaviate_client(local=config['vdb']['local'])
vector_store = ragpoc.vdb.get_vector_store(client, config['vdb']['collection'])

history = ""

query_text = "Who finances EBA?"

preprocessed_query = ragpoc.refs.preprocess_query(query_text, history, llm)

context_sample = ragpoc.refs.get_context_sample(vector_store, preprocessed_query['related_concepts'])

assessment_outcome = ragpoc.refs.assess_query(
    preprocessed_query['implied_query'], 
    ragpoc.refs.context_prep(context_sample), 
    llm)

####

context = (
    ragpoc.refs.get_chunks(
        vector_store, 
        preprocessed_query['related_concepts'],
        200)
    )

def get_rag_answer_prompt() -> ChatPromptTemplate:
    template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context ordered by decreasing relevance to answer the question, and any additional context you may have. Make the user aware of any contradictions or erroneous knowledge implied by the question. For each point in your answer you must cite the corresponding context uuid in square brackets (print only groups of hexadecimal numbers in square brackets). You may only cite from the context, ensuring that the content of the citation actually relates to the point made. Spread references evenly across your answer, avoid citing two uuids in a row and do not print place references in bulk at the end. Your answer must be understandable to an uninformed reader. Use a {style} writing style, no matter what the style of the question and context is. Do not repeat the question.
     
Parts of the context may be irrelevant, so do not feel forced to use and cite the irrelevant parts. Never mention "the context" or "the sources provided" because the user doesn't provide them, they are automatically retrieved without the user's intervention. 

Use Markdown to format your answer but do not add a title. You may use bullet points but only if they add clarity to the answer.

If the question includes instructions about the format of the answer, those instructions take precedence over any other formatting guidance.
Question: {question}
Context: {context}"""

    return ChatPromptTemplate.from_template(template)

prompt_answer = get_rag_answer_prompt()

fit_calibration_chain = (
    prompt_answer
    | (lambda x: ragpoc.refs.measure_prompt_tokens(x, encoding, return_prompt_len=True))
)

prepped_context = context_prep(context)

fit_data = [fit_calibration_chain.invoke({
    'question': preprocessed_query['implied_query'], 
    'context': prepped_context[0:k],
    'style': config['chatbot']['conversation_style']})
 for k in range(1,200)]


import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

sns.lineplot(x=[a[1] for a in fit_data], y=[a[0] for a in fit_data])

plt.show()

# Parameters for ragpoc.utils.measure_prompt_tokens
np.polyfit([a[1] for a in fit_data], [a[0] for a in fit_data], 1)

# Parameters for ragpoc.utils.estimate_str_len
np.polyfit([a[0] for a in fit_data], [a[1] for a in fit_data], 1)

measure_chain = (
    prompt_answer
    | (lambda x: ragpoc.refs.measure_prompt_tokens(x, encoding))
)

estimate_chain = (
    prompt_answer
    | (lambda x: ragpoc.refs.estimate_prompt_tokens(x, llm))
)

measure_chain.invoke({
    'question': preprocessed_query['implied_query'], 
    'context': prepped_context[0:200],
    'style': config['chatbot']['conversation_style']})

estimate_chain.invoke({
    'question': preprocessed_query['implied_query'], 
    'context': prepped_context[0:200],
    'style': config['chatbot']['conversation_style']})


prepped_context = context_prep(context)

n_chunks = ragpoc.refs.estimate_chunk_count(config['llm']['input_tokens_target'], prepped_context, llm)

np.mean([len(str(x)) for x in prepped_context])


context_length = measure_chain.invoke({
    'question': preprocessed_query['implied_query'], 
    'context': prepped_context[0:n_chunks],
    'style': config['chatbot']['conversation_style']})

n_chunks_final = round(config['llm']['input_tokens_target'] / context_length * n_chunks)

measure_chain.invoke({
    'question': preprocessed_query['implied_query'], 
    'context': prepped_context[0:n_chunks_final],
    'style': config['chatbot']['conversation_style']})


### Try the final function

narrow_raw, narrow_formatted = ragpoc.refs.get_narrow_rag_answer(
    preprocessed_query['implied_query'], 
    preprocessed_query['related_concepts'], 
    config['llm']['input_tokens_target'], 
    config['vdb']['chunk_size'],
    vector_store,
    config['chatbot']['conversation_style'],
    llm,
    encoding,
    alpha = 0.5)

broad_raw, broad_formatted = ragpoc.refs.get_broad_rag_answer(
    preprocessed_query['implied_query'], 
    preprocessed_query['related_concepts'], 
    config['llm']['input_tokens_target'], 
    config['vdb']['chunk_size'],
    vector_store,
    config['chatbot']['conversation_style'],
    llm,
    encoding,
    alpha = 0.5)

overview_raw, overview_formatted = ragpoc.refs.get_overview_rag_answer(
    preprocessed_query['implied_query'], 
    preprocessed_query['related_concepts'], 
    config['llm']['input_tokens_target'], 
    config['vdb']['chunk_size'],
    vector_store,
    config['chatbot']['conversation_style'],
    llm,
    encoding,
    alpha = 0.5)
