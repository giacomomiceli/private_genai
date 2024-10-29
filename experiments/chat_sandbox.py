import rag_proof_of_concept as ragpoc

import streamlit as st

import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)
existing_loggers = list(logging.root.manager.loggerDict.keys())
# Set logging level to INFO for all loggers under rag_proof_of_concept
for logger_name in existing_loggers:
    if logger_name.startswith('rag_proof_of_concept'):
        logging.getLogger(logger_name).setLevel(logging.INFO)

config = ragpoc.utils.load_config()

llm = ragpoc.utils.LLMClient(model=config['llm']['model'])

client = ragpoc.vdb.get_weaviate_client(local=config['vdb']['local'])
knowledge_vs = ragpoc.vdb.get_vector_store(client, config['vdb']['collection'])
knowledge_connection = client.collections.get(config['vdb']['collection'])
use_case_vs = ragpoc.vdb.get_vector_store(client, "RAG_use_cases")

msg_container = st.empty()

history = ""

expected_chunk_size = ragpoc.context.get_context_element_sizes(knowledge_connection)


#user_input = "How many documents mention the EU?"
#user_input = "What is in my documents?"
#user_input = "Why is your answer so hard to read?"
user_input = "How should a bank prepare for CCAR?"

query_assessment = ragpoc.agent.pre.preprocess_query(
    query_text=user_input, 
    history=history,
    llm=llm,
    use_case_vs=use_case_vs
)

print(query_assessment)

# match query_assessment['query_type']:
#     case 'count_documents':
raw_answer, formatted_answer = ragpoc.agent.use_cases.count_documents.get_count_documents_answer(
    query_text=query_assessment['implied_query'],
    context_query=query_assessment['related_concepts'],
    knowledge_vs=knowledge_vs,
    style=config['chatbot']['conversation_style'],
    llm=llm,
    msg_container=msg_container)

#     case 'collection_overview':
raw_answer, formatted_answer = ragpoc.agent.use_cases.collection_overview.get_collection_overview_answer(
    query_text=query_assessment['implied_query'],
    knowledge_connection=knowledge_connection,
    style=config['chatbot']['conversation_style'],
    llm=llm,
    msg_container=msg_container)

#     case 'self':
raw_answer, formatted_answer = ragpoc.agent.use_cases.self.get_self_answer(
    query_text=query_assessment['implied_query'],
    history=history,
    style=config['chatbot']['conversation_style'],
    llm=llm,
    msg_container=msg_container)

#     case _:
context_sample = ragpoc.context.get_context_sample(
    knowledge_vs=knowledge_vs, 
    query=query_assessment['related_concepts'])

formatted_context_sample, _ = ragpoc.context.format_context(context_sample, include_granularity=True)

assessment_outcome = ragpoc.agent.use_cases.generic.assess_query(
    query_text=query_assessment['implied_query'],
    context = formatted_context_sample,
    llm = llm)


# --------------------------------------------------------------------------------------------------

    
prompt_query_assessment =  ragpoc.agent.use_cases.generic.get_query_assessment_prompt(
    question=query_assessment['implied_query'],
    context=formatted_context_sample)

import re, json
response = llm.generate(prompt=prompt_query_assessment, max_new_tokens=512)

input = re.sub(r'^.*?\[', '[', response)
    # Remove everything to the right of the last ']'
input = re.sub(r'\].*$', ']', input)
# Remove any remaining markdown code block delimiters
input = re.sub(r"```\s*json\s*|```\s*", "", input)
json.loads(input)


# --------------------------------------------------------------------------------------------------



print(assessment_outcome)

# assessment_outcome['context_relevant'] is False
raw_answer, formatted_answer = ragpoc.agent.use_cases.generic.get_simple_answer(
    query_text=query_assessment['implied_query'],
    history=history,
    style=config['chatbot']['conversation_style'],
    llm=llm)

# assessment_outcome['context_relevant'] is True
raw_answer, formatted_answer = ragpoc.agent.use_cases.generic.get_rag_answer(
    query_text=query_assessment['implied_query'], 
    context_query=query_assessment['related_concepts'],
    tokens_target=config['llm']['input_tokens_target'],
    expected_chunk_size=expected_chunk_size,
    knowledge_vs=knowledge_vs,
    style=config['chatbot']['conversation_style'],
    llm=llm)
    #keyword_search_weight = 0,
    #context_granularity_allocation = {ChunkGranularity.RAW: 1})

MAX_NEW_TOKENS = 1024
ragpoc.agent.use_cases.generic.get_context(
    prompt=ragpoc.agent.use_cases.generic.get_rag_answer_prompt(question=query_assessment['implied_query'], style=config['chatbot']['conversation_style'], max_new_tokens=round(1.25*MAX_NEW_TOKENS)),
    context_query=query_assessment['related_concepts'],
    tokens_target=config['llm']['input_tokens_target'], 
    expected_chunk_size=expected_chunk_size,  
    knowledge_vs=knowledge_vs, 
    llm=llm) 
    #keyword_search_weight=0,
    #granularity_allocation={ChunkGranularity.RAW: 1})



# Run in one go
raw_answer, formatted_answer = ragpoc.agent.use_cases.generic.get_generic_answer(
    query_text=query_assessment['implied_query'],
    context_query=query_assessment['related_concepts'],
    tokens_target=config['llm']['input_tokens_target'],
    expected_chunk_size=expected_chunk_size,
    knowledge_vs=knowledge_vs,
    history=history,
    style=config['chatbot']['conversation_style'],
    llm=llm,
    msg_container=msg_container)
