import rag_proof_of_concept as ragpoc
from langchain_openai import ChatOpenAI

import json

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

config = ragpoc.utils.load_config()

llm = ChatOpenAI(model=config['llm']['model'])
client = ragpoc.vdb.get_weaviate_client(local=config['vdb']['local'])
vector_store = ragpoc.vdb.get_vector_store(client, config['vdb']['collection'])

history = ""

user_input = "Who finances EBA?"

# How should I prepare for a job interview with Helveting?

preprocessed_query = ragpoc.agent.pre.preprocess_query(user_input, history, llm)

if type(preprocessed_query['related_concepts']) == list:
    context_request = ", ".join(preprocessed_query['related_concepts'])
else:
    context_request = preprocessed_query['related_concepts']

context_sample = ragpoc.context.get_context_sample(vector_store, context_request)

def get_query_assessment_prompt() -> ChatPromptTemplate:
    
    template = """You are an assistant for query-answering tasks. Below is a query for you, together with a context sample extracted from a vector store containing the user's documents. Your task is to pre-assess the query. Your output will be passed to json.loads, it should be pure JSON without any Python comments, chunk markers, or other non-JSON elements. Include keys "about_yourself" "context_relevant" "query_scope" and "focus" following the instructions below and using the Query and Context provided at the bottom.

    *about_yourself: "True" (the query asks about your behavior, capabilities or any other of your properties) or "False". Queries that challenge the quality of your previous answers are considered about yourself.

    *context_relevant: "True, justification" (the context sample seems relevant and helpful to answer the question) or "False" (you have to rely on your general knowledge to answer). Replace "justification" by an explanation of your rationale.

    *query_scope: "narrow" (the query is about a single very very very specific thing), "broad" (the query relates to a topic which contains several more granular items), "overview" (the query implies providing a summary, overview or counting relevant documents).

    *focus: "concepts" (the query meaning is not sensitive to the exact word choice, it could be easily rephrased with synonyms) or "keywords" (the query focuses on individual keywords that do not have widely accepted synonyms).
   
Query: {question}

Context: {context}"""
    return ChatPromptTemplate.from_template(template)

prompt_query_assessment =  get_query_assessment_prompt()

query_assessment_chain = (
    prompt_query_assessment
    | llm
    | StrOutputParser()
)

json.loads(
    query_assessment_chain.invoke({
        'question': preprocessed_query['implied_query'], 
        'context': ragpoc.context.format_context(context_sample)}))[0]
