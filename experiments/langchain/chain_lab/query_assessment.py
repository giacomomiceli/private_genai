import json
import rag_proof_of_concept as ragpoc
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

config = ragpoc.utils.load_config()

llm = ChatOpenAI(model=config['llm']['model'])
client = ragpoc.vdb.get_weaviate_client(local=config['vdb']['local'])
knowledge_vs = ragpoc.vdb.get_vector_store(client, config['vdb']['collection'])
use_case_vs = ragpoc.vdb.get_vector_store(client, "RAG_use_cases")

from rag_proof_of_concept.context import get_context_sample
from rag_proof_of_concept.agent.pre import preprocess_query

def get_query_assessment_prompt() -> PromptTemplate:
    template = """You are an assistant for query-answering tasks. Below is a query for you, together with a context sample extracted from a vector store containing the user's documents. Your task is to pre-assess the query. Your output will be passed to json.loads, it should be pure JSON without any Python comments, chunk markers, or other non-JSON elements. Include keys "context_relevant" "granularity" and "focus" following the instructions below and using the Query and Context provided at the bottom.

    *context_relevant: "True" (the context sample is related and helpful to answer the question) or "False" (you have to rely on your general knowledge to answer).

    *granularity: The context granularity which is most helpful.

    *focus: "concepts" (the query meaning is not sensitive to the exact word choice, it could be easily rephrased with synonyms) or "keywords" (the query focuses on individual keywords that do not have widely accepted synonyms).
   
Query: {question}

Context: {context}"""
    return PromptTemplate.from_template(template)

prompt_query_assessment =  get_query_assessment_prompt()

query_assessment_chain = (
    prompt_query_assessment
    | llm
    | StrOutputParser()
)

history = ""

query_text = "what does EBA require regarding adjustments to historical NTI?"

pre = preprocess_query(
    query_text=query_text, 
    history=history, 
    llm=llm, 
    use_case_vs=use_case_vs)

context_sample = get_context_sample(
    knowledge_vs=knowledge_vs, 
    query=pre['related_concepts'])

json.loads(query_assessment_chain.invoke({'question': pre['implied_query'], 'context': context_sample}))



json.loads(query_assessment_chain.invoke({'question': "How are xVA adjustment regulated for tier 3 banks?", 'context': context_sample}))

json.loads(query_assessment_chain.invoke({'question': "Give me an summary of banking regulations.", 'context': context_sample}))

json.loads(query_assessment_chain.invoke({'question': "Why did you mention only 2 documents?", 'context': context_sample}))