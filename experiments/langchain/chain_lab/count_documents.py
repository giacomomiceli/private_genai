import json
import rag_proof_of_concept as ragpoc
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.document import Document


from langchain_weaviate.vectorstores import WeaviateVectorStore
from weaviate.client import WeaviateClient

# Internal imports
from rag_proof_of_concept.utils import ChunkGranularity
from rag_proof_of_concept.vdb import FilterSpecs, get_weaviate_filter

from rag_proof_of_concept.context import get_document_summaries

config = ragpoc.utils.load_config()

llm = ChatOpenAI(model=config['llm']['model'])
client = ragpoc.vdb.get_weaviate_client(local=config['vdb']['local'])
knowledge_vs = ragpoc.vdb.get_vector_store(client, config['vdb']['collection'])
use_case_vs = ragpoc.vdb.get_vector_store(client, "RAG_use_cases")

documents = get_document_summaries(
    knowledge_vs = knowledge_vs, 
    query= "EBA", 
    n = None, 
    keyword_search_weight = 0.5)

[doc for doc in documents if doc[1] > 0.6]