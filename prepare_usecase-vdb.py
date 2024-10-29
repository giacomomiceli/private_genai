import rag_proof_of_concept as ragpoc
import logging
from langchain_openai import ChatOpenAI
import tiktoken
import os
import yaml
from langchain.schema.document import Document

from rag_proof_of_concept.vdb import load_to_vector_store, clear_vector_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = ragpoc.utils.load_config()

client = ragpoc.vdb.get_weaviate_client(local=config['vdb']['local'])

llm = ChatOpenAI(model=config['llm']['model'])
encoding = tiktoken.encoding_for_model(llm.model_name)

use_cases_folder = 'use_cases'
use_cases_content = []

for filename in os.listdir(use_cases_folder):
    if filename.endswith('.yaml'):
        with open(os.path.join(use_cases_folder, filename), 'r') as file:
            use_case = yaml.safe_load(file)
            use_cases_content.append(use_case)

use_cases = [Document(page_content=avatar, metadata = {'use_case': use_case['label'], 'use_case_description': use_case['description']}) for use_case in use_cases_content for avatar in use_case['avatars'] ]


clear_vector_store(client, "RAG_use_cases")

load_to_vector_store(
    use_cases,
    client,
    "RAG_use_cases")

use_case_vs = ragpoc.vdb.get_vector_store(client, "RAG_use_cases")

use_case_vs.similarity_search_with_score(query = "overview", alpha = 0)
