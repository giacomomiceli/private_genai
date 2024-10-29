import rag_proof_of_concept as ragpoc

from langchain_openai import ChatOpenAI

config = ragpoc.utils.load_config()

llm = ChatOpenAI(model=config['llm']['model'])
client = ragpoc.vdb.get_weaviate_client(local=config['vdb']['local'])

knowledge_connection = client.collections.get(config['vdb']['collection'])

knowledge_connection.query.fetch_objects()