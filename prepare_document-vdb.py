import rag_proof_of_concept as ragpoc

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

client = ragpoc.vdb.get_weaviate_client(local=config['vdb']['local'])

llm = ragpoc.utils.LLMClient(model=config['llm']['model'])

ragpoc.readers.create_collection(
    client, 
    config['vdb']['collection'],
    config['data']['path'],
    config['vdb']['chunk_size'],
    config['vdb']['sentences_per_summary'],
    llm, 
    include_chunk_summaries = config['vdb']['include_chunk_summaries'],
    batch_size=config['vdb']['summarization_batch_size'])

# Check the average size of the chunks
ragpoc.context.get_context_element_sizes(client, config['vdb']['collection'])
