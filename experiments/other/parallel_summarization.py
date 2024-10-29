import rag_proof_of_concept as ragpoc
import re

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

pages = ragpoc.readers.load_directory(config['data']['path'])

chunks = ragpoc.readers.split_documents(pages, config['vdb']['chunk_size'])

# --------------------------------------------------------------------------------------------------
group_size = 5
n_sentences = 5

group = pages[0:group_size]

import json
chunk_batch = json.dumps(
    [{'id': id, 'content': re.sub('\n+', ' ', chunk.page_content)} for id, chunk in zip(range(len(group)), group)],
    ensure_ascii=False)

import inflect
inflect_engine = inflect.engine()

chunk_summary_prompt = ragpoc.readers.get_chunk_batch_summary_prompt(
    text = chunk_batch, 
    inflect_engine = inflect_engine,
    n_sentences = n_sentences)

response = llm.generate(prompt = chunk_summary_prompt, max_new_tokens = 128*group_size*n_sentences)

out = ragpoc.utils.read_json(response)

# --------------------------------------------------------------------------------------------------

# YAML variant
group_size = 4
n_sentences = 5

group = pages[0:group_size]

import yaml

re.sub('\n+', ' ', group[0].page_content)


chunk_batch = yaml.safe_dump(
    [{'item': re.sub('\n+', ' ', chunk.page_content)} for chunk in group],
    allow_unicode=True,
    default_flow_style=False,
    width=2**16)

import inflect
inflect_engine = inflect.engine()

chunk_summary_prompt = ragpoc.readers.get_chunk_batch_summary_prompt(
    text = chunk_batch,
    n_items = group_size,
    inflect_engine = inflect_engine,
    n_sentences = n_sentences)

llm.measure_prompt(chunk_summary_prompt)
llm.validate_prompt_size(chunk_summary_prompt)

response = llm.generate(prompt = chunk_summary_prompt, max_new_tokens = 128*group_size*n_sentences)


#modified_response = re.sub(r'-.*?:', '- summary:', response)

out = yaml.safe_load(re.sub(r"```\s*yaml\s*|```\s*", "", response))

# --------------------------------------------------------------------------------------------------

page_summaries1 = ragpoc.readers.summarize_pages(pages[0:32], config['vdb']['sentences_per_summary'], llm, batch_size=1)
page_summaries2 = ragpoc.readers.summarize_pages(pages[0:32], config['vdb']['sentences_per_summary'], llm, batch_size=4)

document_summaries1 = ragpoc.readers.summarize_documents(pages, config['vdb']['sentences_per_summary'], llm)
#document_summaries2 = ragpoc.readers.summarize_documents(page_summaries2, config['vdb']['sentences_per_summary'], llm)

page_summaries1[0].page_content
page_summaries2[0].page_content

import numpy as np

(65/47)/(np.sum([len(x.page_content) for x in page_summaries1])/np.sum([len(x.page_content) for x in page_summaries2]))

