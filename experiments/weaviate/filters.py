import rag_proof_of_concept as ragpoc
import logging

logging.basicConfig(level=logging.INFO)


from rag_proof_of_concept.vdb import FilterSpecs
from rag_proof_of_concept.readers import ChunkGranularity

filter_test = FilterSpecs({"granularity": [ChunkGranularity.PAGE_SUMMARY.value, True], "test": [1, True]})