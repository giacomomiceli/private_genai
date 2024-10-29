import rag_proof_of_concept as ragpoc
import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

config = ragpoc.refs.load_config()

client = ragpoc.vdb.get_weaviate_client(local=config['vdb']['local'])

vector_store = ragpoc.vdb.get_vector_store(client, config['vdb']['collection'])

collection = client.collections.get(config['vdb']['collection'])

from rag_proof_of_concept.readers import ChunkGranularity
import weaviate.classes as wvc
from weaviate.collections.classes.filters import _FilterValue

raw_sample = collection.query.fetch_objects(
    limit = 100,
    filters = wvc.query.Filter.by_property("level").not_equal(ChunkGranularity.RAW.value)
)

import numpy as np

np.mean([len(x.properties['text']) for x in raw_sample.objects])
