import rag_proof_of_concept as ragpoc
from openai import OpenAI

client = OpenAI()

messages = [
    {"role": "system", "content": "Answer in json format."},
    {"role": "user", "content": "Categorize the tone and topic of the following sentence using the keys 'tone' and 'topic': Today we had an awesome breakfast at home."},
]

messages_simple_answer = [
    {"role": "system", "content": """Answer the question below adopting a concise professional style. Make sure it continues the conversation flow based on the provided conversation history. Try your best to give an answer but disclaim prominently that it is based on your general knowledge.
Use Markdown to format your answer."""},
    
  {"role": "user", "content": """
  Query: Are there any equivalent stress tests by other regulators?
  Conversation history: <User: Tell me about EBA stress testing.> <Assistant: The EBA stress test is the yearly stress test by the European Banking Authority>
  """},
]

messages_rag_answer = [
    {"role": "system", "content": """Use the following pieces of retrieved context ordered by decreasing relevance to answer the question below.
For each point in your answer you must write the corresponding {HEX_ID_LENGTH} hexadecimal digits of the context uuid in braces.
Use Markdown to format your answer."""},
    
  {"role": "user", "content": """Query: What are the most important regulations in EBA stress testing?
Context: {{'uuid': A1B0, 'content': 'In EBA stress testing, Banks must ensure scenario consistence of all of their projections'.},
{'uuid': 5F13, 'content': 'The EBA has the power to initiate and coordinate EU-wide stress tests in cooperation with the European Systemic Risk Board (ESRB), the European Central Bank (ECB), and the European Commission (EC)'}}"""}]


response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=messages_rag_answer,
    max_tokens=512,
    n=1,
    stop=None
)


llm_cloud = ragpoc.utils.LLMClient('gpt-4o-mini')
print('Model: ' + llm_cloud.model)
print('GPU: ' + llm_cloud.gpu)
print('\n' + llm_cloud.generate(messages_rag_answer))

llm_local = ragpoc.utils.LLMClient('https://18e3-34-19-34-236.ngrok-free.app/')
print('Model: ' + llm_local.model)
print('GPU: ' + llm_local.gpu)
print('\n' + llm_local.generate(messages_rag_answer))


