import rag_proof_of_concept as ragpoc
messages = [
    {"role": "system", "content": "Answer in json format."},
    {"role": "user", "content": "Categorize the tone and topic of the following sentence using the keys 'tone' and 'topic': Today we had an awesome breakfast at home."},
]

with open("testfile.txt", "a") as f:
    f.write(str(messages) + "\n-------------\n")

#llm = ragpoc.utils.LLMClient('gpt-4o-mini')
llm = ragpoc.utils.LLMClient('https://0f9bd1f8f88e8921c8.gradio.live/')
llm = ragpoc.utils.LLMClient('https://5c7a-34-125-69-212.ngrok-free.app/')

print(llm.generate(messages))

print(llm._encode(messages))

print(llm.measure_prompt(messages))

print(llm.estimate_prompt(messages))

print(llm.validate_prompt_size(messages))

print(llm.token_count_to_str_length(59))