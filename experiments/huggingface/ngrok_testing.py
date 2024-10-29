import requests

text = "Task: Describe a nice day using five words. Answer: "

messages = [
    {"role": "system", "content": "Answer in json format."},
    {"role": "user", "content": "Categorize the tone and topic of the following sentence using the keys 'tone' and 'topic': Today we had an awesome breakfast at home."},
]

url = "https://4f36-34-125-69-212.ngrok-free.app/"

url_model_name = url + "api/model_name"
url_gpu_name = url + "api/gpu_name"
url_generate = url + "api/generate"
url_measure_prompt = url + "api/measure_prompt"

response_model_name = requests.get(url_model_name)
print(response_model_name.text)

response_gpu_name = requests.get(url_gpu_name)
print(response_gpu_name.text)

response_text = requests.post(url_generate, json={"prompt": text, "max_new_tokens": 64})
print(response_text.text)

response_messages = requests.post(url_generate, json={"prompt": messages, "max_new_tokens": 64})
print(response_messages.text)

response_measure_text = requests.post(url_measure_prompt, json={"prompt": text})
print(response_measure_text.json())

response_measure_messages = requests.post(url_measure_prompt, json={"prompt": messages})
print(response_measure_messages.json())
