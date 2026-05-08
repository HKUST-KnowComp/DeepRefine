from openai import OpenAI
client = OpenAI(base_url="http://0.0.0.0:8129/v1", api_key="")
response = client.chat.completions.create(
    model="Qwen/Qwen3-30B-A3B-Instruct-2507",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ],
    temperature=0.0,
)
print(response.choices[0].message.content)
