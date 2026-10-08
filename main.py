from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-6-astra",
    input="Explain artificial intelligence in one short sentence."
)

print(response.output_text)