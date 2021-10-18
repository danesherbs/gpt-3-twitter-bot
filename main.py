import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

prompts = [
    "The most important long-term investment you will make is",
    "A sign of success:",
    "You don't need to work hard, you need to",
    "Funny how people are",
    "As you get older",
    "Hey friend ",
]

response = openai.Completion.create(
    engine="davinci",
    prompt=prompts[-1],
    max_tokens=30,
    n=3,
    best_of=20,
    temperature=0.9,
)

print(response)
