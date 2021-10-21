import os
import random
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

prompts = [
    # "The most important long-term investment you will make is",
    # "A sign of success:",
    # "You don't need to work hard, you need to",
    # "Funny how people are",
    # "As you get older",
    # "Hey friend",
    # "What's the best feeling on Earth?",
    # "What's a wholesome saying?",
    # "The most important thing?",
    # "",
    "You can avoid a lot of drama by simply",
    "Frightened of change?",
    "You are not defined by",
]

random_prompt = prompts[random.randint(0, len(prompts) - 1)]

response = openai.Completion.create(
    engine="davinci",
    prompt=random_prompt,
    max_tokens=30,
    n=10,
    best_of=20,
    temperature=0.9,
    frequency_penalty=0.1,
)

print(random_prompt)
print(response)
