import os
import random
import openai
import tweepy


openai.api_key = os.getenv("OPENAI_API_KEY")

TWITTER_CONSUMER_KEY = os.environ.get("TWITTER_CONSUMER_KEY")
TWITTER_CONSUMER_SECRET = os.environ.get("TWITTER_CONSUMER_SECRET")
TWITTER_ACCESS_KEY = os.environ.get("TWITTER_ACCESS_KEY")
TWITTER_ACCESS_SECRET = os.environ.get("TWITTER_ACCESS_SECRET")

prompts = [
    "The most important long-term investment you will make is",
    "A sign of success:",
    "You don't need to work hard, you need to",
    "Funny how people are",
    "As you get older",
    "Hey friend",
    "What's the best feeling on Earth?",
    "What's a wholesome saying?",
    "The most important thing?",
    "",
    "You can avoid a lot of drama by simply",
    "Frightened of change?",
    "You are not defined by",
]


def tweet(message):
    auth = tweepy.OAuthHandler(TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET)
    auth.set_access_token(TWITTER_ACCESS_KEY, TWITTER_ACCESS_SECRET)
    api = tweepy.API(auth)
    api.update_status(message)


def tweet_random_message():
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

    random_message = response.choices[random.randint(0, len(response.choices) - 1)]

    tweet(random_message)
