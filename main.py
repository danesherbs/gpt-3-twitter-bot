import os
import random
import openai
import tweepy
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="main.log",
)

openai.api_key = os.getenv("OPENAI_API_KEY")

TWITTER_API_KEY = os.environ.get("TWITTER_API_KEY")
TWITTER_API_SECRET = os.environ.get("TWITTER_API_SECRET")
TWITTER_BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN")
TWITTER_ACCESS_TOKEN = os.environ.get("TWITTER_ACCESS_TOKEN")
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
    # "",
    "You can avoid a lot of drama by simply",
    "Frightened of change?",
    "You are not defined by",
]


def tweet(message):
    auth = tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_SECRET)
    auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET)
    api = tweepy.API(auth)
    api.update_status(message)


def tweet_random_message(dry_run=True):
    random_prompt = prompts[random.randint(0, len(prompts) - 1)]

    response = openai.Completion.create(
        engine="davinci",
        prompt=random_prompt,
        max_tokens=40,
        n=1,
        best_of=100,
        temperature=1,
        frequency_penalty=0.1,
    )

    random_completetion = response.choices[
        random.randint(0, len(response.choices) - 1)
    ]["text"]

    random_message = random_prompt + random_completetion
    shortened_random_message = get_first_two_sentences(random_message)

    if not dry_run:
        tweet(shortened_random_message)
    else:
        print(shortened_random_message)


def get_first_two_sentences(string):
    if len(string) == 0:
        return string

    string = string[0].upper() + string[1:]
    sentences = string.split(".")

    if len(sentences) < 2:
        return string

    return sentences[0] + "." + sentences[1] + "."


if __name__ == "__main__":
    tweet_random_message(dry_run=False)
