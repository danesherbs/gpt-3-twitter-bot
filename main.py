import os
import random
import tweepy
import models


TWITTER_API_KEY = os.environ.get("TWITTER_API_KEY")
TWITTER_API_SECRET = os.environ.get("TWITTER_API_SECRET")
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
    "Nothing delights the mind as much as",
    "You can avoid a lot of drama by simply",
    "Frightened of change?",
    "You are not defined by",
    "You will earn the respect of everyone if you begin by",
    "It is not because things are difficult that we do not dare; it is because we",
    "Devote the rest of your life to",
    "Value your time more than your",
    "Wealth consists not in having great possesions, but in",
]


def tweet(message):
    auth = tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_SECRET)
    auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET)
    api = tweepy.API(auth)
    api.update_status(message)


def tweet_random_message(dry_run=True):
    random_prompt = prompts[random.randint(0, len(prompts) - 1)]
    [random_completion] = models.my_gpt.generate(random_prompt, max_length=220)

    if not dry_run:
        tweet(random_completion)

    print(random_completion)


if __name__ == "__main__":
    tweet_random_message(dry_run=False)
