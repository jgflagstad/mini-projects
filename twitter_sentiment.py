import tweepy
from textblob import TextBlob

consumer_key = 'f1XdrxOpRugrAPxGwjEjOOgrS'
consumer_secret = 'ohnRQxag2kmlBvoereRUVgH3eXldk8iQYxF7McGePsZUs3g5De'
access_token = '2581848348-uVFCWTGBqWqHZqIawWSk4DtoSmuOqzib1XYCBhb'
access_token_secret = '8AqDDMPPUHiq6sCq5UoP0kiSG9m8vpiz4GRLnSDW7thiZ'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

public_tweets = api.search('Trump')

for tweet in public_tweets:
	print(tweet.text)
	analysis = TextBlob(tweet.text)
	print(analysis.sentiment)