#gets COUNT tweets from user's timeline

import os
import tweepy
import cPickle as pickle
from config import Config

#constants
COUNT = 200

#tweepy configuration
keys = file('config.cfg')
cfg = Config(keys)

consumer_key= cfg.consumer_key
consumer_secret= cfg.consumer_secret

access_token= cfg.access_token
access_token_secret= cfg.access_token_secret

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

def get_tweets(username, isVerified):
  if isVerified:
    file_name = './verified/'+username+'/'+username+'_tweets.pickle'
  else:
    file_name = './unverified/'+username+'/'+username+'_tweets.pickle'

  #save tweets
  with open(file_name, 'wb') as f:
    pickler = pickle.Pickler(f, -1)
    tweet_count = 0
    for tweet in tweepy.Cursor(api.user_timeline,screen_name=username).items(200):
      pickler.dump(tweet)
      tweet_count = tweet_count +1
      print tweet_count

if __name__ == "__main__":
  for directory in os.listdir("verified/"):
    if directory == ".DS_Store":
      continue
    print directory
    get_tweets(directory, True)
  for directory in os.listdir("unverified/"):
    print directory
    get_tweets(directory, False)
