import os
import tweepy
import json
import time
from config import Config
from tweepy.parsers import RawParser
import cPickle as pickle

keys = file('configu.cfg')
cfg = Config(keys)

consumer_key= cfg.consumer_key
consumer_secret= cfg.consumer_secret

access_token= cfg.access_token
access_token_secret= cfg.access_token_secret

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

def process_user(user):

    #check if data already collected
    user_directory = './unverified/'+user.screen_name+'/'
    if os.path.exists(user_directory) and os.listdir(user_directory):
      print 'User '+user.screen_name+'s info has already been collected.'
      return

    #create necessary directories
    if not os.path.exists(user_directory):
      os.makedirs(user_directory)

    #save user info
    file_name = './unverified/'+user.screen_name+'/'+user.screen_name+'_info.pickle'
    with open(file_name, 'w') as f:
      pickler = pickle.Pickler(f, -1)
      pickler.dump(user)

    #save tweets
    file_name = './unverified/'+user.screen_name+'/'+user.screen_name+'_tweets.pickle'
    with open(file_name, 'a') as f:
      pickler = pickle.Pickler(f, -1)
      for tweet in tweepy.Cursor(api.user_timeline,screen_name=user.screen_name).items(200):
        pickler.dump(tweet)

class StreamListener(tweepy.StreamListener):
    def on_status(self, tweet):
        if not tweet.user.verified:
          print str(tweet.user.screen_name)
          process_user(tweet.user)
    def on_connect(self):
      print "Connected"
    def on_disconnect(self, notice):
      """Called when twitter sends a disconnect notice
      Disconnect codes are listed here:
      https://dev.twitter.com/docs/streaming-apis/messages#Disconnect_messages_disconnect
      """
      print notice
      return
    def on_limit(self, track):
        """Called when a limitation notice arrvies"""
        print "limit notice arrived"
        return

    def on_error(self, status_code):
        """Called when a non-200 status code is returned"""
        print status_code
        return False

    def on_timeout(self):
        """Called when stream connection times out"""
        print "Time out"
        return

l = StreamListener()
streamer = tweepy.Stream(auth=auth, listener=l)
streamer.sample()
