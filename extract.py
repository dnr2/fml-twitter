#extract features from data
import os
import pickletools
import cPickle as pickle
import pprint
import datetime

def extract_tweets_features(filepath, classification):

  file = open(filepath, 'rb')
  unpickler = pickle.Unpickler(file)

  hashtag_count = 0
  symbols_count = 0
  urls_count = 0
  user_mentions_count = 0
  favorite_count = 0
  retweet_count = 0
  truncated_count = 0
  hashtag_avg = 0
  symbols_avg = 0
  urls_avg = 0
  user_mentions_avg = 0
  favorite_avg = 0
  retweet_avg = 0
  
  for tweet_count in range(1, 200):
    try:
      tweet = unpickler.load()
      # num hashtags
      hashtag_count += len(tweet.entities['hashtags'])
      # num symbols
      symbols_count += len(tweet.entities['symbols'])
      # num urls
      urls_count += len(tweet.entities['urls'])
      # num user_mentions
      user_mentions_count += len(tweet.entities['user_mentions'])
      # favorite_count
      favorite_count += tweet.favorite_count
      # retweet_count
      retweet_count += tweet.retweet_count
    except EOFError:
      pprint.pprint(tweet_count)
      break

  hashtag_avg = hashtag_count / float(tweet_count)
  symbols_avg = symbols_avg / float(tweet_count)
  urls_avg = urls_avg / float(tweet_count)
  user_mentions_avg = user_mentions_avg / float(tweet_count)
  favorite_avg =  favorite_count / float(tweet_count)
  retweet_avg = retweet_count / float(tweet_count)

  resultString =( str(hashtag_count) + "," + str(hashtag_avg) + ","
                + str(symbols_count) + "," + str(symbols_avg) + ","
                + str(urls_count) + "," + str(urls_avg) + ","
                + str(user_mentions_count) + "," + str(user_mentions_avg) + ","
                + str(favorite_count) + ","+ str(favorite_avg) + ","
                + str(retweet_count) + "," + str(retweet_avg))

  return resultString


def extract_info_features(filepath, classification):
  if classification == "verified":
    resultString = "1,"
  else:
    resultString = "0,"

  with open(filepath, 'rb') as file:
    info = pickle.load(file)
    #followers_count
    resultString += str(info.followers_count) + ","
    #friends_count
    resultString += str(info.friends_count) + ","
    #listed_count
    resultString += str(info.listed_count) + ","
    #statuses_count
    resultString += str(info.statuses_count) + ","
    
    #contributors_enabled
    resultString += str( "1" if info.contributors_enabled else "0" ) + ","
    #created_at - number of days from 1 Nov 2005, before creation of twitter
    resultString += str( (info.created_at - datetime.datetime.strptime("1 Nov 05", "%d %b %y")).days ) + ","
    #geo_enabled
    resultString += str( "1" if info.geo_enabled else "0" ) + ","
    
    return resultString

if __name__ == "__main__":
  #create csv file
  with open("data.csv", 'w') as newFile:
    for directory in os.listdir("verified/"):
      featureString = ""
      infoFeatures = ""
      tweetsFeatures = ""
      if directory == ".DS_Store":
        continue
      for file in os.listdir("verified/"+directory+"/"):
        fileName, fileExtension = os.path.splitext(file)
        if fileExtension == ".pickle":
          if fileName.endswith("_info"):
            infoFeatures = extract_info_features("verified/"+directory+"/"+file, "verified").rstrip()
          if fileName.endswith("_tweets"):
            tweetsFeatures = extract_tweets_features("verified/"+directory+"/"+file, "verified").rstrip()
      featureString = infoFeatures + tweetsFeatures
      newFile.write(featureString+"\n")
    for directory in os.listdir("unverified/"):
      featureString = ""
      infoFeatures = ""
      tweetsFeatures = ""
      if directory == ".DS_Store":
        continue
      for file in os.listdir("unverified/"+directory+"/"):
        fileName, fileExtension = os.path.splitext(file)
        if fileExtension == ".pickle":
          if fileName.endswith("_info"):
            infoFeatures = extract_info_features("unverified/"+directory+"/"+file, "unverified").rstrip()
          if fileName.endswith("_tweets"):
            tweetsFeatures = extract_tweets_features("unverified/"+directory+"/"+file, "unverified").rstrip()
      featureString = infoFeatures + tweetsFeatures
      newFile.write(featureString+"\n")
