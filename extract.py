#extract features from data
import os
import pickletools
import cPickle as pickle
import pprint
import datetime
import time

from sklearn.feature_extraction.text import TfidfTransformer

"""
NLP FEATURES: 

features from the user:
- description
- lang (user's self-declared user interface language)
- screen_name
- name

features from the tweets
- lang (machine-detected language of the Tweet text)
- text ( The actual UTF-8 text of the status update ) 
"""
use_nlp = True #add user description as a column in the data file


#remove undesired characters
def clean_string( my_str ) :
  ret = str( my_str.encode('ascii', 'ignore') )
  ret = ret.replace(",","").replace('\n','').replace("\r","").replace("\t","").replace("\"","")
  ret = "\"" + ret + "\""
  return ret
  
def extract_tweets_features(filepath, classification):
  file = open(filepath, 'rb')
  try:
    unpickler = pickle.Unpickler(file)
  except EOFError:
    return ""
  
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
    except Exception, e:
      pprint.pprint("bug twitter" + filepath)
      return ""
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
    try:
      info = pickle.load(file)     
    except Exception, e:
      pprint.pprint("bug in" + filepath)
      return ""
      
    #create features from user description
    if use_nlp :
      if info.lang[:2] == "en" : # only add user with descriptions in English 
        description = clean_string( info.description if (info.description is not None) else "" )
        screen_name = clean_string( info.screen_name if (info.screen_name is not None) else ""  )
        user_name = clean_string( info.name if (info.name is not None) else "" )
        
        resultString += description + ","        
        resultString += screen_name + ","
        resultString += user_name + ","
        
      else :
        return ""
    
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
  filename = "data.csv"
  # if use_nlp :
  filename = "data_nlp.csv"
  
  with open(filename, 'w') as newFile:
    
    #TODO REMOVE!
    cont = 0
    
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
      
      #case the user data is incomplete, skip this user
      if infoFeatures == "" or tweetsFeatures == "" :
        continue
      pprint.pprint(directory)
      featureString = infoFeatures + tweetsFeatures
      newFile.write(featureString+"\n")
      
      #TODO REMOVE!
      cont = cont + 1
      if cont > 2000 :
        break
    
    #TODO REMOVE!
    cont = 0
    
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
      
      #case the user data is incomplete, skip this user
      if infoFeatures == "" or tweetsFeatures == "" :
        continue 
      featureString = infoFeatures + tweetsFeatures
      pprint.pprint(directory)
      newFile.write(featureString+"\n")
      
      #TODO REMOVE!
      cont = cont + 1
      if cont > 2000 :
        break