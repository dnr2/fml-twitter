#Convert Pickle Object of User and Tweet to Data for scikit-learn

import json
import pickle
import pprint
import os
import dumper

def convert_verified_pickle_to_json(directory):
  if not directory == ".DS_Store":
    info_file_name = "./verified/"+directory+"/"+directory+"_info.pickle"
    tweets_file_name = "./verified/"+directory+"/"+directory+"_tweets.pickle"

    #user file processing
    with open(info_file_name, 'r') as info_file:
      info = pickle.load(info_file)
      with open("./verified/"+directory+"/"+directory+"_info.json", 'w') as info_file_json:
        d = json.dumps(info._json)
        info_file_json.write(d)

    #tweets file processing
    with open(tweets_file_name, 'r') as tweets_file:
      tweets = pickle.load(tweets_file)
      with open("./verified/"+directory+"/"+directory+"_tweets.json", 'w') as tweets_file_json:
        d = json.dumps(tweets._json)
        tweets_file_json.write(d)

def convert_unverified_pickle_to_json(directory):
  if not directory == ".DS_Store":
    info_file_name = "./unverified/"+directory+"/"+directory+"_info.pickle"
    tweets_file_name = "./unverified/"+directory+"/"+directory+"_tweets.pickle"

    #user file processing
    with open(info_file_name, 'r') as info_file:
      info = pickle.load(info_file)
      with open("./unverified/"+directory+"/"+directory+"_info.json", 'w') as info_file_json:
        d = json.dumps(info._json)
        info_file_json.write(d)

    #tweets file processing
    with open(tweets_file_name, 'r') as tweets_file:
        tweets = pickle.load(tweets_file)
        with open("./unverified/"+directory+"/"+directory+"_tweets.json", 'w') as tweets_file_json:
          d = json.dumps(tweets._json)
          tweets_file_json.write(d)

if __name__ == "__main__":
  for directory in os.listdir('verified/'):
    print directory
    convert_verified_pickle_to_json(directory)
  for directory in os.listdir('unverified/'):
    print directory
    convert_unverified_pickle_to_json(directory)
