import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import re
from datetime import datetime
import os
import pickle
import numpy as np
import os
import pandas as pd
import json
from utils_collection_urls import *
from utils_collection_tweets import *
import re

db_file_name = "Twitter_accounts.db"
keywords_list_path = "tweets_keywords.csv"
ij_keywords_list_path = "ijsdd_keywords.csv"
year_input = list(range(2007, 2023))
login_accounts(db_file_name)
main_folder_twitter='all_tweets'
tweets_alread_saved='tweets_files_already_saved_locally.csv'
accounts_credentials_file_name="account_credentials.txt"
accounts_credentials_file_format="username:password:email:email_password"


df_cols_names=["original_tweet_id", "user_id", "tweet_text", "user_username", "tweet_lang",
                                         "tweet_coordinates", "tweet_date", "user_location", "tweet_place",
                                         "tweet_hastags", "tweet_inreplytotweetid", "tweet_retweetedTweet",
                                         "tweet_quotedTweet",'file_start_date','file_end_date']

# files_already_collected_aux1=list(set([file.split('Twitter_')[0][:-1] for file in pd.read_csv(tweets_alread_saved).iloc[:,-1].tolist() if '/Twitter/' in  file]))
files_already_collected_aux2=list(set([file.split('Twitter_')[0].split('AUX')[0] for file in collect_filenames_in_directory(main_folder_twitter)if '/Twitter/' in file]))
files_already_collected=files_already_collected_aux2 #files_already_collected_aux1+

ij_keywords_aux = pd.read_csv(ij_keywords_list_path)
ij_keywords = (ij_keywords_aux[ij_keywords_aux.apply(pd.notnull)].reset_index(drop=True).iloc[:, -1].tolist())

non_ijkeywords_aux=pd.read_csv(keywords_list_path,sep=';').iloc[:,-1]
non_ijkeywords=non_ijkeywords_aux[non_ijkeywords_aux.apply(pd.notnull)].tolist()

df_dates_ymd = create_date_df(year_input)
start_date_list = (df_dates_ymd[df_dates_ymd.month.astype(int) % 3 == 1].groupby(["year", "month"])["day"].apply(min).reset_index().apply(lambda row: f"{str(row['year'])}-{str(row['month'])}-{str(row['day'])}", axis=1).tolist())
end_date_list = (df_dates_ymd[df_dates_ymd.month.astype(int)%3==0].groupby(["year","month"])["day"].apply(max).reset_index().apply(lambda row: f"{str(row['year'])}-{str(row['month'])}-{str(row['day'])}", axis=1).tolist())
dates_to_consider = tuple(zip(start_date_list, end_date_list))[::-1]

#In case we want to exclude any specific direcotry
dirs_to_exclude=[]

# for keyword in non_ijkeywords:

for ij_keyword in ij_keywords:
    df_tweets = pd.DataFrame(columns=df_cols_names)
    df_quotedTweets = pd.DataFrame(columns=df_cols_names)
    df_retweetedTweets = pd.DataFrame(columns=df_cols_names)
    directory_path = f"{main_folder_twitter}/{'Twitter'.title()}/{replace_non_alphanumeric(input_string=ij_keyword)}/"
    dir=replace_non_alphanumeric(input_string=ij_keyword)
    # directory_path = f"{main_folder_twitter}/{'Twitter'.title()}/{replace_non_alphanumeric(input_string=keyword)}/"
    # dir=replace_non_alphanumeric(input_string=keyword)
    if dir not in dirs_to_exclude:
        file_path_main_part=directory_path
        # file_path_main_part = f'{directory_path}{replace_non_alphanumeric(input_string=ij_keyword)}'
        print(file_path_main_part) # Does not include the part with the time that was saved
        if directory_path not in files_already_collected:  # If the information was already collected we not need to collect it again

            # AUX CODE
            main_folder = main_folder_twitter
            # nr_already_collected=1
            url = ij_keyword
            for newspaper in ['Twitter','Retweeted_Twitter','QuotedTweet_Twitter']:
                directory_path = f"{main_folder}/{newspaper.title()}/{replace_non_alphanumeric(input_string=url)}/".replace(
                    '//', '/')
                if os.path.isdir(directory_path):
                    files_aux=[file for file in os.listdir(directory_path) if f"AUX" in file and url in file]
                else:
                    files_aux=[]


                if (len(files_aux)!=len(dates_to_consider)) & (len(files_aux)>0):
                    if newspaper=='Twitter':
                        aux=pd.concat([pd.read_csv(file) for file in files_aux], axis=0)
                        df_tweets=aux.loc[:,[col for col in aux.columns if 'Unnamed' not in col]].drop_duplicates()
                    elif newspaper=='Retweeted_Twitter':
                        aux = pd.concat([pd.read_csv(file) for file in files_aux], axis=0)
                        df_retweetedTweets=aux.loc[:,[col for col in aux.columns if 'Unnamed' not in col]].drop_duplicates()
                    if newspaper=='QuotedTweet_Twitter':
                        aux = pd.concat([pd.read_csv(file) for file in files_aux], axis=0)
                        df_quotedTweets=aux.loc[:,[col for col in aux.columns if 'Unnamed' not in col]].drop_duplicates()


            nr_already_collected=len(files_aux)
            aux_idx = nr_already_collected
            # AUX CODE ended here

            for start_date, end_date in dates_to_consider[nr_already_collected:]: # AUX CODE after dates to consider
                # start_year = start_date[:4]
                # start_month = start_date[5:7]
                # end_year = end_date[:4]
                # end_month = end_date[5:7]
                nr_requests=0
                while True:
                    df_tweets_aux = asyncio.run(collect_tweets(f'{ij_keyword}', start_date, end_date, db_file_name))
                    # df_tweets_aux = asyncio.run(collect_tweets(f'{keyword} {ij_keyword}', start_date, end_date, db_file_name))
                    if not df_tweets_aux.empty:
                        df_tweets_aux.columns = df_cols_names

                        df_quotedTweet_aux_first = pd.DataFrame(
                            df_tweets_aux.apply(lambda row: get_tweet_info(row['original_tweet_id'], row['tweet_quotedTweet']),
                                                    axis=1).tolist())
                        df_quotedTweet_aux = df_quotedTweet_aux_first[df_quotedTweet_aux_first.tweet_id.apply(pd.notnull)]

                        df_retweetedTweet_aux_first = pd.DataFrame(df_tweets_aux.apply(
                            lambda row: get_tweet_info(row['original_tweet_id'], row['tweet_retweetedTweet']), axis=1).tolist())
                        df_retweetedTweet_aux = df_retweetedTweet_aux_first[df_retweetedTweet_aux_first.tweet_id.apply(pd.notnull)]
                        break
                    elif nr_requests==0:
                        check_status_accounts(db_file_name=db_file_name, file_name=accounts_credentials_file_name, file_format=accounts_credentials_file_format)
                        nr_requests+=1
                    elif nr_requests==1:
                        df_tweets_aux=pd.DataFrame(columns=df_cols_names)
                        df_retweetedTweet_aux=pd.DataFrame(columns=df_cols_names)
                        df_quotedTweet_aux=pd.DataFrame(columns=df_cols_names)
                        break
                df_tweets = pd.concat([df_tweets, df_tweets_aux], axis=0)
                df_retweetedTweets = pd.concat([df_retweetedTweets, df_retweetedTweet_aux], axis=0)
                df_quotedTweets = pd.concat([df_quotedTweets, df_quotedTweet_aux], axis=0)

                print("df_tweets.shape :" + str(df_tweets.shape))
                print("df_retweetedTweets.shape :" + str(df_retweetedTweets.shape))
                print("df_quotedTweets.shape :" + str(df_quotedTweets.shape))

                # AUX CODE start
                if aux_idx!=len(dates_to_consider): #AUX CODE
                    for df,newspaper in zip([df_tweets,df_retweetedTweets,df_quotedTweets],['Twitter','Retweeted_Twitter','QuotedTweet_Twitter']):
                        save_locally(
                            main_folder=main_folder_twitter,
                            df=df,
                            url=ij_keyword,
                            # url=keyword,
                            newspaper=newspaper,
                            start_year=min(dates_to_consider)[0][:4],
                            start_month=min(dates_to_consider)[0][5:7],
                            end_year=max(dates_to_consider)[1][:4],
                            end_month=max(dates_to_consider)[1][5:7],
                            twitter= True,
                            twitter_aux= f"AUX_{aux_idx}",
                        )
    

                
                aux_idx+=1 #AUX CODE end

            for df, newspaper in zip([df_tweets, df_retweetedTweets, df_quotedTweets],
                                     ['Twitter', 'Retweeted_Twitter', 'QuotedTweet_Twitter']):
                save_locally(
                    main_folder=main_folder_twitter,
                    df=df,
                    url = ij_keyword,
                    # url=keyword,
                    newspaper=newspaper,
                    start_year=min(dates_to_consider)[0][:4],
                    start_month=min(dates_to_consider)[0][5:7],
                    end_year=max(dates_to_consider)[1][:4],
                    end_month=max(dates_to_consider)[1][5:7],
                    # twitter= True,
                    # twitter_aux= ij_keyword,

                )




            print('STOP')
            print('--------------')
