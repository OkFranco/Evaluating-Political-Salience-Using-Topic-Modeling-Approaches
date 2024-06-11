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

# path="/mnt/c/Users/Utilizador/OneDrive/Personal_time/JGS/Thesis publication/08. Aux files/percent_coding_chars.csv"
initial_directory_path_news = "all_news_final" #to fill
destination_directory_path_news = "all_news_text_filtered_cleaned_final"
initial_directory_path_tweets = "all_tweets" #to fill
destination_directory_path_tweets = "all_tweets_text_filtered_cleaned_final"
portuguese_stopwords_final_path='portuguese_stopwords_final.json'


dataframes_list_news = read_files_in_directory(initial_directory_path_news)
df_news = pd.concat(dataframes_list_news, ignore_index=True)

dataframes_list_tweets = read_files_in_directory(initial_directory_path_tweets)
df_tweets = pd.concat(dataframes_list_tweets, ignore_index=True)

with open(portuguese_stopwords_final_path, 'r') as json_file:
    portuguese_stopwords_final=json.load(json_file)

df_news['full_text']=df_news['full_text'].apply(extensive_text_cleaning).apply(lambda text: remove_stop_words(text,portuguese_stopwords_final))



save_locally(
    main_folder=destination_directory_path_news, df=df_news, url='All', newspaper='All'
)

save_locally(
    main_folder=destination_directory_path_tweets, df=df_tweets, url='All', newspaper='All'
)
