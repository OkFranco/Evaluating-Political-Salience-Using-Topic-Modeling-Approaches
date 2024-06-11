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
from unidecode import unidecode

main_folder_source = "all_tweets_filtered"
main_folder_destination = "all_tweets_final"  # TBD 'all_tweets'
keywords_list_path = "ijsdd_keywords.csv"
portuguese_stopwords_final_path = "portuguese_stopwords_final.json"
min_len = 2



# with open(portuguese_stopwords_final_path, "r") as json_file:
#     portuguese_stopwords_final = json.load(json_file)

portuguese_stopwords_final=open_json_dumped_file(portuguese_stopwords_final_path)

ij_keywords_aux = pd.read_csv(keywords_list_path)
ij_keywords = (
    ij_keywords_aux[ij_keywords_aux.apply(pd.notnull)]
    .reset_index(drop=True)
    .iloc[:, -1]
    .tolist()
)

dataframes_list_aux = read_files_in_directory(directory=main_folder_source)
df_tweets = pd.concat(dataframes_list_aux, axis=0)

file_filtered_tweet_text_cleaned = df_tweets.tweet_text.apply(
    extensive_text_cleaning
).apply(lambda text: remove_stop_words(text, portuguese_stopwords_final))
df_tweets["full_text_final_cleaned_nostopwords"] = file_filtered_tweet_text_cleaned
df_tweets["date"] = df_tweets["tweet_date"].apply(
    lambda datetime_str: datetime.fromisoformat(datetime_str.replace("Z", "+00:00"))
)



#Add conditions to leverage as filters during the modelling
pattern_aux = r"" + "|".join(map(re.escape, ij_keywords+['futuras geracoes']))
df_tweets_sdd_ji = (
    df_tweets["full_text_final_cleaned_nostopwords"]
    .apply(lambda text: "/".join(re.findall(pattern_aux, text, re.IGNORECASE)))
    .apply(lambda val: len(val) != 0)
)
df_tweets["cond_sdd_ji"] = df_tweets_sdd_ji
df_tweets[f"cond_len_{min_len}"] = (
    df_tweets["full_text_final_cleaned_nostopwords"].str.split().apply(len) >= min_len
)

save_locally(
    main_folder=main_folder_destination, df=df_tweets, url="final", newspaper="Twitter",csv_pickle='pickle'
)
