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


main_folder_source = "all_news_text_filtering1_wdate"
main_folder_destination = "all_news_text_filtering1_wdate_cleaned"  # TBD 'all_tweets'
keywords_list_path = "ijsdd_keywords.csv"
portuguese_stopwords_final_path = "portuguese_stopwords_final.json"
text_size_lower_bound = 100 #the value from urls_news_filtering_I
min_len=text_size_lower_bound


with open(portuguese_stopwords_final_path, "r") as json_file:
    portuguese_stopwords_final = json.load(json_file)

ij_keywords_aux = pd.read_csv(keywords_list_path)
ij_keywords = (
    ij_keywords_aux[ij_keywords_aux.apply(pd.notnull)]
    .reset_index(drop=True)
    .iloc[:, -1]
    .tolist()
)

dataframes_list_aux = read_files_in_directory(directory=main_folder_source)
df = pd.concat(dataframes_list_aux, axis=0)
print("df.columns" +str(df.columns))


file_filtered_full_text_cleaned = df.full_text.apply(
    extensive_text_cleaning
).apply(lambda text: remove_stop_words(text, portuguese_stopwords_final))
df["full_text_final_cleaned_nostopwords"] = file_filtered_full_text_cleaned
print("line 45")
#Add conditions to leverage as filters during the modelling
pattern_aux = r"" + "|".join(map(re.escape, ij_keywords))
df_sdd_ji = (
    df["full_text_final_cleaned_nostopwords"]
    .apply(lambda text: "/".join(re.findall(pattern_aux, text, re.IGNORECASE)))
    .apply(lambda val: len(val) != 0)
)
print("line 53")
df["cond_sdd_ji"] = df_sdd_ji
df[f"cond_len_{min_len}"] = (
    df["full_text_final_cleaned_nostopwords"].str.split().apply(len) >= min_len
)
print("line 58")
print("line 63")
os.system(f'rm -r {main_folder_source}')

save_locally(
    main_folder=main_folder_destination, df=df, url="all_text", newspaper="News",csv_pickle='pickle'
)

