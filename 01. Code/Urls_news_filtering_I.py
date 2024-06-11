import requests
import pandas as pd
from datetime import datetime
import time
import unicodedata
import random
from sklearn.preprocessing import OneHotEncoder
import ast
import json
import re
from itertools import combinations
import numpy as np
import os
import pandas as pd
from datetime import datetime, timedelta
from utils_collection_urls import *

##Inputs
source_directory = "all_news_text"
destination_directory = "all_news_text_filtering1"

full_text_approach1 = ["CHECK URL", "None"]
full_text_approach2 = [
    "None",
    "No article in textextracted",
    "Object moved Object moved to here",
    "Object moved Object Moved This object may be found here",
    "Catchable fatal error",
]
full_text_approach3 = [
    "None",
    "No article in textextracted",
    "Object moved Object moved to here",
    "Object moved Object Moved This object may be found here",
    "Catchable fatal error",
    "No article in noFrame",
    "encontrada (404) - Arquivo.pt",
]

text_size_upper_bound = 10000
text_size_lower_bound = 100

dataframes_list = read_files_in_directory(directory=source_directory)

df = pd.concat(dataframes_list, axis=0)
df = df.reset_index(drop=True) #TOREMOVE
print("df.columns" +str(df.columns))
## Define final full_text variable

pattern_approach1 = r"\b(?:" + "|".join(map(re.escape, full_text_approach1)) + r")\b"
cond_approach1 = df.full_text_approach1.astype(str).str.contains(pattern_approach1)

pattern_approach2 = r"\b(?:" + "|".join(map(re.escape, full_text_approach2)) + r")\b"
cond_approach2 = df.full_text_approach2.astype(str).str.contains(pattern_approach2)

df["cond_approach1"] = cond_approach1
df["cond_approach2"] = cond_approach2


df["full_text"] = df.apply(lambda row: assign_full_text(row), axis=1)
df_final_aux = df.loc[
    :,
    [col for col in df.columns if ("approach1" not in col) & ("approach2" not in col)],
]
df_final_aux["domain_main"] = df_final_aux.ID.apply(
    lambda input_string: re.split(r"(\d+)", input_string, 1)[0].strip("_")
)
df_final_aux = df_final_aux.sort_values(by="date_published", ascending=True)

### Remove observations with no text collected (only null if appraoch 1 and approach 2 do not have text)

cond_text_not_null = df_final_aux.full_text.apply(pd.notnull)

### Remove text with exactly the same text

cond_same_text_news = df_final_aux.full_text.duplicated(keep="first")

### Remove text based on number of words
df_final_aux["nr_words"] = df_final_aux.full_text.astype(str).str.split().apply(
    len
)
cond_min_nr_words = df_final_aux.nr_words >= text_size_lower_bound
cond_max_nr_words = df_final_aux.nr_words <= text_size_upper_bound

df_final = df_final_aux[
    (cond_min_nr_words)
    & (cond_max_nr_words)
    & (~cond_same_text_news)
    & (cond_text_not_null)
].reset_index(drop=True)


save_locally(
    main_folder=destination_directory,
    df=df_final,
    url="all_text",
    newspaper="News",
    csv_pickle='pickle'
)
