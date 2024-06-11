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
source_directory = "all_news_text_filtering1"
destination_directory = "all_news_text_filtering1_wdate"

dataframes_list = read_files_in_directory(directory=source_directory)
df = pd.concat(dataframes_list, axis=0)
print("df.columns" +str(df.columns))

dates=df.apply(lambda row: collect_date_approach3(row['full_text_approach3'],row['url']),axis=1)
df['date_published_inferred']=dates
df['date_final']=df.apply(lambda row: define_final_date(row),axis=1)

os.system(f'rm -r {source_directory}')

save_locally(
    main_folder=destination_directory,
    df=df,
    url="all_text",
    newspaper="News",csv_pickle='pickle'
)


