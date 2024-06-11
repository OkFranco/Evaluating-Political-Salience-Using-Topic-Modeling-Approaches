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
path = "dominios_jornais_v6.csv"
directory_path = "all_news_final"
directory_path_after = "all_text_filtered"
errors_directory = "Errors"

domains = pd.read_csv(path, sep=";")

dataframes_list = read_files_in_directory(directory_path)
dataframes_list_after = read_files_in_directory(directory_path_after)

domains["dominio_treated"] = domains.dominio.apply(replace_non_alphanumeric)


# outputs to save
output1 = []
output2 = pd.DataFrame()

# Check some urls were filtered, but no cols were removed:
df_all_before=pd.concat(dataframes_list,axis=0)
df_all_after=pd.concat(dataframes_list_after,axis=0)

if df_all_after.shape[0]==df_all_before.shape[0]:
    error_message = (
        f"ERROR_A: No urls was filtered for!"
    )
    # print(error_message)
    output1.append(error_message)

cols_df_all_after=set(df_all_after.columns)
cols_df_all_before=set(df_all_before.columns)

if len(cols_df_all_before-cols_df_all_after)!=0:
    error_message = (
        f"ERROR_B: {cols_df_all_before-cols_df_all_after} columns were removed!"
    )
    # print(error_message)
    output1.append(error_message)


#Save outputs with errors
try:
    if pd.Series(output1).shape[0]!=0:
        pd.Series(output1).to_csv(
            f'{errors_directory}/Error_1_check_XX_urls_script_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
except:
    create_directory(errors_directory)
    if pd.Series(output1).shape[0]!=0:
        pd.Series(output1).to_csv(
            f'{errors_directory}/Error_1_check_XX_urls_script_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
