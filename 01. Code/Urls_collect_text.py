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
aux_file = pd.read_csv("percent_coding_chars.csv", sep=";")
directory_path = "all_text_filtered"
main_folder_final = "all_news_text"
news_alread_saved='news_files_already_saved_locally.csv'

##Code
# Recursive function to go through each subdirectory and read files
# Call the function for the specified directory
print('here')
dataframes_list = read_files_in_directory(directory_path)
print('here1')


## Collect information
for idx, df in enumerate(dataframes_list[::-1]):

    #### By checking each file after an interaction we can account for jobs in parallel

    ##Identify news that were already collected
    dirs_urls_filtered = collect_filenames_in_directory(directory_path)[::-1]
    dirs_urls_filtered_treated = pd.Series(dirs_urls_filtered).str.split("2024").str[0]
    dirs_text = collect_filenames_in_directory(main_folder_final)[::-1] #+pd.read_csv(news_alread_saved).iloc[:,-1].tolist()
    dirs_text_treated = (pd.Series(dirs_text).str.split("2024").str[0].str.replace(main_folder_final, directory_path))


    # Indexes of the files that were not already collected and saved. This approach can be implemented because all files have 2024 as year of collection and both read_files_in_directory and collect_filenames_in_directory operate over the sames files in teh same order

    files_to_consider = list(dirs_urls_filtered_treated[~dirs_urls_filtered_treated.isin(dirs_text_treated.tolist())].index)
    print(idx)
    if idx in files_to_consider:
        # Newspaper_aux is assigned in read_files_in_directory, and was previously used to create newspaper and domain_main cols

        # This domain is the name of the folder child to save onlen
        domain_main = list(set(df.domain_main.tolist()))[0]
        newspaper = list(set(df.newspaper.tolist()))[0]

        if len(set(df.newspaper.tolist())) != 1:
            print(f"ERROR in file {idx}")

        series_w_texts = df.apply(lambda row: collect_text(row, aux_file), axis=1)
        df_w_texts = pd.DataFrame(
            series_w_texts.tolist(),
            columns=[
                "url_request",
                "url",
                # "full_text_final",
                "full_text_approach1",
                "full_text_approach2",
                "full_text_approach3",
                "ID",
                "timestamp",
                "date_published",
            ],
        )

        # df_w_texts["date_published_approach3"] = df_w_texts.full_text_approach3.apply(collect_date_approach3)
        # df_w_texts["date_published"] = df_w_texts.apply(
        #     lambda row: row["date_published_approach3"][0]
        #     if (pd.isnull(row["date_published"]))
        #     & (len(row["date_published_approach3"]) == 1)
        #     else row["date_published"],
        #     axis=1,
        # )
        start_file_name=dirs_urls_filtered[idx].split('/')[-1].split('_')[0]
        print(f'dirs_urls_filtered[idx] :{dirs_urls_filtered[idx]}')
        print(f"start_file_name :{start_file_name}")
        print(f"start_file_name[0].isdigit() :{str(start_file_name[0].isdigit())}")
        if start_file_name[0].isdigit():
            save_locally(
                main_folder=main_folder_final,
                df=df_w_texts,
                url=domain_main,
                newspaper=newspaper,
                twitter=True,
                twitter_aux=start_file_name
            )

        else:

            save_locally(
                    main_folder=main_folder_final,
                    df=df_w_texts,
                    url=domain_main,
                    newspaper=newspaper,
                )
