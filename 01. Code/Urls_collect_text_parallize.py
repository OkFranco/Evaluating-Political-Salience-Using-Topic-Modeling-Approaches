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
import sys


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3.10 my.py start_index end_index")
        sys.exit(1)

    start_index = int(sys.argv[1])
    end_index = int(sys.argv[2])

    ##Inputs
    aux_file = pd.read_csv("percent_coding_chars.csv", sep=";")
    directory_path = "all_text_filtered"
    main_folder_final = "all_news_text"
    news_alread_saved='news_files_already_saved_locally.csv'

    ##Code
    # Recursive function to go through each subdirectory and read files
    # Call the function for the specified directory

    dataframes_list = read_files_in_directory(directory_path)

    ##Identify news that were already collected

    dirs_urls_filtered = collect_filenames_in_directory(directory_path)[::-1]
    dirs_urls_filtered_treated = pd.Series(dirs_urls_filtered).str.split("2024").str[0]
    dirs_text = collect_filenames_in_directory(main_folder_final)[::-1] #+pd.read_csv(news_alread_saved).iloc[:,-1].tolist()
    dirs_text_treated = (pd.Series(dirs_text).str.split("2024").str[0].str.replace(main_folder_final, directory_path))

    # Indexes of the files that were not already collected and saved. This approach can be implemented because all files have 2024 as year of collection and both read_files_in_directory and collect_filenames_in_directory operate over the sames files in teh same order
    files_to_consider_aux = list(dirs_urls_filtered_treated[~dirs_urls_filtered_treated.isin(dirs_text_treated.tolist())].index)[::-1]




    files_to_consider=files_to_consider_aux[start_index:end_index]

    print("start_index:end_index :"+str(str(start_index)+'_'+str(end_index)))

    ## Collect information
    for idx, df in enumerate(dataframes_list[::-1]):
        if idx in files_to_consider:
            # Newspaper_aux is assigned in read_files_in_directory, and was previously used to create newspaper and domain_main cols

            # This domain is the name of the folder child to save on
            domain_main = list(set(df.domain_main.tolist()))[0]
            newspaper = list(set(df.newspaper.tolist()))[0]

            if len(set(df.newspaper.tolist())) != 1:
                print(f"ERROR in file {idx}")

            #TOREMOVE
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
            start_file_name = dirs_urls_filtered[idx].split('/')[-1].split('_')[0]
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

