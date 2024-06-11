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
dirs_urls_filtered = collect_filenames_in_directory(directory_path)
dirs_urls_filtered_treated = pd.Series(dirs_urls_filtered).str.split("2024").str[0]
print('here2')
min_size=30000
nr_parts=1000

## Collect information
for idx, df in enumerate(dataframes_list):

    #### By checking each file after an interaction we can account for jobs in parallel

    ##Identify news that were already collected
    if idx==0:
        print('here3')

    # dirs_text_chunk = collect_filenames_in_directory(directory_path+'_before_chunk')  # +pd.read_csv(news_alread_saved).iloc[:,-1].tolist()
    # dirs_text_treated_chunk = (pd.Series(dirs_text_chunk).str.replace('_before_chunk','').str.split("2024").str[0])

    dirs_text = collect_filenames_in_directory(main_folder_final)#+pd.read_csv(news_alread_saved).iloc[:,-1].tolist()
    dirs_text_treated = (pd.Series(dirs_text).str.split("2024").str[0].str.replace(main_folder_final, directory_path))

    if idx==0:
        print('here4')

    # Indexes of the files that were not already collected and saved. This approach can be implemented because all files have 2024 as year of collection and both read_files_in_directory and collect_filenames_in_directory operate over the sames files in teh same order
    files_to_consider = list(dirs_urls_filtered_treated[~(dirs_urls_filtered_treated.isin(dirs_text_treated.tolist()))].index)

    if idx==0:
        print('here5')

    # Dataframes with size bigger than 30000 will be divided into sub files
    if (idx in files_to_consider) & (df.shape[0]>=min_size): #TOREMOVE

        file_name_to_chunks=dirs_urls_filtered[idx]
        print(f'FILE TO CHUNCK: {file_name_to_chunks}')
        domain_main = list(set(df.domain_main.tolist()))[0]
        newspaper = list(set(df.newspaper.tolist()))[0]


        #Copy the df to other directory not to lose the information
        save_locally(
            main_folder=f"{directory_path}_before_chunk",
            df=df,
            url=domain_main,
            newspaper=newspaper,
        )

        #Divide into chunks and save in the correct directory
        indices = np.array_split(df.index, nr_parts)
        dfs = [df.iloc[index] for index in indices]
        for idx_new,df_new in enumerate(dfs):
            # This domain is the name of the folder child to save onlen

            save_locally(
                    main_folder=directory_path,
                    df=df_new.reset_index(drop=True),
                    url=domain_main,
                    newspaper=newspaper,
                    twitter=True,
                    twitter_aux=str(idx_new)
                )

        # Erase the original df, since we already saved a copy there is no problem
        os.remove(file_name_to_chunks)

