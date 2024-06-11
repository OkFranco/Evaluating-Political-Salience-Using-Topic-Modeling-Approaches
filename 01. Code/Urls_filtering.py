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
path = "percent_coding_chars.csv"
directory_path = "all_news_final"
file_path_endpoints_not_to_consider = "Endpoint_not_to_include/"  # "/mnt/c/Users/Utilizador/OneDrive/Personal_time/JGS/Thesis publication/08. Aux files/"
main_folder_final = "all_text_filtered"
to_remove_ends = [
    ".css",
    ".gif",
    ".jpg",
    ".js",
    ".woff",
    ".swf",
    ".mp3",
    ".avi",
    ".exe",
    ".scr",
    ".pdf",
    ".xml",
    "Authentication",
    ".xsl",
    "/rss/",
    "/RSS/",
    ".rss",
    "robots.txt",
    "/fonts/",
    ".png",
    ".jpeg",
    ".svg",
    ".php",
]  # ,'.asp', ,'.aspx'
to_remove_ends_types = [
    "image",
    "audio",
    "application",
    "video",
    "xml",
    "css",
    "javascript",
    "plain",
]


aux_file = pd.read_csv(path, sep=";")


# Recursive function to go through each subdirectory and read files
# Call the function for the specified directory
dataframes_list = read_files_in_directory(directory_path)

df_all_urls_aux = pd.concat(dataframes_list, ignore_index=True)
df_all_urls = df_all_urls_aux.drop(
    [col for col in df_all_urls_aux.columns if "Unnamed" in col], axis=1
)

### 1 - remove urls with specific status
cond_status = (
    ~df_all_urls.status.astype("str")
    .str.split(".")
    .str[0]
    .isin(["401", "403", "404", "500", "530", "(Internal"])
)  # Internal refers to robots.txt
df_final_aux1 = df_all_urls_aux[cond_status].reset_index(drop=True)


### 2- remove repeated urls
df_final_aux1["url_aux"] = df_final_aux1.url.str.replace("https", "http", regex=True).str.replace("www.", "", regex=True)
df_final_aux2 = (
    df_final_aux1.sort_values(by="timestamp", ascending=True)
    .drop_duplicates(subset="url_aux", keep="first")
    .drop("url_aux", axis=1)
    .reset_index(drop=True)
)


### 3-Remove specific files types and urls' mime options


df_final_aux3 = df_final_aux2[
    (
        ~df_final_aux2["mime"].str.contains(
            "|".join([re.escape(s) for s in to_remove_ends_types]), regex=True
        )
    )
    & (
        ~df_final_aux2.url.str.contains(
            "|".join([re.escape(s) for s in to_remove_ends]), regex=True
        )
    )
].reset_index(drop=True)


### 4- remove urls with endpoints that are not to be considered

# define endpoints


aux_file_dict = aux_file.set_index("coded char")["original chars"].to_dict()
aux_df = (
    df_final_aux3.url.apply(lambda row: update_string(aux_file_dict, row))
    .str.split("\\.pt//")
    .str[-1]
    .str.split("\\.pt/")
    .str[-1]
    .str.split("\\.pt:80/")
    .str[-1]
    .str.split("/")
    .str[0]
    .str.split("\\.asp")
    .str[0]
    .str.split("\\.html")
    .str[0]
    .str.split("\\.php")
    .str[0]
    .str.split("\\?")
    .str[0]
)
df_final_aux3["endpoint_main"] = aux_df

# Collect endpoints not to include

endpoints_not_to_include_path_aux = [
    {
        file.split("_")[0]: open_json_dumped_file(
            file_path_endpoints_not_to_consider + file
        )
    }
    for file in os.listdir(file_path_endpoints_not_to_consider)
    if "endpoints_not_to_include" in file
]
endpoints_not_to_include_path = {}
for d in endpoints_not_to_include_path_aux:
    endpoints_not_to_include_path.update(d)

df_final_aux3["endpoint_not_to_include"] = df_final_aux3.apply(
    lambda row: row["endpoint_main"]
    in endpoints_not_to_include_path[row["newspaper"].lower()],
    axis=1,
)

df_final_aux4 = df_final_aux3[
    df_final_aux3.endpoint_not_to_include == False
].reset_index(drop=True)

df_final = df_final_aux4


# Save teh filtered urls locally
newspaper_domain_combos = (
    df_final.loc[:, ["newspaper", "domain_main"]]
    .drop_duplicates()
    .apply(lambda row: tuple(row), axis=1)
    .tolist()
)

for combo in newspaper_domain_combos:
    df_to_save = df_final[
        (df_final.newspaper == combo[0]) & (df_final.domain_main == combo[1])
    ].reset_index(drop=True)

    save_locally(
        main_folder=main_folder_final, df=df_to_save, url=combo[1], newspaper=combo[0]
    )
