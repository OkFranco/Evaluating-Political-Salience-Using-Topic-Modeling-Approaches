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
import re
from utils_topification import *


# 1- define local variables and import files
doc_type = "tweets"
tokenization_approach = "unigrams_bigrams"
topification_approach = "nmf"
embedding_approach = "doc2vec"
n_components_list = '' #TBD
nr_top_words = 50
main_destination_folder = f"topification/{doc_type}/{topification_approach}/"
destination_file_name_main = (
    f"jisdd_{tokenization_approach}_nostopwords_{topification_approach}_"
)
col_with_text = "full_text_final_cleaned_nostopwords"


main_folder_source = "all_tweets_final"

portuguese_stopwords_final_path='portuguese_stopwords_final.json'

dataframes_list_aux = read_files_in_directory(directory=main_folder_source)
df_text_aux = pd.concat(dataframes_list_aux, axis=0)

df_text=df_text_aux[(df_text_aux.cond_len_2==True)&
                    (df_text_aux.cond_sdd_ji==True)].reset_index(drop=True)

documents = df_text.loc[:, col_with_text].tolist()

with open(portuguese_stopwords_final_path, 'r') as json_file:
    portuguese_stopwords_final=json.load(json_file)


# 3- implement topification
if topification_approach == "nmf":
    results = nmf_topification(
        documents,
        tokenization_approach,
        n_components_list,
        nr_top_words,
        embedding_approach,
        portuguese_stopwords_final,
        topification_approach
    )

    for name,model in results:
        create_directory(main_destination_folder)
        model.save(
            main_destination_folder + destination_file_name_main  +model.name
        )
    # save_dictionary_to_file(
    #     error_list, main_destination_folder, destination_file_name_main + "error_list"
    # )
    # save_dictionary_to_file(
    #     H_list, main_destination_folder, destination_file_name_main + "H_list"
    # )
    # save_dictionary_to_file(
    #     W_list, main_destination_folder, destination_file_name_main + "W_list"
    # )
    # save_dictionary_to_file(
    #     results, main_destination_folder, destination_file_name_main + "results"
    # )

elif topification_approach == "top2vec":
    model = top2vec_topification(documents, tokenization_approach, embedding_approach, portuguese_stopwords_final,topification_approach)
    create_directory(main_destination_folder)
    model.save(
        main_destination_folder + destination_file_name_main + embedding_approach
    )
