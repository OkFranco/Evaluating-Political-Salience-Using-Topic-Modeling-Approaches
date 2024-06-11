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



main_folder_source = "all_news_text_filtering1_wdate_cleaned"
main_folder_destination = "all_news_text_filtering1_wdate_cleaned_filtering2"  # TBD 'all_tweets'
start_number=0
end_number=1000 #length should be around 300 so 1000 will include all options
collect_load_idxs_to_compare="collect"
file_path_idxs='news_aux_info'
aux_file_idxs='idxs_to_compare'
directory_child_all_idx_combinations="all_idx_combinations"
directory_child_idx_to_keep="idx_to_keep"
threshold_rel_diff_nr_words=0.1 #to fill
threshold_collection_similarity=0.95

dataframes_list_aux = read_files_in_directory(directory=main_folder_source)
df = pd.concat(dataframes_list_aux, axis=0)
print("df.columns" +str(df.columns))


if collect_load_idxs_to_compare=='collect':
    domains_already_considered=os.listdir(file_path_idxs+'/'+directory_child_all_idx_combinations.title())
    df_idx_to_collect=df[df.domain_main.apply(replace_non_alphanumeric).isin(domains_already_considered)==False]

    domains_to_consider=list(df_idx_to_collect.domain_main.value_counts().index)[:10]

    find_idxs_to_compare(df_file_1_filtered=df_idx_to_collect[df_idx_to_collect.domain_main.isin(domains_to_consider)],
                         threshold_rel_diff_nr_words=threshold_rel_diff_nr_words, start_number=start_number, end_number=end_number, file_path=file_path_idxs,threshold_collection_similarity=threshold_collection_similarity,directory_child_all_idx_combinations=directory_child_all_idx_combinations,directory_child_idx_to_keep=directory_child_idx_to_keep)
elif collect_load_idxs_to_compare=='read':
    pass
    # idx_combinations_to_compare_aux=read_files_in_directory(file_path_idxs)
    # idx_combinations_to_compare=list(pd.concat(idx_combinations_to_compare_aux,axis=1).iloc[:,-1].unique())
    # for file in collect_filenames_in_directory(file_path_idxs):
    #     idx_combinations_to_compare_aux.update(open_json_dumped_file(file))
    # idx_combinations_to_compare=[]
    # for key, idxs in idx_combinations_to_compare_aux.items():
    #     idx_combinations_to_compare+=list(idxs)
    # print("len(idx_combinations_to_compare) :"+str(len(idx_combinations_to_compare)))
    # print("[len(idxs) for key, idxs in idx_combinations_to_compare_aux] :"+str([len(idxs) for key, idxs in idx_combinations_to_compare_aux.items()]))

# idx_to_keep_final_aux=read_files_in_directory(file_path_idxs)
# idx_to_keep_final=pd.concat(idx_to_keep_final_aux,axis=1).iloc[:,-1].unique()
#
# all_idx_combinations_aux=read_files_in_directory(directory_child_all_idx_combinations)
# all_idx_combinations=pd.concat(all_idx_combinations_aux,axis=1).iloc[:,-1].unique()
#
# # idx_to_keep_final=calculate_collection_similarity(df_file_1_filtered=df,results_threshold=results_threshold,idx_combinations_to_compare=idx_combinations_to_compare,file_path=file_path_idxs)
# #
# # all_idx_combinations_aux=list(set(pd.Series(idx_combinations_to_compare).apply(lambda row:row[0]).tolist()+pd.Series(idx_combinations_to_compare).apply(lambda row:row[1]).tolist()))
# # all_idx_combinations=df.iloc[all_idx_combinations_aux].index
# #
# all_idxs=list(set(df.index)-set(all_idx_combinations)|set(idx_to_keep_final))
#
# cond_similar_news_to_keep=df.reset_index().loc[:,'index'].isin(all_idxs)
# cond_similar_news_to_keep.index=df.index
#
# df_file_final=df[cond_similar_news_to_keep].reset_index(drop=True)
#
# save_locally(
#     main_folder=main_folder_destination, df=df_file_final, url="all_text", newspaper="News",csv_pickle='pickle'
# )
#
# os.system(f'rm -r {main_folder_source}')