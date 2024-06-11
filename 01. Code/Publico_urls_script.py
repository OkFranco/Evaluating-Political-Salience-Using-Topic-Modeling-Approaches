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

##INPUTS
year_input = list(range(2007, 2023))
newspaper = "Publico"
main_folder_to_consider = "all_news_final"


# path="/mnt/c/Users/Utilizador/OneDrive/Personal_time/JGS/Thesis publication/08. Aux files/dominios_jornais_v2.csv"
path = "/users3/spac/jfranco/dominios_jornais_v6.csv"

path_log_file='~/logs.csv'

# in case we want to skip specific urls
domains = pd.read_csv(path, sep=";")
domains = domains.loc[:,[col for col in domains.columns if 'Unnamed' not in col]]

urls_to_exclude = list(set(domains.apply(
    lambda row: identify_urls_not_consider(
        main_folder=main_folder_to_consider, newspaper=newspaper, url=row["dominio"], to_collect=row['to_collect']
    ),
    axis=1,
).tolist()))

print('urls_to_exclude :'+str(urls_to_exclude))

domains_to_consider_notmain = domains[
    (domains.jornal == newspaper)
    & (domains.main == False)
    & (~domains.dominio.isin(urls_to_exclude))
].reset_index(drop=True)

domains_to_consider_notmain[
    "dominio"
] = domains_to_consider_notmain.dominio.str.strip().tolist()

to_collect_status_update_III = collect_all_news_urls(
    years=year_input,
    domains_to_consider=domains_to_consider_notmain,
    newspaper=newspaper,
    collect_urls_per_year_yearmonth="y",
    save_per_date=False,
    publico_specific_date_inurl=False,
    main_folder=main_folder_to_consider,
    path_aux_file=path ,
    path_log_file=path_log_file
)

domains_to_consider_main = domains[
    (domains.jornal == newspaper)
    & (domains.main == True)
    & (~domains.dominio.isin(urls_to_exclude))
].reset_index(drop=True)
domains_to_consider_main[
    "dominio"
] = domains_to_consider_main.dominio.str.strip().tolist()

to_collect_status_update_I = collect_all_news_urls(
    years=year_input,
    domains_to_consider=domains_to_consider_main,
    newspaper=newspaper,
    collect_urls_per_year_yearmonth="ym",
    save_per_date=True,
    publico_specific_date_inurl=False,
    main_folder=main_folder_to_consider,
    path_aux_file=path ,
path_log_file=path_log_file
)

to_collect_status_update_II = collect_all_news_urls(
    years=year_input,
    domains_to_consider=domains_to_consider_main,
    newspaper=newspaper,
    collect_urls_per_year_yearmonth="ym",
    save_per_date=True,
    publico_specific_date_inurl=True,
    main_folder=main_folder_to_consider,
    path_aux_file=path ,
path_log_file=path_log_file
)



#to_collect_status_update_I.update(to_collect_status_update_II)
#to_collect_status_update_I.update(to_collect_status_update_III)

# We need to re read the file, in case the script from the other newspaper has run in parallel and update the file
# meanwhile
#domains = pd.read_csv(path, sep=";")
#domains["to_collect"] = domains.apply(
#    lambda row: update_to_collect_status_update(
#        row=row, to_collect_status_update=to_collect_status_update_I
#    ),axis=1
#)
#domains.to_csv(path, sep=";")
