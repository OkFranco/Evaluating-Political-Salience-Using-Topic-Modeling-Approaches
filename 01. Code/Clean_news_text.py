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
directory_path='all_news_text'

##Code
# Recursive function to go through each subdirectory and read files
# Call the function for the specified directory

dataframes_list_aux=read_files_in_directory(directory_path)


df = pd.concat(dataframes_list_auxs,axis=1).reset_index(drop=True)

##Clean files

#

## Save locally
save_locally(main_folder=main_folder_final,df=df_w_texts,url=domain_main,newspaper=newspaper)