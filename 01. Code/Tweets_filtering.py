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
from unidecode import unidecode

main_folder_twitter='all_tweets_thesis'
main_folder_destination = 'all_tweets_filtered'
path_keywords= "/users3/spac/jfranco/tweets_keywords_separated.csv"
ij_keywords_list_path = "ijsdd_keywords.csv"
path_delimitadores_PT="/users3/spac/jfranco/Coordenadas_limitadoras_PT.csv"
locais_brasil = ["São Paulo", "Rio de Janeiro", "Brasília", "Salvador", "Fortaleza", "Belo Horizonte", "Manaus",
                 "Curitiba", "Recife", "Porto Alegre", "Belém", "Goiânia", "São Luís", "Rio Branco", "Maceió", "Macapá",
                 "Vitória", "Cuiabá", "Campo Grande", "João Pessoa", "Teresina", "Natal", "Porto Velho", "Boa Vista",
                 "Florianópolis", "Aracaju", "Palmas", "Acre", "Alagoas", "Amapá", "Amazonas", "Bahia", "Ceará",
                 "Espírito Santo", "Goiás", "Maranhão", "Mato Grosso", "Mato Grosso do Sul", "Minas Gerais", "Pará",
                 "Paraíba", "Paraná", "Pernambuco", "Piauí", "Rio Grande do Norte", "Rio Grande do Sul", "Rondônia",
                 "Roraima", "Santa Catarina", "Sergipe", "Tocantins","Brasil",'Brazil']
locais_angola = ["Bengo", "Benguela", "Bié", "Cabinda", "Cuando Cubango", "Cuanza Norte", "Cuanza Sul", "Cunene",
                 "Huambo", "Huíla", "Luanda", "Lunda Norte", "Lunda Sul", "Malanje", "Moxico", "Namibe", "Uíge",
                 "Zaire", "Kuito", "Menongue", "N'Dalatando", "Sumbe", "Ondjiva", "Lubango", "Luena", "Dundo",
                 "Saurimo", "Mbanza Kongo", "Angola"]
locais_cabo_verde = ["Santo Antão", "São Vicente", "Santa Luzia", "São Nicolau", "Sal", "Boa Vista", "Maio", "Santiago",
                     "Fogo", "Brava", "Ponta do Sol", "Mindelo", "Ribeira Brava", "Espargos", "Sal Rei", "Vila do Maio",
                     "Praia", "São Filipe", "Vila Nova Sintra", ]
locais_guine = ["Bafatá", "Biombo", "Bissau", "Bolama", "Cacheu", "Gabú", "Oio", "Quinara", "Tombali", "Quinhámel",
                "Farim", "Mansôa", "Fulacunda", "Catió","Guiné-Bissau"]
locais_mocambique = ["Cabo Delgado", "Gaza", "Inhambane", "Manica", "Maputo", "Nampula", "Niassa", "Sofala", "Tete",
                     "Zambézia", "Pemba", "Xai-Xai", "Chimoio", "Matola", "Lichinga", "Beira", "Quelimane","Moçambique"]
locais_sao_tome = ["São Tomé", "Príncipe", "Santo António",'São Tomé e Príncipe']
locais_timor = ["Bobonaro", "Cova Lima", "Díli", "Ermera", "Lautém", "Liquiçá", "Manatuto", "Manufahi", "Oecusse",
                "Viqueque", "Maliana", "Suai", "Gleno", "Lospalos", "Timor-Leste","Timor"]

PT_coordinates_limits=pd.read_csv(path_delimitadores_PT,sep=';')



dataframes_list_aux=read_files_in_directory(main_folder_twitter)
df_tweets_aux=pd.concat(dataframes_list_aux,axis=0)

if 'thesis' in main_folder_twitter:
    df_tweets_aux=df_tweets_aux.rename(columns={"text":"tweet_text","datetime":"tweet_date"})
    df_tweets_aux["tweet_lang"]="pt"
    df_tweets_aux["tweet_coordinates"]=df_tweets_aux.apply(lambda row: np.NaN,axis=1)
    df_tweets_aux["user_location"]=np.NaN
    df_tweets_aux["tweet_place"]=np.NaN
    df_tweets_aux["original_tweet_id"]=df_tweets_aux.reset_index().loc[:,'index'].tolist()


#1- Find the unique tweets to consider. Bad requests to teh API can lead to the same tweet to be collected mulitple times
#2- to account for unique text from unique people (this will remove the effect from retweets and quoted).
df_tweets=df_tweets_aux.drop_duplicates(subset=['original_tweet_id','tweet_text']).reset_index(drop=True)
print("line 63 df_tweets.tweet_text.apply(pd.isnull).value_counts() :"+str(df_tweets.tweet_text.apply(pd.isnull).value_counts()))

#COND I
#Identify tweets that mention the keywords (we did the search by keyword, so if there is no keyword it should be removed)
#We allow that keywords that were not used in the query are also searched during the process (ex.: keyword='ambiente', tweet='estado do clima', tweet maintains because includes clima)
tweets_keywords_jisdd_words_aux = pd.read_csv(ij_keywords_list_path)
tweets_keywords_jisdd_words = (
    tweets_keywords_jisdd_words_aux[tweets_keywords_jisdd_words_aux.apply(pd.notnull)]
    .reset_index(drop=True)
    .iloc[:, -1]
    .tolist()
)
tweets_keywords_generic_words_aux=pd.read_csv(path_keywords,sep=';').loc[:,'generic_words'].str.lower()
tweets_keywords_generic_words=tweets_keywords_generic_words_aux[tweets_keywords_generic_words_aux.apply(pd.notnull)].apply(unidecode).tolist()




print("line 81 df_tweets.tweet_text.apply(pd.isnull).value_counts() :"+str(df_tweets.tweet_text.apply(pd.isnull).value_counts()))
cond_keywords_I=df_tweets.tweet_text.apply(unidecode).apply(lambda row: check_keywords(row,tweets_keywords_jisdd_words))
# cond_keywords_II=df_tweets.tweet_text.apply(unidecode).apply(lambda row: check_keywords(row,tweets_keywords_generic_words))
cond_keywords=cond_keywords_I #& cond_keywords_II
print("line 85 df_tweets.tweet_text.apply(pd.isnull).value_counts() :"+str(df_tweets.tweet_text.apply(pd.isnull).value_counts()))
#COND II
cond_lang=df_tweets.tweet_lang=='pt'

print("df_tweets_aux.tweet_coordinates.value_counts() :"+str(df_tweets_aux.tweet_coordinates.value_counts()))
print("line 90 df_tweets.tweet_text.apply(pd.isnull).value_counts() :"+str(df_tweets.tweet_text.apply(pd.isnull).value_counts()))
#COND III
cond_III_a = df_tweets.tweet_coordinates.apply(
    lambda row: check_country_coordinates(row, PT_coordinates_limits))
print("line 94 df_tweets.tweet_text.apply(pd.isnull).value_counts() :"+str(df_tweets.tweet_text.apply(pd.isnull).value_counts()))
locais_pt_lang_nonPT = locais_brasil + locais_angola + locais_cabo_verde + locais_guine + locais_mocambique + locais_sao_tome + locais_timor

pattern_locais_pt_lang_nonPT = r'\b(?:' + '|'.join(
    map(re.escape, [unidecode(local) for local in locais_pt_lang_nonPT])) + r')\b'

if df_tweets.user_location.isna().all():
    cond_III_b = df_tweets.user_location
else:
    cond_III_b = df_tweets.user_location.fillna('').apply(unidecode).replace('', np.NaN).str.contains(
        pattern_locais_pt_lang_nonPT, na=np.NaN, flags=re.IGNORECASE).apply(lambda row: not row if pd.notnull(row) else row)
print("line 105 df_tweets.tweet_text.apply(pd.isnull).value_counts() :"+str(df_tweets.tweet_text.apply(pd.isnull).value_counts()))
df_tweets['tweet_place_country'] = df_tweets.tweet_place.apply(get_country_place_tweet)
print("line 107 df_tweets.tweet_text.apply(pd.isnull).value_counts() :"+str(df_tweets.tweet_text.apply(pd.isnull).value_counts()))
cond_III_c = df_tweets.tweet_place_country.apply(check_country_place_tweet).apply(
    lambda local: local in [1, 3] if pd.notnull(local) else local)

print("line 110 df_tweets.tweet_text.apply(pd.isnull).value_counts() :"+str(df_tweets.tweet_text.apply(pd.isnull).value_counts()))

cond_location = ((cond_III_a.fillna(True) != False) & (cond_III_b.fillna(True) != False) & (
            cond_III_c.fillna(True) != False)) ##Consider that if there is no information the result is considered

print("line 113 df_tweets.tweet_text.apply(pd.isnull).value_counts() :"+str(df_tweets.tweet_text.apply(pd.isnull).value_counts()))

df_tweets_final=df_tweets[(cond_location)&(cond_lang)&(cond_keywords)].reset_index(drop=True)

print("df_tweets_final.tweet_text.apply(pd.isnull).value_counts() :"+str(df_tweets_final.tweet_text.apply(pd.isnull).value_counts()))

save_locally(
    main_folder=main_folder_destination,
    df=df_tweets_final,
    url='tweets_filtered',
    newspaper='Twitter',
    csv_pickle='pickle'

)

