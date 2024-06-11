import asyncio
from twscrape import API, gather
from twscrape.logger import set_log_level
from utils_collection_urls import *
from unidecode import unidecode
import re
import numpy as np
import sqlite3

##Collect Tweetts

def delete_account(account_username: str, db_file_name:str):
    os.system(f"twscrape --db {db_file_name} del_accounts {account_username}")
    return "Completed"

def check_accounts(db_file_name:str):
    os.system(f'twscrape --db {db_file_name} accounts')
    return "Completed"

def add_account(file_name: str, file_format: str, db_file_name:str):
    os.system(f"twscrape --db {db_file_name} add_accounts {file_name} {file_format}")
    return "Completed"

def collect_info_accounts_db(db_file_name):
    conn = sqlite3.connect(db_file_name)
    cursor = conn.cursor()
    cursor.execute('SELECT username,active  FROM accounts')
    rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=['username', 'active'])
    return df

def check_status_accounts(db_file_name:str="Twitter_accounts.db", file_name:str = "account_credentials.txt", file_format: str = "username:password:email:email_password"):
    df = collect_info_accounts_db(db_file_name)
    while True:
        if df.active.isin([0]).all():
            [delete_account(account_username, db_file_name) for account_username in df.username.unique()]
            add_account(file_name, file_format, db_file_name)
            time.sleep(60)
            login_accounts(db_file_name)
            df = collect_info_accounts_db(db_file_name)
            print("check_status_accounts df :"+str(df))
        else:
            break
    return "Completed"

def login_accounts(db_file_name:str):
    os.system(f'twscrape --db {db_file_name} login_accounts')

def get_tweet_info(original_tweet_id,tweet_child):
    keys=["tweet_id", "user_id", "user_username", "tweet_text", "tweet_lang", "tweet_coordinates", "tweet_date", "user_location", "tweet_place", "tweet_hastags", "tweet_inreplytotweetid"]
    if tweet_child!=None:
        content=[tweet_child.id,
        tweet_child.user.id_str,
        tweet_child.user.username,
        tweet_child.rawContent,
        tweet_child.lang,
        tweet_child.coordinates,
        tweet_child.date,
        tweet_child.user.location,
        tweet_child.place,
        tweet_child.hashtags,
        tweet_child.inReplyToTweetId
        ]
        output =  dict(zip(keys,content))
    else:
        output = dict(zip(keys,[None for key in keys]))
    output['original_tweet_id']=original_tweet_id
    return output

async def collect_tweets(keyword: str, start_date: str, end_date: str, db_file_name: str):
    async def main(keyword: str, start_date: str, end_date: str):
        api = API(db_file_name)
        await api.pool.login_all()
        output = []
        q = f"{keyword} lang:pt since:{start_date} until:{end_date} "  # 2022-12-31 format
        async for tweet in api.search(q):
            output.append((tweet.id, tweet.user.username, tweet.rawContent,
                           tweet.user.id_str,tweet.lang,tweet.coordinates,
                          tweet.date,tweet.user.location,tweet.place,tweet.hashtags,tweet.inReplyToTweetId,
                          tweet.retweetedTweet,tweet.quotedTweet,start_date[:7], end_date[:7])) #[:7] not to include the day
            #output.append(tweet.json())
        return output

    # Here's the correction:
    output = await main(keyword, start_date, end_date)
    df_output = pd.DataFrame(output)

    return df_output

# Filter Tweets
def is_within_portugal_limits(latitude, longitude, PT_coordinates_limits):
    # Define the boundaries of Portugal

    output = []

    for type_PT_loc in PT_coordinates_limits.PT_loc.unique():
        aux = PT_coordinates_limits[PT_coordinates_limits.PT_loc == type_PT_loc]

        lat_min = aux[(aux.long_lat == 'lat') & (aux.min_max == 'min')].value.tolist()[0]
        long_min = aux[(aux.long_lat == 'long') & (aux.min_max == 'min')].value.tolist()[0]
        lat_max = aux[(aux.long_lat == 'lat') & (aux.min_max == 'max')].value.tolist()[0]
        long_max = aux[(aux.long_lat == 'long') & (aux.min_max == 'max')].value.tolist()[0]

        # Check if the coordinates fall within the boundaries
        if (lat_min <= latitude <= long_max) and (long_min <= longitude <= long_min):
            output.append(True)
        else:
            output.append(False)
    return any(output)


def check_country_coordinates(coordinates_obj, PT_coordinates_limits):
    if pd.isnull(coordinates_obj):
        output = np.NaN
    else:
        latitude_idx = coordinates_obj.index("latitude=-")
        longitude_idx = coordinates_obj.index("longitude=")

        long = float(coordinates_obj[longitude_idx + len("longitude="):latitude_idx].strip().strip(',').strip())

        lat = float(coordinates_obj[latitude_idx + len("latitude="):].strip(')').strip())

        output = is_within_portugal_limits(lat, long, PT_coordinates_limits)
    return output

def get_country_place_tweet(tweet_place):
    if pd.isnull(tweet_place):
        output = np.NaN
    else:
        try:
            left_idx=tweet_place.index("country='")
            right_idx=tweet_place[tweet_place.index("country='"):].index(',')
            output = tweet_place[left_idx+len("country='"):left_idx+right_idx].strip("'")
        except:
            output=np.NaN
    return output

def check_country_place_tweet(tweet_place_country):
    if pd.isnull(tweet_place_country):
        output = tweet_place_country
    elif tweet_place_country == 'Portugal':
        output = 1
    elif tweet_place_country in ["Brasil", "Angola", "Cabo Verde", "Guiné-Bissau", "Moçambique", "São Tomé e Príncipe",
                                 "Timor-Leste",'Brazil', 'Angola', 'Cape Verde', 'Guinea-Bissau', 'Mozambique', 'São Tomé and Príncipe', 'East Timor']:
        output = 2
    else:
        output = 3
    return output