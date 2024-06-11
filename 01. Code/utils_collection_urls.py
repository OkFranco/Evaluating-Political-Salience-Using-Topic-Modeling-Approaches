from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import WhitespaceTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import numpy as np
import os
import pandas as pd
from datetime import datetime, timedelta, timezone
from newspaper import Article
import json
from unidecode import unidecode
import re
import string
from nltk.corpus import stopwords
from itertools import chain
import requests
from bs4 import BeautifulSoup

"""
In this script we define all functions that are leveraged during the collection and treatment of news present in Arquivo.pt
The functions are divided considering the stages where they are used. Functions leveraged in different stages are stated  
"""


# XXX_urls_script
## Aux functions
def generate_date_range(year: int):
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    date_range = [
        start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)
    ]
    return date_range


def create_date_df(years: list):
    df_dates_aux = []
    for year in years:
        df_dates_aux.append(
            pd.DataFrame(generate_date_range(year=year), columns=["dates"])
        )

    df_dates = pd.concat(df_dates_aux, axis=0).reset_index(drop=True)

    df_dates_ymd = pd.concat(
        [
            df_dates.dates.dt.year.astype(str),
            df_dates.dates.dt.month.astype(str).apply(lambda val: val.zfill(2)),
            df_dates.dates.dt.day.astype(str).apply(lambda val: val.zfill(2)),
        ],
        axis=1,
    )

    df_dates_ymd.columns = ["year", "month", "day"]

    return df_dates_ymd


def create_directory(directory_path: str):
    # Check if the directory exists
    if not os.path.exists(directory_path):
        # Create the directory
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")


def url_definition(
    newspaper: str,
    url_aux: str,
    main: bool = False,
    year: str = "",
    month: str = "",
    day: str = "",
    publico_specific_date_inurl: bool = True,
):
    year = str(year)
    month = str(month)
    day = str(day)

    # The main and non main condition is for the situations where urls are collected differently
    if newspaper == "Publico":
        if main:
            if publico_specific_date_inurl:
                url_publico = f"{url_aux}/{year}/{month}"
                params_publico = {
                    "url": url_publico,  # "/{day}",
                    "output": "json",
                    "matchType": "prefix",
                    "filter=": "!~status:4&filter=!~status:5",
                    "status": "200",
                    # "limit":'100000',
                    "fields": "url,timestamp,mime,status",
                }
                url = f"https://arquivo.pt/wayback/cdx?url={params_publico['url']}&\
                          matchType={params_publico['matchType']}&\
                          output={params_publico['output']}&\
                          filter={params_publico['filter=']}&\
                          status={params_publico['status']}&\
                          fields={params_publico['fields']}"
            else:
                url_publico = f"{url_aux}"  # /{year}/{month}"
                params_publico = {
                    "url": url_publico,  # "/{day}",
                    "output": "json",
                    "matchType": "prefix",
                    "from": f"{year}{month}",  # {day}
                    "to": f"{year}{month}",  # {day}
                    "filter=": "!~status:4&filter=!~status:5",
                    "status": "200",
                    # "limit":'100000',
                    "fields": "url,timestamp,mime,status",
                }
                url = f"https://arquivo.pt/wayback/cdx?url={params_publico['url']}&\
                matchType={params_publico['matchType']}&\
                from={params_publico['from']}&\
                to={params_publico['to']}&\
                output={params_publico['output']}&\
                filter={params_publico['filter=']}&\
                status={params_publico['status']}&\
                fields={params_publico['fields']}"
        else:
            url_publico = url_aux
            params_publico = {
                "url": url_publico,  # "/{day}",
                "output": "json",
                "matchType": "prefix",
                "from": f"{year}{month}",  # {day}
                "to": f"{year}{month}",  # {day}
                "filter=": "!~status:4&filter=!~status:5",
                "status": "200",
                # "limit":'100000',
                "fields": "url,timestamp,mime,status",
            }
            url = f"https://arquivo.pt/wayback/cdx?url={params_publico['url']}&\
            from={params_publico['from']}&\
            to={params_publico['to']}&\
            matchType={params_publico['matchType']}&\
            output={params_publico['output']}&\
            filter={params_publico['filter=']}&\
            status={params_publico['status']}&\
            fields={params_publico['fields']}"

    elif newspaper == "Expresso":
        if main:
            url_expresso = url_aux
            params_expresso = {
                "url": url_expresso,
                "output": "json",
                "matchType": "prefix",
                "from": f"{year}{month}",
                "to": f"{year}{month}",
                # "filter": f"url:/{year}-{month}",  # "-{day}",
                "filter=": "!~status:4&filter=!~status:5",
                "status": "200",
                # "limit":'100000',
                "fields": "url,timestamp,mime,status",
            }

            url = f"https://arquivo.pt/wayback/cdx?url={params_expresso['url']}&\
                matchType={params_expresso['matchType']}&\
                from={params_expresso['from']}&\
                to={params_expresso['to']}&\
                output={params_expresso['output']}&\
                filter={params_expresso['filter=']}&\
                status={params_expresso['status']}&\
                fields={params_expresso['fields']}"
        else:
            # The only difference is the from to (here is only year)
            params_expresso = {
                "url": url_aux,
                "output": "json",
                "matchType": "prefix",
                "from": f"{year}",
                "to": f"{year}",
                # "filter": f"url:/{year}-{month}-{day}",
                "filter=": "!~status:4&filter=!~status:5",
                "status": "200",
                # "limit":'100000',
                "fields": "url,timestamp,mime,status",
            }

            url = f"https://arquivo.pt/wayback/cdx?url={params_expresso['url']}&\
                matchType={params_expresso['matchType']}&\
                output={params_expresso['output']}&\
                from={params_expresso['from']}&\
                to={params_expresso['to']}&\
                filter={params_expresso['filter=']}&\
                status={params_expresso['status']}&\
                fields={params_expresso['fields']}"

    elif newspaper == "CM":
        if main:
            url_cm = url_aux
            params_cm = {
                "url": url_cm,
                "output": "json",
                "matchType": "prefix",
                "from": f"{year}{month}",
                "to": f"{year}{month}",
                # "filter": f"url:detalhe",  # "/{year}-",
                "filter=": "!~status:4&filter=!~status:5",
                "status": "200",
                # "limit":'100000',
                "fields": "url,timestamp,mime,status",
            }

            url = f"https://arquivo.pt/wayback/cdx?url={params_cm['url']}&\
                            matchType={params_cm['matchType']}&\
                            output={params_cm['output']}&\
                            from={params_cm['from']}&\
                            to={params_cm['to']}&\
                            filter={params_cm['filter=']}&\
                            status={params_cm['status']}&\
                            fields={params_cm['fields']}"
        else:
            # The only difference is the from to (here is only year)
            url_cm = url_aux
            params_cm = {
                "url": url_cm,
                "output": "json",
                "matchType": "prefix",
                "from": f"{year}",
                "to": f"{year}",
                # "filter": f"url:detalhe",  # "/{year}-",
                "filter=": "!~status:4&filter=!~status:5",
                "status": "200",
                # "limit":'100000',
                "fields": "url,timestamp,mime,status",
            }

            url = f"https://arquivo.pt/wayback/cdx?url={params_cm['url']}&\
                                matchType={params_cm['matchType']}&\
                                output={params_cm['output']}&\
                                from={params_cm['from']}&\
                                to={params_cm['to']}&\
                                filter={params_cm['filter=']}&\
                                status={params_cm['status']}&\
                                fields={params_cm['fields']}"

    url_final = url.replace("  ", "")
    return url_final


def collect_news(url: str, path_log_file: str):
    timeout_error = False
    general_error=False
    try:
        # We define a timeout parameter because some requests do not encounter values in teh server and with this we speed up the process"
        response = requests.get(url, timeout=60)
    except requests.Timeout:
        # This will occur when in the first 5 seconds we were not able to start the request to the server
        # because there was not information for that url
        save_log(url, path_log_file)
        timeout_error = True
    except:
        # ConnectionError will occur when the requests was being made but the timeout restriction cutted it earlier
        # So we just provide unlimited time for the request to be made
        try:
            response = requests.get(url)
        except:
            # If there is an error in the previous step it means we surpassed the number of requests per minute/hour
            time.sleep(5)
            try:
                response = requests.get(url)
            except:
                general_error=True
    print("url :" + str(url))
    if timeout_error or general_error:
        df_news = pd.DataFrame(columns=["url", "timestamp", "mime", "status"])
    else:
        # json_objects_aux = response.text.strip()
        # json_objects = json_objects_aux.split('\n')
        # if json_objects == ['']:
        #    df_news = pd.DataFrame(columns=['url', 'timestamp', 'mime', 'status'])
        # else:
        #    json_data_list = [json.loads(obj) for obj in json_objects]
        #    df_news = pd.DataFrame(json_data_list, columns=['url', 'timestamp', 'mime', 'status'])

        try:
            json_objects = response.text.strip().split("\n")
            json_data_list = [
                json.loads(obj)
                for obj in json_objects
                if obj.strip() and obj is not None and pd.notnull(obj)
            ]
        except:
            aux_count = 1
            while True:
                time.sleep(5)
                response = requests.get(url)
                json_objects = response.text.strip().split("\n")
                print("error json_objects :" + str(json_objects))
                print("len error json_objects :" + str([len(i) for i in json_objects]))

                # Normally the code break with the following list ['<!DOCTYPE HTML PUBLIC "-//IETF//DTD HTML 2.0//EN">', '<html><head>', '<title>500 Internal Server Error</title>']
                #:['<!DOCTYPE HTML PUBLIC "-//IETF//DTD HTML 2.0//EN">', '<html><head>', '<title>429 Too Many Requests</title>',
                cond1 = (
                    len(
                        [
                            obj
                            for obj in json_objects
                            if "500 Internal Server Error" in obj
                        ]
                    )
                    == 0
                )
                cond2 = (
                    len([obj for obj in json_objects if "Too Many Requests" in obj])
                    == 0
                )
                if cond1 & cond2:
                    json_data_list = []
                    for obj in json_objects:
                        obj = (
                            obj.strip() if obj else None
                        )  # Strip whitespace from obj if it's not None
                        if obj and pd.notnull(obj):
                            try:
                                json_data = json.loads(obj)
                                json_data_list.append(json_data)
                            except json.JSONDecodeError:
                                # Handle JSON decoding errors (e.g., invalid JSON strings)
                                # Log the error or take appropriate action
                                pass

                    # json_data_list = [json.loads(obj) for obj in json_objects if
                    #             obj.strip() and obj is not None and pd.notnull(obj)]
                    break

                elif aux_count == 3:
                    json_data_list = []
                    break
                else:
                    aux_count += 1
        df_news = pd.DataFrame(
            json_data_list, columns=["url", "timestamp", "mime", "status"]
        )

    return df_news


def save_log(url, path_log_file):
    try:
        df_log_file = pd.read_csv(path_log_file, sep=";")
        df_log_file = df_log_file.loc[
            :, [col for col in df_log_file.columns if "Unnamed" not in col]
        ]
    except:
        df_log_file = pd.DataFrame(columns=["url", "date_error"])
    new_line = {"url": url, "date_error": datetime.now().strftime("%Y%m%d_%H%M%S")}
    df_log_file = pd.concat([df_log_file, pd.DataFrame([new_line])], ignore_index=True)

    df_log_file.to_csv(path_log_file, sep=";")
    return "Log concluded"


def identify_urls_not_consider(
    main_folder: str, newspaper: str, url: str, to_collect: bool
):
    # Needs to be the same way the directory to save on is defined in save_locally
    directory_path = f"{main_folder}/{newspaper.title()}/{replace_non_alphanumeric(input_string=url)}/"
    try:
        os.listdir(directory_path)
        if to_collect:
            output = ""
        else:
            output = url
    except FileNotFoundError:
        output = ""
    # If there is no error, means the directory does not exist and we do need to collect the url related information
    return output


def update_to_collect_status_update(to_collect_status_update: dict, path: str):
    def aux(to_collect_status_update: dict, row: pd.Series):
        try:
            # to_collect_status_update only has specif urls, so we need the try except to account for dominios not present
            output = to_collect_status_update[row["dominio"]]
        except:
            output = row["to_collect"]

        return output

    # We need to re read the file, in case the script from the other newspaper has run in parallel and update the file meanwhile
    try:
        domains = pd.read_csv(path, sep=";")
    except:
        print("path (update_to_collect_status_update) :" + str(path))
        domains = pd.read_csv(path, sep=";")
    domains["to_collect"] = domains.apply(
        lambda row: aux(row=row, to_collect_status_update=to_collect_status_update),
        axis=1,
    )
    domains = domains.drop([col for col in domains if "Unnamed" in col], axis=1)
    domains.to_csv(path, sep=";")
    return "Updated"


def collect_all_news_urls(
    years: list,
    domains_to_consider: pd.DataFrame,
    newspaper: str,
    collect_urls_per_year_yearmonth: str,
    save_per_date: bool = True,
    publico_specific_date_inurl: bool = True,
    # In the case of Publico we want to collect main urls through 2 forms.
    main_folder: str = "all_news_final",
    path_aux_file: str = "",  # will provide error
    path_log_file: str = "logs.csv",
):
    # Replace 2023 with the desired year
    df_dates_ymd = create_date_df(years=years)

    if collect_urls_per_year_yearmonth == "y":
        df_dates_ymd_to_consider = df_dates_ymd.drop_duplicates(subset=["year"])
    elif collect_urls_per_year_yearmonth == "ym":
        df_dates_ymd_to_consider = df_dates_ymd.drop_duplicates(
            subset=["year", "month"]
        )

    to_collect_status_update = {}

    for idx, row in df_dates_ymd_to_consider.iterrows():
        url_aux = row["dominio"]
        main = row["main"]
        df_news_newspaper_list = []

        print(url_aux)

        for idx, row in df_dates_ymd_to_consider.iterrows():
            year = row["year"]
            month = row["month"]
            day = row["day"]
            url = url_definition(
                newspaper=newspaper,
                url_aux=url_aux,
                main=main,
                year=year,
                month=month,
                day=day,
                publico_specific_date_inurl=publico_specific_date_inurl,
            )

            df_news_newspaper_aux = collect_news(url=url, path_log_file=path_log_file)

            if save_per_date:
                df_to_save = df_news_newspaper_aux

                df_to_save["newspaper"] = newspaper.title()
                df_to_save["domain_main"] = replace_non_alphanumeric(
                    input_string=url_aux
                )
                if df_to_save.shape[0] == 0:
                    df_to_save["ID"] = pd.Series(dtype="object")
                else:
                    df_to_save["ID"] = df_to_save.reset_index().apply(
                        lambda row: f"{row['newspaper']}_{row['domain_main']}_{row['index']}_{row['timestamp']}".lower(),
                        axis=1,
                    )

                save_locally(
                    main_folder=main_folder,
                    df=df_to_save,
                    url=url_aux,
                    newspaper=newspaper,
                    start_year=year,
                    start_month=month,
                    end_year="",
                    end_month="",
                )
                to_collect_status_update[url_aux] = False
            else:
                df_news_newspaper_list.append(df_news_newspaper_aux)

        if save_per_date:
            continue
        else:
            df_to_save = pd.concat(df_news_newspaper_list, axis=0).reset_index(
                drop=True
            )

            df_to_save["newspaper"] = newspaper.title()
            df_to_save["domain_main"] = replace_non_alphanumeric(input_string=url_aux)
            if df_to_save.shape[0] == 0:
                df_to_save["ID"] = pd.Series(dtype="object")
            else:
                df_to_save["ID"] = df_to_save.reset_index().apply(
                    lambda row: f"{row['newspaper']}_{row['domain_main']}_{row['index']}_{row['timestamp']}".lower(),
                    axis=1,
                )

            save_locally(
                main_folder=main_folder,
                df=df_to_save,
                url=url_aux,
                newspaper=newspaper,
                start_year=year,
                start_month=month,
                end_year="",
                end_month="",
            )
            to_collect_status_update[url_aux] = False

        # We run update_to_collect_status_update after each link in case the code breaks meanwhile
        update_to_collect_status_update(
            to_collect_status_update=to_collect_status_update, path=path_aux_file
        )

    return to_collect_status_update


# Collect_news_text
def encode_url(tstamp: str, url_original: str, aux_file: pd.DataFrame, type_url: str):
    url_coded = []
    for i in url_original:
        try:
            url_coded.append(
                aux_file.set_index(list(aux_file.columns)[1]).to_dict()["coded char"][i]
            )
        except:
            url_coded.append(i)
    url_coded = "".join(url_coded)
    if type_url == "noFrame":
        return f"https://arquivo.pt/noFrame/replay/{tstamp}id_/{url_coded}"
    elif type_url == "textextracted":
        return f"https://arquivo.pt/textextracted?m={url_coded}%2F{tstamp}"


### Collect_text appraoch #1 - collect text using the noFrame url but with newspaper library https://newspaper.readthedocs.io/en/latest/ that has revealed itself effective after manual validation
def collect_text_approach1(row: pd.Series, aux_file: pd.DataFrame):
    tstamp = row.loc["timestamp"]
    url_original = row.loc["url"]
    url = encode_url(tstamp, url_original, aux_file, "noFrame")
    try:
        article = Article(url, language="pt")

        article.download()
        article.parse()
        # Sometimes the meta_description is part of the main text. Otherwise is a resume that corresponds to the lead (see https://www.publico.pt/2024/02/16/politica/noticia/invasao-privacidade-filhos-rui-tavares-dominou-debate-ventura-2080658)
        full_text = (
            ". ".join(
                [article.title]
                + [
                    summary
                    for summary in [article.meta_description]
                    if summary not in article.text
                ]
                + [article.text]
            )
            .strip(".")
            .strip()
        )

        date = article.publish_date
        if date!=None:
            date=date.date()
    except requests.exceptions.ConnectionError as e:
        full_text = "ConnectionError"
    except:
        full_text = "CHECK URL"
        date = np.NaN

    return url, url_original, full_text, date


### Collect_text approach #2 - collect text from textextracted endpoint
def collect_text_approach2(row: pd.Series, aux_file: pd.DataFrame):
    tstamp = row.loc["timestamp"]
    url_original = row.loc["url"]

    url = encode_url(tstamp, url_original, aux_file, "textextracted")

    max_attempts = 2  # Maximum number of attempts
    attempt = 0  # Initialize attempt counter

    while True:
        try:
            response = requests.get(url)
            full_text = response.text
            if "500 Internal Server Error" not in full_text and "Too Many Requests" not in full_text:
                break
            else:
                time.sleep(5)
        except requests.exceptions.ReadTimeout:
            time.sleep(5)
        except UnicodeDecodeError:
            full_text="Error"
        except requests.exceptions.ConnectionError as e:
            if attempt < max_attempts:  # Check if maximum attempts not reached
                attempt += 1  # Increment attempt counter
                time.sleep(5)  # Wait for 5 minutes
            else:
                full_text = "No article in textextracted"
                break


    if (
        ("" == full_text)
        or ("Resource ID doesn't exist" in full_text)
        or ("500 Internal Server Error" in full_text)
        or ("Too Many Requests" in full_text)
        or ("Moved Permanently Moved Permanently" in full_text)
        or ("Object moved Object moved here" in full_text)
    ):
        full_text = "No article in textextracted"

    date = np.NaN  # The API does not return the newsarticle publish_date

    return url, url_original, full_text, date


### Collect_text approach #3 - collect text from html ('noFrame')
def collect_text_approach3(row: pd.Series, aux_file: pd.DataFrame):
    tstamp = row.loc["timestamp"]
    url_original = row.loc["url"]

    url = encode_url(tstamp, url_original, aux_file, "noFrame")
    print("url :" + str(url))

    max_attempts = 2  # Maximum number of attempts
    attempt = 0  # Initialize attempt counter

    while True:
        try:
            response = requests.get(url)
            full_text = response.text
            if "500 Internal Server Error" not in full_text and "Too Many Requests" not in full_text:
                break
            else:
                time.sleep(5)
        except requests.exceptions.ReadTimeout:
            time.sleep(5)
        except UnicodeDecodeError:
            full_text="Error"
        except requests.exceptions.ConnectionError as e:
            if attempt < max_attempts:  # Check if maximum attempts not reached
                attempt += 1  # Increment attempt counter
                time.sleep(5)  # Wait for 5 minutes
            else:
                full_text = "Error"
                break

    if ("" == full_text) or ("Error" in full_text) or ("default backend" in full_text) or ("A página procurada não foi encontrada no Arquivo.pt" in full_text) or ("Página não encontrada (404) - Arquivo.pt" in full_text):
        full_text = "No article in noFrame"

    date = np.NaN  # The API does not return the newsarticle publish_date

    return url, url_original, full_text, date


# Main functions to collect urls

def collect_text(row: pd.Series, aux_file: pd.DataFrame):
    tstamp = row.loc["timestamp"]

    print(row['url'])

    # First we try to collect the text directly from the API
    url, url_original, full_text_approach1, date_approach1 = collect_text_approach1(row, aux_file)
    url, url_original, full_text_approach2, date_approach2 = collect_text_approach2(row, aux_file)
    url, url_original, full_text_approach3, date_approach3 = collect_text_approach3(row, aux_file)



    return (
        url,
        url_original,
        #full_text_final,
        full_text_approach1,
        full_text_approach2,
        full_text_approach3,
        row.loc["ID"],
        tstamp,
        date_approach1,
    )


def collect_date_approach3(full_text_approach3, url):
    """We prefered to stop after approach is found, having prioritized approaches based on their likehood to
    reflect the true publishing date. We than implement a manual validation to check the correct date.
    If collecting all identified dates we would potentially collect dates that are not important to us"""

    published_date_values_datetime = [datetime(1900, 1, 1, 0, 0, 0, 0, tzinfo=timezone.utc)]

    try:
        soup = BeautifulSoup(full_text_approach3, 'html.parser')

        # 1st approach - collect via find_all
        approaches = {'meta': [{'property': 'impresa:publishedDate'}, {'property': 'article:published_time'},
                               {"id": "ctl00_GsaDate"}, {"property": "article:published_time"}],
                      'time': [{'datetime': True}, {'class': 'timestamp value'}, {'class': 'time published'},
                               {'class': 'entry-date'}],
                      'p': [{'class': 'date'}, {'class': 'timeStamp publishedDate'}],
                      'p ': [{'datetime': True}],
                      'span': [{'class': 'date'}, {'class': 'data'}, {'class': 'post-credits'},
                               {'class': 'NoticiaEsquerdaFotoTxtBold'}, {"class": "minifontred"},
                               {"id": "ctl00_ctl00_weekDate"}, {"id": "ctl00_ContentPlaceHolder1_data"},
                               {"id": "ctl00_ContentPlaceHolder1_DataAutor"},
                               {"id": "ctl00_ContentPlaceHolder1_txtData"}],
                      'div': [{'id': 'hora'}, {'class': 'data'}, {'class': "act_time arial"},
                              {"class": "textoPequeno", "style": "padding-bottom:5px; color:#999999"},
                              {"class": "date"}],
                      'td': [{'class': 'DIGITAL_ArialAzul_11CAPS'}],
                      'a': [{'class': 'subheaderTitle centered-element'}]}

        published_dates = []

        datetime_formats = ['%a, %d %b %Y %H:%M:%S %Z', "%d.%m.%Y %H:%M", "%d.%m.%Y", "%d/%m/%Y - %H:%M",
                            "%A, %d %B %Y %I:%M %p", "%d %b %Y", "%Y-%m-%d %H:%M:%S", "%a, %d %b %Y",
                            "%d/%m/%Y", "%Y-%m-%d", "%d/%m/%y - %H:%M", "%B %d, %Y %I:%M %p", "%b %d, %Y %I:%M %p",
                            "%d %B %Y", "%d %B %Y - %H:%M", "%d %b %Y - %I:%M %p", "%d/%m/%Y %H:%M:%S",
                            "%d-%b-%Y %H:%M", "%A, %d %b %y %I:%M %p", '%d de %B de %Y, %H:%M',
                            '%a, %d %b %Y %H:%M:%S %Z', "%d.%m.%Y - %H:%M", "%d de %B de %Y", '%d-%m-%Y %H:%M:%S',
                            '%d de %B de %Y às %H:%M', "%d.%m.%Y %H%M", "%Y%m%d", '%Y-%m-%dT%H:%M:%S.%fZ',
                            '%Y-%m-%d%H:%M:%S.%f',
                            '%d/%m/%y', '%d/%m/%Y %H:%M', '%d/%m/%Y%H:%M', '%d.%m.%Y - %H%M']

        portuguese_months = {'janeiro': 'January', 'fevereiro': 'February', 'março': 'March', 'abril': 'April',
                             'maio': 'May', 'junho': 'June', 'julho': 'July', 'agosto': 'August',
                             'setembro': 'September', 'outubro': 'October', 'novembro': 'November',
                             'dezembro': 'December'}

        outer_break_cond = False

        for sub_approach_key, sub_approaches in approaches.items():
            for sub_approach_group in sub_approaches:

                if (sub_approach_key == 'td') & ('DIGITAL_ArialAzul_11CAPS' in list(sub_approach_group.values())):
                    if ('digital.publico.clix' not in url):
                        continue

                key = list(sub_approach_group.keys())[0]
                published_dates = soup.find_all(sub_approach_key.strip(), {key: sub_approach_group[key]})
                published_dates = [i for i in published_dates if i != None]
                if len(published_dates) > 0:

                    # Extract content from the meta tags
                    if sub_approach_group[key] == True:
                        published_date_values = [meta[key] for meta in published_dates]
                    elif sub_approach_key in ['p', 'time', 'span', 'div', 'td', 'a']:
                        published_date_values = [meta.string for meta in published_dates]
                    elif sub_approach_key in ['p ', 'meta']:
                        published_date_values = [meta['content'] for meta in published_dates]

                    published_date_values = [i for i in published_date_values if i != None]
                    try:
                        try:
                            published_date_values_datetime = [
                                datetime.fromisoformat(clean_time(date_string).replace('Z', '+00:00')) for
                                date_string in published_date_values]
                        except:
                            published_date_values_datetime = [
                                datetime.fromisoformat(date_string.replace('Z', '+00:00')) for
                                date_string in published_date_values]

                        if len(published_date_values_datetime) == 0:
                            published_date_values_datetime = [datetime(1900, 1, 1, 0, 0, 0, 0, tzinfo=timezone.utc)]
                        else:
                            break
                        outer_break_cond = True
                        break



                    except:

                        published_date_values_datetime = []

                        for format_str in datetime_formats:
                            for date_string in published_date_values:
                                if 'b' in format_str or 'B' in format_str:
                                    date_string = ' '.join(
                                        [portuguese_months.get(word.lower(), word.lower()) for word in
                                         date_string.split()])
                                    cleaned_string = date_string

                                else:
                                    cleaned_string = clean_time(date_string)
                                try:
                                    # Try to convert each date string using the current format

                                    published_date_values_datetime.append(
                                        datetime.strptime(cleaned_string, format_str).replace(tzinfo=timezone.utc))
                                    outer_break_cond = True
                                    break
                                except ValueError:
                                    continue
                            if outer_break_cond:
                                break

                        if len(published_date_values_datetime) == 0:
                            published_date_values_datetime = [datetime(1900, 1, 1, 0, 0, 0, 0, tzinfo=timezone.utc)]
                        else:
                            break
            if outer_break_cond:
                break

        # 2nd approach - collect via find in text

        strings_to_check = ['article_publication_date']  # Add more strings as needed

        if len(published_dates) == 0:
            for string_ in strings_to_check:
                scripts_with_publication_date = [i for i in soup.find_all('script') if
                                                 i.string is not None and string_ in i.string]
                if len(scripts_with_publication_date) != 0:
                    pattern = r'(?<={}\s")[^"]+'.format(string_)
                    for script in scripts_with_publication_date:
                        published_date_values_datetime = re.findall(pattern, script.string)
                        if published_date_values_datetime:
                            published_date_values_datetime = [
                                datetime.strptime(date_string, '%a, %d %b %Y %H:%M:%S %Z').replace(tzinfo=timezone.utc)
                                for date_string in published_date_values_datetime]

    except:
        published_date_values_datetime = [datetime(1000, 1, 1, 0, 0, 0, 0, tzinfo=timezone.utc)]

    published_date_values_datetime = [date.date() for date in published_date_values_datetime]

    return published_date_values_datetime


#Extensive text cleaning and remove stopwords

def recursively_replace_double_spaces(text):
    if "  " in text:
        return recursively_replace_double_spaces(
            text.replace("  ", " ")
        )

    return text

def extensive_text_cleaning(text):
    chars_to_remove = ['\uf020', '--', '---', '–»', '––', '–––', '‘', '‘»', '‘’', '’', '’»', '’’', '’’’', '’’’’', '‚',
                       '™', '∋', '~', '¢', '§', '®', '°', '±', 'µ', '·', 'º', '¿', 'ø', 'ˆ', '̀', '́', '̃', '̧', 'α',
                       'β', 'ε', 'μ', 'һ', '֊', '‖', '‗', '„', '‡', '•', '…', '‰', '›', '‼', '€', '∨', '≻', '⎯', '⩽',
                       '�', '—', '«', 'ª', '”', 'æ', '¾', '½', '»', '“', '£', '\t', '\n', '\xa0', '+']

    # convert to lowercase
    text = text.lower()

    # remove weird chars
    pattern = "|".join(re.escape(char) for char in chars_to_remove)
    text = re.sub(pattern, ' ', text)

    # clean text ad hoc
    # key problems: there are 50k interventions with primeiro solo and 50k with primeiro-ministro. removing the -
    # would provide too much "power" in primeiro. so we cannot remove it. the same applies to
    # Centro Democrático Social - Partido popular with
    # not removing the dash we will lead to cases where the dash maintains when it should not. these are minimal
    # cases in relative terms with the problem of primeiro-ministro (~100 observations per example)
    # the problem with primeiro-ministro only occurs when properly written (wrongly is like primeiro - ministro). the
    # wrong cases account to ~100 interventions.

    # therefore we only maintain the - when there is no space around them.
    # when there is space around the - the wrong cases overpower and so we should remove them.
    # me, te, etc (ex.: dá-me) are remove in stopwords.
    text = re.sub(r'\s-\s|-\s|\s-', ' ', text)

    # unicode
    # This is used because frequently the characters are poorly recognized by the machine.
    # This impacts normal portuguese words as well (ex.: opção -> opcao). We do not see a problem with that, except
    # potentially for lemmatization. Let's see
    text = unidecode(text)

    # remove punctuation
    pattern_punkt = r'[' + re.escape(string.punctuation.replace('-', '')) + ']'
    text = re.sub(pattern_punkt, " ", text)

    # remove numbers
    text = re.sub('[0-9]', ' ', text)

    # repeat here
    text = re.sub(r'\s-\s|-\s|\s-', ' ', text)

    # remove single letters between spaces:
    # 1) these account for regular stopwords in the portuguese language (e,a,o)
    # 2) these account for situations where after cleaning the text regular stopwords appear but are not relevant
    # decreto-lei n.º5 65/V-
    text = re.sub('(?:^|\s)[a-z](?=\s|$)', ' ', text.lower())

    # Remove extra spaces
    text = recursively_replace_double_spaces(text)

    return text.lower().strip()

def define_stopwords():
    #### identify stopwords to remove
    portuguese_stopwords_I = stopwords.words('portuguese')

    url = "https://www.ranks.nl/stopwords/portuguese"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    portuguese_stopwords_II = [td.text.split() for td in soup.find_all('td')]

    pronouns = ["me", "te", "vos", "os", "as", "o", "a", "lhe les", "ta", "to", "ma",
                "mo", "nos", "la", "los", "las", "lo", "na", "no", "nas", "nos"]

    aux_portuguese_stopwords_final = list((portuguese_stopwords_II, [portuguese_stopwords_I]))

    aux2_portuguese_stopwords_final = list(chain.from_iterable(aux_portuguese_stopwords_final))

    aux_3portuguese_stopwords_final = list(chain.from_iterable(aux2_portuguese_stopwords_final)) + pronouns

    aux_4portuguese_stopwords_final = list(set(aux_3portuguese_stopwords_final))

    portuguese_stopwords_final = list(
        set(list(chain.from_iterable([i.split() for i in aux_4portuguese_stopwords_final]))))
    return portuguese_stopwords_final

def remove_stop_words(text, stopwords):
    chars_to_remove = ['--', '---', '----']
    # Get the list of Portuguese stop words
    pattern = r'\b(?:{})\b'.format('|'.join(map(re.escape, stopwords)))
    text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # remove weird chars
    # after removing - words with multiple - can have more than 1 (ex.: dar-lhe-ei -> dar--ei)
    pattern = "|".join(re.escape(char) for char in chars_to_remove)
    text = re.sub(pattern, ' ', text)

    # repeat here
    # When removing the pronouns in verbs (ex.: dar-lhe) now be have dar- and we need to remove the -
    while re.findall(r'\s-\s|-\s|\s-', text):
        text = re.sub(r'\s-\s|-\s|\s-', ' ', text)

    # When removing pronounts and - in text we can be left out with single letters (ex.: d-a)
    text = re.sub('(?:^|\s)[a-z](?=\s|$)', ' ', text.lower())

    # Remove extra spaces
    text = recursively_replace_double_spaces(text)

    return text.strip()


#Urls_news Filtering I
def assign_full_text(row):
    if row['cond_approach1']==False:
        output=row['full_text_approach1']
    elif row['cond_approach2']==False:
        output=row['full_text_approach2']
    else:
        output=np.NaN
    return output


# Assign date variable
def clean_time(input_string):
    output=re.sub(r'[^\d.,:/ -]', '', input_string)

    chars_to_strip=',.- '
    while True:
        if len(output)==len(output.strip(chars_to_strip)):
            break
        else:
            output=output.strip(chars_to_strip)
    return output.replace('  ','')

def collect_date_approach3(full_text_approach3, url):
    """We prefered to stop after approach is found, having prioritized approaches based on their likehood to
    reflect the true publishing date. We than implement a manual validation to check the correct date.
    If collecting all identified dates we would potentially collect dates that are not important to us"""

    published_date_values_datetime = [datetime(1900, 1, 1, 0, 0, 0, 0, tzinfo=timezone.utc)]

    try:
        soup = BeautifulSoup(full_text_approach3, 'html.parser')

        # 1st approach - collect via find_all
        approaches = {'meta': [{'property': 'impresa:publishedDate'}, {'property': 'article:published_time'},
                               {"id": "ctl00_GsaDate"}],
                      'time': [{'datetime': True}, {'class': 'timestamp value'}, {'class': 'time published'},
                               {'class': 'entry-date'}],
                      'p': [{'class': 'date'}, {'class': 'timeStamp publishedDate'}],
                      'p ': [{'datetime': True}],
                      'span': [{'class': 'date'}, {'class': 'data'}, {'class': 'post-credits'},
                               {'class': 'NoticiaEsquerdaFotoTxtBold'}],
                      'div': [{'id': 'hora'}, {'class': 'data'}, {'class': "act_time arial"}],
                      'td': [{'class': 'DIGITAL_ArialAzul_11CAPS'}],
                      'a': [{'class': 'subheaderTitle centered-element'}]}

        published_dates = []

        datetime_formats = ['%a, %d %b %Y %H:%M:%S %Z', "%d.%m.%Y %H:%M", "%d.%m.%Y", "%d/%m/%Y - %H:%M",
                            "%A, %d %B %Y %I:%M %p", "%d %b %Y", "%Y-%m-%d %H:%M:%S", "%a, %d %b %Y",
                            "%d/%m/%Y", "%Y-%m-%d", "%d/%m/%y - %H:%M", "%B %d, %Y %I:%M %p", "%b %d, %Y %I:%M %p",
                            "%d %B %Y", "%d %B %Y - %H:%M", "%d %b %Y - %I:%M %p", "%d/%m/%Y %H:%M:%S",
                            "%d-%b-%Y %H:%M", "%A, %d %b %y %I:%M %p", '%d de %B de %Y, %H:%M',
                            '%a, %d %b %Y %H:%M:%S %Z', "%d.%m.%Y - %H:%M", "%d de %B de %Y", '%d-%m-%Y %H:%M:%S',
                            '%d de %B de %Y às %H:%M', "%d.%m.%Y %H%M", "%Y%m%d", '%Y-%m-%dT%H:%M:%S.%fZ',
                            '%Y-%m-%d%H:%M:%S.%f',
                            '%d/%m/%y', '%d/%m/%Y %H:%M', '%d/%m/%Y%H:%M']

        portuguese_months = {'janeiro': 'January', 'fevereiro': 'February', 'março': 'March', 'abril': 'April',
                             'maio': 'May', 'junho': 'June', 'julho': 'July', 'agosto': 'August',
                             'setembro': 'September', 'outubro': 'October', 'novembro': 'November',
                             'dezembro': 'December'}

        outer_break_cond = False

        for sub_approach_key, sub_approaches in approaches.items():
            for sub_approach_group in sub_approaches:

                if (sub_approach_key == 'td') & ('DIGITAL_ArialAzul_11CAPS' in list(sub_approach_group.values())):
                    if ('digital.publico.clix' not in url):
                        continue

                key = list(sub_approach_group.keys())[0]
                published_dates = soup.find_all(sub_approach_key.strip(), {key: sub_approach_group[key]})

                published_dates = [i for i in published_dates if i != None]
                if len(published_dates) > 0:
                    # Extract content from the meta tags
                    if sub_approach_group[key] == True:
                        published_date_values = [meta[key] for meta in published_dates]
                    elif sub_approach_key in ['p', 'time', 'span', 'div', 'td', 'a']:
                        published_date_values = [meta.string for meta in published_dates]
                    elif sub_approach_key in ['p ', 'meta']:
                        published_date_values = [meta['content'] for meta in published_dates]

                    published_date_values = [i for i in published_date_values if i != None]

                    try:
                        published_date_values_datetime = [
                            datetime.fromisoformat(clean_time(date_string).replace('Z', '+00:00')) for
                            date_string in published_date_values]
                        if len(published_date_values_datetime) == 0:
                            published_date_values_datetime = [datetime(1900, 1, 1, 0, 0, 0, 0, tzinfo=timezone.utc)]
                        else:
                            break
                        outer_break_cond = True
                        break



                    except:

                        published_date_values_datetime = []

                        for format_str in datetime_formats:
                            for date_string in published_date_values:
                                if 'b' in format_str or 'B' in format_str:
                                    date_string = ' '.join(
                                        [portuguese_months.get(word.lower(), word.lower()) for word in
                                         date_string.split()])
                                    cleaned_string = date_string

                                else:
                                    cleaned_string = clean_time(date_string)

                                try:
                                    # Try to convert each date string using the current format
                                    published_date_values_datetime.append(
                                        datetime.strptime(cleaned_string, format_str).replace(tzinfo=timezone.utc))
                                    outer_break_cond = True
                                    break
                                except ValueError:
                                    continue
                            if outer_break_cond:
                                break

                        if len(published_date_values_datetime) == 0:
                            published_date_values_datetime = [datetime(1900, 1, 1, 0, 0, 0, 0, tzinfo=timezone.utc)]
                        else:
                            break
            if outer_break_cond:
                break

        # 2nd approach - collect via find in text

        strings_to_check = ['article_publication_date']  # Add more strings as needed

        if len(published_dates) == 0:
            for string_ in strings_to_check:
                scripts_with_publication_date = [i for i in soup.find_all('script') if
                                                 i.string is not None and string_ in i.string]
                if len(scripts_with_publication_date) != 0:
                    pattern = r'(?<={}\s")[^"]+'.format(string_)
                    for script in scripts_with_publication_date:
                        published_date_values_datetime = re.findall(pattern, script.string)
                        if published_date_values_datetime:
                            published_date_values_datetime = [
                                datetime.strptime(date_string, '%a, %d %b %Y %H:%M:%S %Z').replace(tzinfo=timezone.utc)
                                for date_string in published_date_values_datetime]

    except:
        published_date_values_datetime = [datetime(1000, 1, 1, 0, 0, 0, 0, tzinfo=timezone.utc)]

    published_date_values_datetime = [date.date() for date in published_date_values_datetime]

    return published_date_values_datetime

def define_final_date(row):
    if len(row['date_published_inferred'])==1:
        output_aux=row['date_published_inferred']
    else:
        output_aux=[row['date_published']]

    if '1900' in str(output_aux):
        output=[datetime.strptime(str(row['timestamp'])[:8], '%Y%m%d').date()]
    else:
        output=output_aux
    return output

# Filter results based on collection similarity (filtering II)
def prioritize_idx_based_on_date(selected_idx_aux: tuple, df_file_1_filtered: pd.DataFrame):
    if len(selected_idx_aux) != 1:
        with_date = [idx for idx in selected_idx_aux if pd.notnull(df_file_1_filtered.loc[idx]['date_final'])]
        if len(with_date) == 1:
            # Prefer the one with date
            output = with_date[0]
        elif len(with_date) > 1:
            # If multiple with date, prefer the shortest (as they are already very similar texts)
            new_preferred = df_file_1_filtered.nr_words.argsort().tolist()[0]#full_text.str.len().argsort().tolist()[0]
            output = with_date[new_preferred]
        elif len(with_date) == 0:
            # If with no date. prefer the shorter file
            try:
                new_preferred = df_file_1_filtered.nr_words.argsort().tolist()[0]#.loc[list(selected_idx_aux)].full_text.str.len().argsort().tolist()[0]
            except:
                print("df_file_1_filtered.nr_words :"+str(df_file_1_filtered.nr_words))
                print("selected_idx_aux :"+str(selected_idx_aux))
                new_preferred = df_file_1_filtered.nr_words.argsort().tolist()[0]
            output = selected_idx_aux[new_preferred]
    elif len(selected_idx_aux) == 1:
        output = selected_idx_aux[0]
    return output


def calculate_collection_similarity(df_file_1_filtered:pd.DataFrame,threshold_collection_similarity:float,idx_combinations_to_compare:list,file_path:str,domain: str, directory_child_all_idx_combinations:str, directory_child_idx_to_keep:str,vectorizer):

    """ Collection similarity is based on paper "Measuring News Similarity Across Ten U.S. News Sites"
    accessible via https://arxiv.org/pdf/1806.09082.pdf"""



    # idx_to_keep_final_aux=set()
    idx_to_keep_final_aux = list() #FINALOPTIONS

    for combo in idx_combinations_to_compare:
        # results={}
        df_aux=df_file_1_filtered.loc[list(combo)]

        matrix_size=len(df_aux) #to fill

        N = np.ones((matrix_size, matrix_size), dtype=int)
        np.fill_diagonal(N, 0)
        O = np.ones((matrix_size, matrix_size), dtype=int)
        D_aux = vectorizer.fit_transform(df_aux.full_text_final_cleaned_nostopwords.tolist())
        D=cosine_similarity(D_aux, dense_output=False)

        num_aux=np.dot(N, D.toarray())
        den_aux=np.dot(N, O)

        # combo_min=combo[0]
        # combo_max=combo[1]
        num=np.linalg.norm(num_aux, ord='fro')
        den=np.linalg.norm(den_aux, ord='fro')

        result=num/den

        if result>=threshold_collection_similarity:
        # results[(combo_min,combo_max)]=result
        # results_bool = {key:val>=results_threshold for key, val in results.items()}

        # """The next part of the code could be simplified, but due to lack of time, it was not"""
        #TOREMOVE OLD
        # df_results_A=pd.DataFrame(index=list(results_bool.keys()),data=list(results_bool.values()),columns=['connection']).reset_index()
        # print("df_results_A :"+str(df_results_A)) #TOREMOVE to try to simplify the code afterwards
        # df_results_A.columns=['group_1','group_2','connection']
        #
        # df_results_B=df_results_A.copy()
        # df_results_B['group_1_aux']=df_results_B.group_1.copy()
        # df_results_B['group_1']=df_results_B.group_2.copy()
        # df_results_B['group_2']=df_results_B.group_1_aux.copy()
        # df_results_B=df_results_B.drop('group_1_aux',axis=1)
        #
        # df_results_total=pd.concat([df_results_A,df_results_B],axis=0)
        #
        # while True:
        #     df_results_total_aux=df_results_total.groupby('group_1')['connection'].apply(lambda val: (val==True).sum())
        #     max_val=df_results_total_aux.max()
        #     if max_val==0:
        #         break
        #     else:
        #         selected_idx_aux=df_results_total_aux[df_results_total_aux==max_val].index
        #         selected_idx=prioritize_idx_based_on_date(selected_idx_aux,df_aux)
        #         df_results_total=df_results_total[df_results_total.group_1==selected_idx]
        #
        #         if df_results_total.shape[0]==1:
        #             break
        # idx_to_keep_aux=list(df_results_total.group_1.unique())
        # idx_to_keep=df_aux.loc[idx_to_keep_aux].index
            selected_idx = [prioritize_idx_based_on_date(selected_idx_aux= combo, df_file_1_filtered= df_aux)]
        else:
            selected_idx = combo

        # idx_to_keep_final_aux.union(set(selected_idx))
        idx_to_keep_final_aux += [el for el in selected_idx] #FINALOPTIONS

    ### if an idx is selected, but is not selected in all pairwise comparisons the remaining selections should not be considered, as if it fails to be selected
    # once it means that there is another text that should be preferred, therefore the number of times that an el is selected needs to be equal to the number
    # of combos that appears in
    idx_combinations_to_compare_list=[el for tup in idx_combinations_to_compare for el in tup]
    frequency_elements = {el:idx_combinations_to_compare_list.count(el) for el in idx_combinations_to_compare_list}
    frequency_elements_to_keep = {el:idx_to_keep_final_aux.count(el) for el in idx_to_keep_final_aux}

    idx_to_keep_final=[el for el in idx_to_keep_final_aux if frequency_elements[el]==frequency_elements_to_keep[el]]

    df_idx_to_keep = pd.DataFrame(list(idx_to_keep_final))
    df_all_idx_combinations = pd.DataFrame(set(idx_combinations_to_compare))
    save_locally(
        main_folder=file_path, df=df_idx_to_keep, url=f"{domain}", newspaper=directory_child_idx_to_keep,
        csv_pickle='pickle'
    )

    save_locally(
        main_folder=file_path, df=df_all_idx_combinations, url=f"{domain}", newspaper=directory_child_all_idx_combinations,
        csv_pickle='pickle'
    )

    return "Completed"

def find_idxs_to_compare(df_file_1_filtered: pd.DataFrame,threshold_rel_diff_nr_words: float, start_number: int, end_number: int, file_path: str,threshold_collection_similarity: float,directory_child_all_idx_combinations:str, directory_child_idx_to_keep: str):
    idx_combinations_to_compare=[]

    vectorizer = TfidfVectorizer(tokenizer=WhitespaceTokenizer().tokenize, lowercase=False, ngram_range=(1, 1))

    options=list(df_file_1_filtered.domain_main.unique())
    for idx_domain,domain in enumerate(options[start_number:end_number]):
        aux_df=df_file_1_filtered[df_file_1_filtered.domain_main==domain]#.reset_index(drop=True)
        print(f"{domain} with shape {str(aux_df.shape)}#{idx_domain} out of {len(options)}")
        # min_=aux_df.nr_words.astype(float).min()
        # max_=aux_df.nr_words.astype(float).max()
        # aux_df['nr_words_stand']=aux_df.nr_words.apply(lambda val: (val/max_))#(val-min_)/(max_-min_))
        # Calculate pairwise relative differences differences using NumPy
        pairwise_diff = np.abs((aux_df.nr_words.values[:, None] - aux_df.nr_words.values) / aux_df.nr_words.values)
        # pairwise_diff = np.abs(aux_df.nr_words_stand.values[:, None] - aux_df.nr_words_stand.values)
        # Create a aux_df.nr_words_standFrame to display the pairwise differences
        pairwise_diff_df = pd.DataFrame(pairwise_diff, index=list(aux_df.index), columns=list(aux_df.index))#range(aux_df.shape[0]))
        mask = np.tril(np.ones(pairwise_diff_df.shape)).astype(bool)  # Mask for values below diagonal
        pairwise_diff_df[mask] = -1  # Set values below diagonal to 1
        indices, columns = np.where((pairwise_diff_df <= threshold_rel_diff_nr_words)&(pairwise_diff_df > 0))
        ###per_domain_idx=tuple(zip(indices,columns))
        per_domain_idx=list(tuple(zip(pairwise_diff_df.index[indices],pairwise_diff_df.columns[columns])))
        calculate_collection_similarity(
            df_file_1_filtered=aux_df, threshold_collection_similarity=threshold_collection_similarity, idx_combinations_to_compare=per_domain_idx, file_path=file_path, domain=domain,directory_child_all_idx_combinations=directory_child_all_idx_combinations, directory_child_idx_to_keep=directory_child_idx_to_keep,vectorizer=vectorizer)
        # df_per_domain_idx = pd.DataFrame(per_domain_idx)
        # save_locally(
        #     main_folder=file_path, df=df_per_domain_idx, url="idx_to_compare", newspaper=f"{domain}",
        #     csv_pickle='pickle'
        # )
        # idx_combinations_to_compare+=per_domain_idx



    # save_locally(df_idx_combinations_to_compare
    # output={(start_number,end_number):idx_combinations_to_compare}
    # save_dictionary_to_file(dictionary=output, file_path=file_path,file_name=file_name)
    return "Completed"

# Aux functions
def clean_text(text):
    # Remove non-alphanumeric characters except for whitespace
    # text = re.sub(r'[^\w\s]', '', text)

    # Remove any whitespace at the beginning or end of the text
    text = text.strip()

    # Replace any consecutive whitespace characters with a single space
    text = re.sub(r"\s+", " ", text)
    return text

def save_locally(
    main_folder: str,
    df: pd.DataFrame,
    url: str,
    newspaper: str,
    start_year: str = "",
    start_month: str = "",
    end_year: str = "",
    end_month: str = "",
    twitter: bool = False,
    twitter_aux: str ="",
    csv_pickle: str = "csv",
):
    # Example usage:
    directory_path = f"{main_folder}/{newspaper.title()}/{replace_non_alphanumeric(input_string=url)}/".replace('//','/')
    create_directory(directory_path=directory_path)
    if twitter:
        file_path = f'{directory_path}{replace_non_alphanumeric(input_string=twitter_aux)}_{newspaper}_{start_year + start_month}{end_year + end_month}_all_news_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    else:
        file_path = f'{directory_path}{newspaper}_{start_year+start_month}{end_year+end_month}_all_news_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    if csv_pickle=='csv':
        df.to_csv(file_path.replace("__", "_").replace('//','/'),escapechar='\\') #
        print(file_path.replace("__", "_").replace('//','/'))
    elif csv_pickle=='pickle':
        print(file_path.replace("__", "_").replace('//','/').replace('.csv','.pkl'))
        df.to_pickle(file_path.replace("__", "_").replace('//','/').replace('.csv','.pkl'))
    return "Finished"



def read_files_in_directory(directory: str='',specific_sep: str=''):
    # Initialize an empty list to store DataFrames for each file
    dataframes_list = []

    for idx, components in enumerate(os.walk(directory)):
        root, dirs, files = components
        for filename in files:
            file_path = os.path.join(root, filename)

            # Assuming the files are in CSV format, adjust this based on your actual file format
            if file_path.endswith(".csv"):
                # Read the file into a DataFrame
                # file_path="all_news/Publico/ecosfera_publico_pt/Publico_201208_all_news_20240229_133106.csv"
                try:
                    if len(specific_sep)==0:
                        df = pd.read_csv(file_path)
                    else:
                        df = pd.read_csv(file_path,sep=specific_sep)
                    # df['newspaper_aux'] = root
                    # Append the DataFrame to the list

                except:
                    df = pd.DataFrame()
                    print(
                        f"Error: The CSV file {file_path} is empty or collection was stopped creating malformed input."
                    )
                dataframes_list.append(df)
            if file_path.endswith(".pkl"):
                # Read the file into a DataFrame
                # file_path="all_news/Publico/ecosfera_publico_pt/Publico_201208_all_news_20240229_133106.csv"
                try:
                    df = pd.read_pickle(file_path)
                    # df['newspaper_aux'] = root
                    # Append the DataFrame to the list

                except:
                    df = pd.DataFrame()
                    print(
                        f"Error: The Pickle file {file_path} is empty or collection was stopped creating malformed input."
                    )
                dataframes_list.append(df)
            # break

        # break
    return dataframes_list


def update_string(dict_, text):
    for key, val in dict_.items():
        text = text.replace(key, val)
    return text


def open_json_dumped_file(file_path):
    with open(file_path, "r") as json_file:
        to_exclude = json.load(json_file)
    return to_exclude


def replace_non_alphanumeric(input_string: str):
    # Use regex to replace non-alphanumeric characters with underscore
    try:
        result_string = re.sub(r"[^a-zA-Z0-9]", "_", input_string)
    except:
        print(f"Type of 'url': {type(input_string)}")
        print(f"Value of 'url': {input_string}")
        result_string = re.sub(r"[^a-zA-Z0-9]", "_", input_string)
    return result_string


def collect_filenames_in_directory(directory):
    # Initialize an empty list to store DataFrames for each file
    filenames_list = []

    for idx, components in enumerate(os.walk(directory)):
        root, dirs, files = components
        for filename in files:
            file_path = os.path.join(root, filename)
            # Assuming the files are in CSV format, adjust this based on your actual file format
            # if file_path.endswith(".csv"):
            # Read the file into a DataFrame
            # file_path="all_news/Publico/ecosfera_publico_pt/Publico_201208_all_news_20240229_133106.csv"
            filenames_list.append(file_path)
    return filenames_list

def clean_time(input_string):
    output=re.sub(r'[^\d.,:/ -]', '', input_string)
    chars_to_strip=',.- '
    while True:
        if len(output)==len(output.strip(chars_to_strip)):
            break
        else:
            output=output.strip(chars_to_strip)
    return output

def save_dictionary_to_file(dictionary, file_path,file_name):
    create_directory(file_path)
    try:
        file_ending_1=str(int(max(dictionary.keys())[0]))
        file_ending_2=str(int(max(dictionary.keys())[1]))

        final_results={f'{key[0]}_{key[1]}':val for key,val in dictionary.items()}
        file_path_final = file_path + file_name + f'_{file_ending_1}_{file_ending_2}.json'
    except:
        final_results=dictionary
        file_path_final = (file_path + file_name + '.json').replace('.json.json','.json') #in case .json is in file_name


    with open(file_path_final, 'w') as file:
        json.dump(final_results, file)
    return "completed"

def harmonic_mean(values):
    # Ensure there are no zero values in the input
    if any(value == 0 for value in values):
        raise ValueError("Input contains zero values. Harmonic mean is undefined in this case.")

    # Calculate the harmonic mean
    reciprocal_sum = sum(1 / value for value in values)
    harmonic_mean_result = len(values) / reciprocal_sum

    return harmonic_mean_result

def pairwise_combinations(words):
    return list(combinations(words, 2))

def generate_sublists(series):
    result_lists = []
    for i in range(len(series)):
        result_lists.append(series[:i+1])
    return result_lists


def mirror_matrix(df):
    # Convert the DataFrame to a NumPy array
    matrix = df.to_numpy()

    # Use np.triu_indices to get the upper triangular indices
    upper_triangle_indices = np.triu_indices(len(df), k=1)

    # Mirror the values across the diagonal
    matrix[upper_triangle_indices] = matrix.T[upper_triangle_indices]

    # Create a new DataFrame with the mirrored matrix
    mirrored_df = pd.DataFrame(matrix, index=df.index, columns=df.columns)

    return mirrored_df

def find_loops(graph, start, current, visited, path):
    visited[current] = True
    path.append(current)

    for neighbor in graph[current]:
        if not visited[neighbor]:
            find_loops(graph, start, neighbor, visited, path)
        elif neighbor == start:
            loops.append(path[:])

    path.pop()
    visited[current] = False

def identify_loops(graph):
    global loops
    loops = []

    for start in graph.keys():
        visited = {key: False for key in graph.keys()}
        path = []
        find_loops(graph, start, start, visited, path)

    return loops

def find_keys_with_value(my_dict, target_value):
    keys_with_value = [key for key, value in my_dict.items() if value == target_value]
    return keys_with_value

def merge_lists(d):
    merged_dict = {}

    for key, lst in d.items():
      aux_key=0
      lsts=[lst]
      for key_2, lst_2 in merged_dict.items():
        if key in lst_2:
          aux_key+=1
          lsts.append(lst_2)
          final_key=key_2
      if aux_key>1:
        raise ValueError(f"Key {key} is present in multiple lists.")
      elif aux_key==0:
        merged_dict[key]=lst
      else:
        aux_value=list(set(chain.from_iterable(lsts)))
        merged_dict[final_key]=aux_value
    return merged_dict

# Function to check if all words in a phrase appear in a given string
def check_phrase_in_string(phrase, string):
    #With this we just force that all sub-words from each keyword (ex.: ministro saude intergeracional) need to appear, disregarding the order
    aux_phrase = unidecode(phrase).split()
    pattern = r'\b(?:' + '|'.join(map(re.escape, aux_phrase)) + r')\b'
    matches = re.findall(pattern, string, flags=re.IGNORECASE)
    return len(matches) == len(aux_phrase)


# Function to check if any phrase from the keywords list appears in a given string
def check_keywords(string,keywords):
    if pd.isna(string) or string is None:  # Check for NaN or None values
        return False
    for phrase in keywords:
        if check_phrase_in_string(phrase, string):
            return True
    return False

