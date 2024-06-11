from utils_topification import *

model_name_tweets='jisdd_unigrams_nostopwords_top2vec_universal-sentence-encoder-multilingual'
model_name_news='jisdd_unigrams_nostopwords_top2vec_doc2vec'
model_name_parliament='jisdd_unigrams_bigrams_nostopwords_top2vec_doc2vec'
source_path='topification/'
destination_path='final_dfs/'

path_parliament_data='df_parliament_metadata.pkl'
path_parliament_data_aux='df_parliament_topification_inputs.pkl'

path_news_data='all_news_text_filtering1_wdate_cleaned_filtering2'
path_tweets_data='all_tweets_final'

data_col_parliament='Date'
data_col_news="date_final" #list
data_col_tweet='tweet_date' #string

data_sources=['tweets','news','parliament']

models_file_path=collect_filenames_in_directory(source_path)
path_tweets=[path for path in models_file_path if model_name_tweets in path and 'tweets' in path][0]
path_news=[path for path in models_file_path if model_name_news in path and 'news' in path][0]
path_parliament=[path for path in models_file_path if model_name_parliament in path and 'parliament' in path][0]

all_paths=[path_tweets,path_news,path_parliament]

df_tweets_aux=pd.concat(read_files_in_directory(path_tweets_data),axis=0)
df_tweets_aux2=df_tweets_aux[(df_tweets_aux.cond_len_2==True)&
                    (df_tweets_aux.cond_sdd_ji==True)].reset_index(drop=True)
df_tweets=df_tweets_aux2.loc[:,data_col_tweet].astype(str).str[:10].apply(lambda date_str:datetime.strptime(date_str, "%Y-%m-%d"))


df_news_aux=pd.concat(read_files_in_directory(path_news_data),axis=0)
df_news_aux2=df_news_aux[(df_news_aux.cond_len_100==True)&
                    (df_news_aux.cond_sdd_ji==True)&
                    (df_news_aux.date_final.apply(lambda val:val[0]).fillna('1900-01-01').astype(str).str[:4].astype(int)>=2007)].reset_index(drop=True)
df_news=df_news_aux2.loc[:,data_col_news].apply(lambda val: val[0]).fillna('1900-01-01').astype(str).str[:10].apply(lambda date_str:datetime.strptime(date_str, "%Y-%m-%d"))

df_parliament=pd.read_pickle(path_parliament_data).reset_index(drop=True)
df_parliament[data_col_parliament]=df_parliament.loc[:,data_col_parliament].astype(str).str[:10].apply(lambda date_str:datetime.strptime(date_str, "%Y-%m-%d"))

df_parliament_aux=pd.read_pickle(path_parliament_data_aux).reset_index(drop=True).loc[:,'ID']

all_date_cols=[df_tweets,df_news,df_parliament]



for path,data_source,data_col in zip(all_paths,data_sources,all_date_cols):

    model=joblib.load(path)
    topic_sizes, topic_nums = model.get_topic_sizes()
    
    #Note: documents, document_scores, document_ids = model.search_documents_by_topic(topic_num=topic, num_docs=size)
    doc_ids_per_topic={num:list(model.search_documents_by_topic(topic_num=num, num_docs=size)[2]) for size,num in zip(topic_sizes,topic_nums)}

    df_doc_ids_per_topic = pd.concat(
        [pd.DataFrame([val, [key for i in range(len(val))]], index=['doc_idx', 'topic_num']).T for key, val in
         doc_ids_per_topic.items()], axis=0).set_index('doc_idx')

    if data_source=='parliament':
        data_col_aux=pd.merge(df_parliament_aux,data_col,on=['ID'],how='left')
        data_col=data_col_aux.loc[:,data_col_parliament]

    df_doc_ids_per_topic['Data']=data_col

    save_locally(main_folder=destination_path,
                    df=df_doc_ids_per_topic,
                    url=data_source,
                    newspaper=data_source,
                 csv_pickle='csv'
                    )

