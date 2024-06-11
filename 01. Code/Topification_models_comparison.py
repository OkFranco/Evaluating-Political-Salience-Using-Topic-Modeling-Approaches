"""
1 - import models
2 - differentiate between top2vec and nmf
     - if nmf insert process to convert to object NMF
3 - identify key metrics to consider
    - calculate_model_topic_coherence (mean and median)
    - calculate_model_topic_coherence_centroid (mean and median)
    - calculate_model_topic_exclusivity_jaccard_similarity (mean and median)
    - calculate_model_coherence_exclusivity_mean (with means and medians)

    - topic aggregation process
        - can we do it with nmf

    - repeat previous but only for the topics that we have pre-selected



    - ability to find our topics

    vocab = model.vocab
    topic_vectors = model.topic_vectors
    document_vectors = model.document_vectors
    topic_sizes, topic_nums = model.get_topic_sizes()
    word_vectors = model.word_vectors
    topic_words, word_scores, topic_nums = model.get_topics(model.get_num_topics())



    files=collect_filenames(topification)
    parliament=[file for file in files if 'parliament' in file and 'doc2vec' in file]

"""
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
import joblib

source_path='topification'
destination_path='topification/results/'
max_decline= -5
pre_defined_topics_bigrams=['saude','seguranca social','educacao','divida publica','ambiente','outro']#['saude','seguranca social','educacao','divida publica','ambiente','agricultura','policia','TAP','RTP']
pre_defined_topics_unigrams=['saude','pensoes','educacao','divida','ambiente','outro'] #['saude','pensoes','educacao','divida','ambiente','agricultura','policia','TAP','RTP']
pre_defined_topic_outro_lowercased =['agricultura', 'policia', 'tap', 'rtp']

#1- collect models
models_file_path_aux=collect_filenames_in_directory(source_path)
models_file_path=[path for path in models_file_path_aux if destination_path not in path and 'old' not in path]
models_names=[path.split('/')[-1] for path in models_file_path]

data_sources=list(set([path.split('/')[1] for path in models_file_path]))

for data_source in data_sources:
    models,all_topic_coherence,all_topic_exclusivity,all_topic_coherence_centroid,aggregated_topics={},{},{},{},{}
    aggregated_all_topic_coherence,aggregated_all_topic_exclusivity,aggregated_all_topic_coherence_centroid={},{},{}

    all_topic_words,aggregated_all_topic_words={},{}

    names_path_pairs_to_consider=[(name,path) for name,path in zip(models_names,models_file_path) if data_source+'/' in path]

    if len(names_path_pairs_to_consider)!=0: #In case there is no model

        for name,path in names_path_pairs_to_consider:
            data_source = path.split('/')[1] + '/'
            topification_approach = path.split('/')[2] + '/'

            model=joblib.load(path)

            models[name]=model

            """" condition in case it is NMF"""

            if 'bigrams' in name:
                pre_defined_topics=pre_defined_topics_bigrams
            else:
                pre_defined_topics = pre_defined_topics_unigrams


            if topification_approach.strip('/')=='nmf':
                topic_words=model.topic_words
                vocab=model.vocab
                word_vectors=model.word_vectors
            elif topification_approach.strip('/')=='top2vec':
                topic_vectors = model.topic_vectors
                document_vectors = model.document_vectors
                topic_sizes, topic_nums = model.get_topic_sizes()
                topic_words, word_scores, topic_nums = model.get_topics(model.get_num_topics())
                vocab = model.vocab
                word_vectors = model.word_vectors


            all_topic_coherence[name]=calculate_model_topic_coherence(topic_words=topic_words,vocab=vocab,word_vectors=word_vectors)
            all_topic_exclusivity[name]=calculate_model_topic_exclusivity_jaccard_similarity(topic_words=topic_words)
            all_topic_words[name]=topic_words.tolist()
            if topification_approach.strip('/')=='top2vec':
                all_topic_coherence_centroid[name]=calculate_model_topic_coherence_centroid(topic_words=topic_words,vocab=vocab,word_vectors=word_vectors,topic_vectors=topic_vectors)

                ### To assess models based on the topics selected to be aggregated

                aggregated_topics_aux=topics_aggregation(model=model, topic_vectors=topic_vectors, document_vectors=document_vectors, topic_sizes=topic_sizes, topic_nums=topic_nums, word_vectors=word_vectors, topic_words=topic_words,max_decline=max_decline,pre_defined_topics=pre_defined_topics,pre_defined_topic_outro_lowercased=pre_defined_topic_outro_lowercased)
                aggregated_topics[name] = aggregated_topics_aux

                aggregated_all_topic_words[name]=define_top_words_aggregated_topics(model=model, aggregated_topics_aux=aggregated_topics_aux, topic_words=topic_words, word_vectors=word_vectors, topic_vectors=topic_vectors,vocab=vocab, pre_defined_topic_outro_lowercased=pre_defined_topic_outro_lowercased)

                all_arrays = [array for array in aggregated_topics_aux.values() if len(array)!=0]
                idxs_topics_to_aggregate = np.concatenate(all_arrays)

                aggregated_topic_vectors = np.array([topic_vectors[idx] for idx in idxs_topics_to_aggregate])
                aggregated_topic_words = np.array([topic_words[idx] for idx in idxs_topics_to_aggregate])

                aggregated_all_topic_coherence[name] = calculate_model_topic_coherence(topic_words=aggregated_topic_words, vocab=vocab,
                                                                            word_vectors=word_vectors)
                aggregated_all_topic_exclusivity[name] = calculate_model_topic_exclusivity_jaccard_similarity(topic_words=aggregated_topic_words)
                aggregated_all_topic_coherence_centroid[name] = calculate_model_topic_coherence_centroid(topic_words=aggregated_topic_words, vocab=vocab,
                                                                                              word_vectors=word_vectors,
                                                                                              topic_vectors=aggregated_topic_vectors)

        all_coherence_centroid_coherence_exclusivity_medians_mean=calculate_model_coherence_exclusivity_mean(all_topic_coherence=all_topic_coherence,all_topic_exclusivity=all_topic_exclusivity,all_topic_coherence_centroid=all_topic_coherence_centroid,topification_approach=topification_approach)

        aggregated_all_coherence_centroid_coherence_exclusivity_medians_mean = calculate_model_coherence_exclusivity_mean(
            all_topic_coherence=aggregated_all_topic_coherence, all_topic_exclusivity=aggregated_all_topic_exclusivity,
            all_topic_coherence_centroid=aggregated_all_topic_coherence_centroid,topification_approach=topification_approach)

        file_path=destination_path+data_source+topification_approach


        save_dictionary_to_file(dictionary=all_topic_words, file_path=file_path, file_name='all_topic_words')
        save_dictionary_to_file(dictionary=all_topic_coherence, file_path=file_path, file_name='all_topic_coherence')
        save_dictionary_to_file(dictionary=all_topic_exclusivity, file_path=file_path, file_name='all_topic_exclusivity')
        save_dictionary_to_file(dictionary=all_topic_coherence_centroid, file_path=file_path, file_name='all_topic_coherence_centroid')
        save_dictionary_to_file(dictionary=all_coherence_centroid_coherence_exclusivity_medians_mean, file_path=file_path, file_name='all_coherence_centroid_coherence_exclusivity_medians_mean')

        """condition != nmf"""
        save_dictionary_to_file(dictionary=aggregated_topics, file_path=file_path, file_name='aggregated_topics')
        save_dictionary_to_file(dictionary=aggregated_all_topic_words, file_path=file_path, file_name='aggregated_all_topic_words')
        save_dictionary_to_file(dictionary=aggregated_all_topic_coherence, file_path=file_path, file_name='aggregated_all_topic_coherence')
        save_dictionary_to_file(dictionary=aggregated_all_topic_exclusivity, file_path=file_path, file_name='aggregated_all_topic_exclusivity')
        save_dictionary_to_file(dictionary=aggregated_all_topic_coherence_centroid, file_path=file_path, file_name='aggregated_all_topic_coherence_centroid')
        save_dictionary_to_file(dictionary=aggregated_all_coherence_centroid_coherence_exclusivity_medians_mean, file_path=file_path, file_name='aggregated_all_coherence_centroid_coherence_exclusivity_medians_mean')


