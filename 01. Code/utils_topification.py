import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.tokenize import WhitespaceTokenizer
import joblib
from sklearn.decomposition import NMF
from datetime import datetime
from unidecode import unidecode
import string
import os
from collections import Counter
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from itertools import combinations
import random
from scipy.sparse import save_npz, load_npz
import pickle
from top2vec import Top2Vec
import nltk
from nltk.util import bigrams
from nltk.corpus import stopwords
import json
from itertools import chain
from gensim.models import Word2Vec, KeyedVectors
from utils_collection_urls import *
import umap


class BigramUnigramTokenizer:
    def __init__(self, stopwords=None, portuguese_stopwords_final=None):
        self.stopwords = stopwords or portuguese_stopwords_final

    def remove_stop_words(self, text):
        # Get the list of Portuguese stop words
        pattern = r"\b(?:{})\b".format("|".join(map(re.escape, self.stopwords)))
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Remove weird chars
        chars_to_remove = ["--", "---", "----"]
        pattern = "|".join(re.escape(char) for char in chars_to_remove)
        text = re.sub(pattern, " ", text)

        # Repeat here
        while re.findall(r"\s-\s|-\s|\s-", text):
            text = re.sub(r"\s-\s|-\s|\s-", " ", text)

        text = re.sub("(?:^|\s)[a-z](?=\s|$)", " ", text)

        # Remove extra spaces
        text = self.recursively_replace_double_spaces(text)

        return text.strip()

    def tokenize(self, text):
        wstokenizer = WhitespaceTokenizer()

        # Tokenize the text into words
        words = wstokenizer.tokenize(text)

        # Remove stop words and weird characters
        cleaned_text = self.remove_stop_words(text)

        # Tokenize the cleaned text into words
        words_bigrams = wstokenizer.tokenize(cleaned_text)

        # Generate bigrams
        bigram_tokens = [" ".join(tup) for tup in list(bigrams(words_bigrams))]

        unigrams_bigrams = bigram_tokens + words

        return unigrams_bigrams

    @staticmethod
    def recursively_replace_double_spaces(text):
        if "  " in text:
            return BigramUnigramTokenizer.recursively_replace_double_spaces(
                text.replace("  ", " ")
            )

        return text

class NMF_model_object:  ##TO REMOVE - change to _personalized not _JF
    def __init__(self, name,topic_words, vocab, word_vectors, H_matrix, W_matrix,error):
        self.name = name
        self.topic_words = topic_words
        self.vocab = vocab
        self.word_vectors = word_vectors
        # self.Headers = headers
        self.H = H_matrix
        self.W = W_matrix
        self.error = error #factorization error

    def save(self, filepath):
        joblib.dump(self, filepath)

def top2vec_topification(
    documents: list = [],
    tokenization_approach: str = "unigrams_bigrams",
    embedding_approach: str = "doc2vec",
    portuguese_stopwords_final: list = [],
    topification_approach: str = "top2vec"

):
    if tokenization_approach == "unigrams_bigrams":
        tokenizer = BigramUnigramTokenizer(portuguese_stopwords_final=portuguese_stopwords_final)
    elif tokenization_approach == "unigrams":
        tokenizer = WhitespaceTokenizer()
    else:
        print("Please insert a valid tokenization_approach")
        return None

    if len(documents) == 0:
        print("No documents inserted")
        return None

    if embedding_approach not in ("doc2vec", "universal-sentence-encoder-multilingual"):
        print("Please insert a valid embedding_approach")
        return None

    if topification_approach=='nmf':
        model = Top2Vec(documents=documents,embedding_model=embedding_approach,tokenizer=tokenizer.tokenize,min_count=1)
    else:
        model = Top2Vec(documents=documents, embedding_model=embedding_approach, tokenizer=tokenizer.tokenize)

    return model


def NMF_model(
    documents: list = [],
    tokenization_approach: str = "unigrams_bigrams",
    n_components: int = -1,
    portuguese_stopwords_final: list=[]
):
    if tokenization_approach == "unigrams_bigrams":
        tokenizer = BigramUnigramTokenizer(portuguese_stopwords_final=portuguese_stopwords_final)
    elif tokenization_approach == "unigrams":
        tokenizer = WhitespaceTokenizer()
    else:
        print("Please insert a valid tokenization_approach")
        return None

    if len(documents) == 0:
        print("No documents inserted")
        return None

    if n_components == -1:
        print("Please select a n_components")
        return None

    ##Implement tfiedf e NMF model
    model = TfidfVectorizer(tokenizer=tokenizer.tokenize, lowercase=False)
    X = model.fit_transform(documents)
    model_headers = model.get_feature_names_out()

    nmf = NMF(n_components=n_components, init="nndsvd", random_state=0, max_iter=1000)
    W = nmf.fit_transform(X)  # W contains the document-topic matrix
    H = nmf.components_  # H contains the topic-term matrix
    error = nmf.reconstruction_err_
    return W, H, error, model_headers


def nmf_topification(
    documents: list = [],
    tokenization_approach: str = "unigrams_bigrams",
    n_components_list: list = [],
    nr_top_words=50,
    embedding_approach: str = "doc2vec",
    portuguese_stopwords_final: list = [],
    topification_approach: str='nmf'
):
    W_list, H_list, error_list, topic_words_list, topics_vec_list = {}, {}, {}, {}, {}
    results, results_topics = {}, {}

    ##Implement vectorization of documents
    model_word2vec = top2vec_topification(
        documents=documents,
        tokenization_approach=tokenization_approach,
        embedding_approach=embedding_approach,
        portuguese_stopwords_final=portuguese_stopwords_final,
        topification_approach=topification_approach
    )
    model_word2vec_word_vectors = model_word2vec.word_vectors
    model_word2vec_vocab = model_word2vec.vocab

    for n_components in n_components_list:
        W, H, error,model_headers = NMF_model(
            documents=documents,
            tokenization_approach=tokenization_approach,
            n_components=n_components,
            portuguese_stopwords_final=portuguese_stopwords_final
        )
        # W_list[(n_components, nr_top_words)] = W.tolist()
        # H_list[(n_components, nr_top_words)] = H.tolist()
        # error_list[(n_components, nr_top_words)] = error

        topic_words, topics_vec = define_topics_properties(
            H_matrix=H,
            nr_top_words=nr_top_words,
            model_headers=model_headers,
            model_word2vec_vocab=model_word2vec_vocab,
            model_word2vec_word_vectors=model_word2vec_word_vectors,
        )
        # topic_words_list[(n_components, nr_top_words)] = topic_words
        # topics_vec_list[(n_components, nr_top_words)] = topics_vec

        nmf_model=NMF_model_object(name='_'.join([str(n_components), str(nr_top_words)]),
                                   topic_words=topic_words,
                                   vocab=model_word2vec_vocab,
                                   word_vectors=model_word2vec_word_vectors,
                                   H_matrix=H,
                                   W_matrix=W,
                                   error=error)

        results[nmf_model.name]=nmf_model

        # topics_coh = {}
        # for topic, vecs in topic_words_list.items():
        #     topics_coh[topic] = calculate_model_topic_coherence(topic_words=vecs,vocab=model_word2vec_vocab,word_vectors=model_word2vec_word_vectors)
        # final_coh = sum(topics_coh.values()) / len(topics_coh.values())
        #
        # topic_exclusivity = calculate_model_topic_exclusivity_jaccard_similarity(
        #     topic_words
        # )
        #
        # results[(n_components, nr_top_words)] = (final_coh, topic_exclusivity)
        #
        # #TOREMOVE
        # doc_type = "parliament"
        # tokenization_approach = "unigrams_bigrams"
        # topification_approach = "nmf"
        # main_destination_folder = f"topification/{doc_type}/nmf/"
        # destination_file_name_main = (
        #     f"jisdd_{tokenization_approach}_nostopwords_{topification_approach}_"
        # )
        #
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
        # # TOREMOVE until here


    return results #error_list, H_list, W_list, results


### Detailed functions per topification appraoch
## NMF




def define_topics_properties(
    H_matrix,
    nr_top_words,
    model_headers,
    model_word2vec_vocab,
    model_word2vec_word_vectors,
):


    topics = {}
    indexes = np.argsort(H_matrix, axis=1)[:,::-1][:, :nr_top_words]

    # Flatten the indexes array to a 1D array
    flattened_indexes = indexes.flatten()

    # Retrieve words corresponding to the indexes
    topic_words = [model_headers[index] for index in flattened_indexes]

    # Reshape the result to match the original shape
    topic_words = np.array(topic_words).reshape(indexes.shape)

    topics_vec = {}
    for topic, words in enumerate(topic_words):
        words_vecs = []
        for word in words:
            idx=model_word2vec_vocab.index(word)
            # idx = np.where(model_word2vec_vocab == word)[0]
            words_vecs.append(model_word2vec_word_vectors[idx][0])

        topics_vec[topic] = np.array(words_vecs)

    return topic_words, topics_vec

def harmonic_mean(values):
      # Ensure there are no zero values in the input
      if any(value == 0 for value in values):
          raise ValueError("Input contains zero values. Harmonic mean is undefined in this case.")

      # Calculate the harmonic mean
      reciprocal_sum = sum(1 / value for value in values)
      harmonic_mean_result = len(values) / reciprocal_sum

      return float(harmonic_mean_result)

def calculate_model_coherence_exclusivity_mean(all_topic_coherence,all_topic_exclusivity,all_topic_coherence_centroid, topification_approach):
  output={}
  for key in all_topic_coherence.keys():
    if topification_approach.strip('/') in key:
      output[key]=harmonic_mean([all_topic_coherence[key],all_topic_exclusivity[key]])
    else:
      output[key]=harmonic_mean([all_topic_coherence[key],all_topic_exclusivity[key],all_topic_coherence_centroid[key]])
  return output


#Functions to assess model quality

def calculate_model_topic_coherence(topic_words: np.array,vocab: np.array,word_vectors: np.array,metric_type:str ='mean'):
  topic_coherences=[]
  for topic in topic_words:
    vecs=[]
    for word in topic:
      vecs.append(word_vectors[vocab.index(word)])

    vecs=np.array(vecs)

    arr=cosine_similarity(vecs)


    # Flatten the resulting array and convert to a list
    below_diagonal_values = arr[np.tril_indices(arr.shape[0], k=-1)]

    if len(below_diagonal_values)!=sum(range(1, len(topic))):
      print('ERROR')

    if metric_type=='mean':
        topic_coherences.append(np.mean(below_diagonal_values))
    elif metric_type=='median':
        topic_coherences.append(np.median(below_diagonal_values))


  if metric_type=='mean':
    model_topic_coherence=np.mean(topic_coherences)
  elif metric_type=='median':
    model_topic_coherence=np.median(topic_coherences)
  return float(model_topic_coherence)

def calculate_model_topic_coherence_centroid(topic_words: np.array,vocab: np.array,word_vectors: np.array,topic_vectors: np.array,metric_type:str ='mean'):

  topic_coherences=[]
  for idx_topic,topic in enumerate(topic_words):
    pairs_cos=[cosine_similarity(word_vectors[vocab.index(word)].reshape(1,-1),topic_vectors[idx_topic].reshape(1,-1))[0][0] for word in topic]

    if metric_type=='mean':
      topic_coherences.append(np.mean(pairs_cos))
    elif metric_type=='median':
      topic_coherences.append(np.median(pairs_cos))
  if metric_type=='mean':
    model_topic_coherence=np.mean(topic_coherences)
  elif metric_type=='median':
    model_topic_coherence=np.median(topic_coherences)
  return float(model_topic_coherence)

def calculate_model_topic_exclusivity_jaccard_similarity(
    topic_words: np.array, metric_type:str="mean"
):
    jaccard_list = []
    topic_words = topic_words
    for topic1, topic2 in combinations(topic_words, 2):
        # We need to do 1-jaccard similarity for higher values be better than low values
        jaccard_list.append(
            1
            - len(set(topic1).intersection(set(topic2)))
            / len(set(list(topic1) + list(topic2)))
        )
    if metric_type == "mean":
        topic_exclusivity = np.mean(jaccard_list)
    elif metric_type == "median":
        topic_exclusivity = np.median(jaccard_list)
    return float(topic_exclusivity)


def compare_topics(row, topic1, topic2):
    num = row.loc[topic1]
    deno = row.loc[topic2]

    if deno == 0:
        deno = aux[aux.loc[:, topic2] != 0].loc[:, topic2].min()
    return num / deno


def assign_topic(Topic_number):
    aux = (stats_topic4_allothertopics == False) | (stats_topic4_topic3 == False)

    if aux.loc[Topic_number] == True:
        output = top_topics_per_doc.loc[Topic_number, 0]
    else:
        output = -1
    return output


## Top2Vec
def get_colors(nr):
    # Use the 'tab20' colormap
    colormap = plt.cm.get_cmap("tab20", nr)

    colors = [colormap(i) for i in range(nr)]

    return colors


def aux_topic_comparison(row):
    if row.loc["topic_name_manuallyselected"] != row.loc["topic_name_predicted"]:
        if row.loc["topic_name_manuallyselected"] == "NA":
            output = row.loc["topic_name_predicted"]
        elif row.loc["topic_name_predicted"] == "NA":
            output = row.loc["topic_name_manuallyselected"]
        else:
            output = "ERROR"
    else:
        output = row.loc["topic_name_predicted"]
    return output


def choose_topic_vector(topic, word_vectors, vocab):
    if topic == "outro":
        word_vector_input = np.mean(
            [
                word_vectors[vocab.index(word)]
                for word in ["agricultura", "policia", "tap", "rtp"]
            ],
            axis=0,
        )
    else:
        word_vector_input = word_vectors[vocab.index(topic.lower())]
    return word_vector_input


def assign_unique_words_to_topic(
    created_topic_nums, created_topic_vectors, word_vectors, model
):
    model_vocab = model.vocab
    df_topic_distance = pd.DataFrame(data=model_vocab, columns=["vocab"])
    for idx, topic in enumerate(created_topic_nums):
        vector = created_topic_vectors[idx]
        df_topic_distance[f"Topic_{topic}_distance"] = df_topic_distance.vocab.apply(
            lambda x: distance.euclidean(vector, word_vectors[model_vocab.index(x)])
        )
    df_topic_distance = df_topic_distance.set_index("vocab")
    df_closest_topic_per_word = df_topic_distance.apply(
        lambda row: df_topic_distance.columns[row.argmin()].split("_distance")[0],
        axis=1,
    )

    return df_closest_topic_per_word.reset_index().groupby(0)["vocab"].apply(list)


def generate_sublists(series):
    result_lists = []
    for i in range(len(series)):
        result_lists.append(series[:i+1])
    return result_lists

def topics_aggregation(model: Top2Vec,
                       topic_vectors: np.array,
                       document_vectors: np.array,
                       topic_sizes: np.array,
                       topic_nums: np.array,
                       word_vectors: np.array,
                       topic_words: np.array,
                       max_decline: int= -5,
                       pre_defined_topics: list=['saude','seguranca social','educacao','divida publica','ambiente','outro'],
                       pre_defined_topic_outro_lowercased : list=['agricultura', 'policia', 'tap', 'rtp'],
                       ):

    # 1- function get topic_vectors,topic_nums, topic_sizes, word_vectors
    # topic_vectors = model.topic_vectors
    # document_vectors = model.document_vectors
    # topic_sizes, topic_nums = model.get_topic_sizes()
    # word_vectors = model.word_vectors
    # topic_words, word_scores, topic_nums = model.get_topics(model.get_num_topics())


    ### Get the nearest topic to each of our pre_definied_topics
    similar_identified_topics = {}
    for topic in pre_defined_topics:
        if topic == 'outro':
            topic_list = pre_defined_topic_outro_lowercased
        else:
            topic_list = [topic]
        try:
            topic_words_aux, word_scores_aux, topic_scores_aux, topic_nums_aux = model.search_topics(keywords=topic_list,
                                                                                                     num_topics=len(
                                                                                                         topic_vectors))
            similar_identified_topics[topic] = dict(zip(topic_nums_aux, topic_scores_aux)) #topic_scores_aux is cosine similarity between topics of top 50 words
        except:
            #In case the word was not learned by the model
            continue

    # print('similar_identified_topics.keys() :'+str(similar_identified_topics.keys())) #to check if agricultura, etc are in keys or outro

    similar_identified_topics_df = pd.DataFrame.from_dict(similar_identified_topics)

    ### See closest pre_defined_topic to each topic, and order the table based difference between closest and second_closest distance to a pre_defined topic
    idxs_max_secondmax=similar_identified_topics_df.apply(lambda row: max(row)-sorted(row)[-2],axis=1).sort_values(ascending=False).index
    similar_identified_topics_df_ordered=similar_identified_topics_df.apply(lambda row: similar_identified_topics_df.columns[np.argmax(row)],axis=1).loc[idxs_max_secondmax]


    ### Associate topic to doc
    topic_documents_docids = []
    topic_documents_topic = []
    for topic, size in zip(topic_nums, topic_sizes):
        documents_aux, document_scores_aux, document_ids_aux = model.search_documents_by_topic(topic_num=topic,
                                                                                               num_docs=size)
        topic_documents_docids.append(document_ids_aux)
        topic_documents_topic.append([topic for el in document_ids_aux])

    topic_documents_output_df = pd.DataFrame(index=chain.from_iterable(topic_documents_docids),
                                             data=chain.from_iterable(topic_documents_topic), columns=['topic'])


    ### Aggregation of topics
    topics_final, topics_final_words = {}, {}
    for idx_plot, topic in enumerate(pre_defined_topics):
        your_series = similar_identified_topics_df_ordered[similar_identified_topics_df_ordered == topic].index
        potential_topics_per_selected_topic = generate_sublists(your_series) #generate_sublists created lists of cummulative series (Ex.: input=(1,2,3) output= [(1), (1,2), (1,2,3)]

        similarities = []


        for potential_topics in potential_topics_per_selected_topic:
            #New topic vector will be the centroid of the documents assigned to it
            new_topic = np.mean([document_vectors[idx_doc] for idx_doc in topic_documents_output_df[
                topic_documents_output_df.topic.isin(potential_topics)].index], axis=0)
            if topic == 'outro':
                word_vector_input = np.mean([word_vectors[model.vocab.index(word.lower())] for word in
                                             pre_defined_topic_outro_lowercased], axis=0)
            else:
                word_vector_input = word_vectors[model.vocab.index(topic.lower())]
            similarity = cosine_similarity(new_topic.reshape(1, -1), word_vector_input.reshape(1, -1))
            similarities.append(similarity[0][0])

        #### Condition 1 - we stop aggregating when the cosine similarity declines more than 5% (5 is the default value)
        #### Condition 2 - to ensure that we only consider the previous condition when the declines occur after reaching the maximum value
        cond_variation = pd.Series(similarities).apply(lambda val: (val / pd.Series(similarities).max() - 1) * 100) > max_decline
        cond_first_variation = pd.Series(similarities).reset_index().loc[:, 'index'].apply(
            lambda val: val <= pd.Series(similarities).argmax())


        try:
            topics_final_aux = pd.Series(similarities)[cond_variation | cond_first_variation].index[-1]
            topics_final[topic] = list(potential_topics_per_selected_topic[topics_final_aux])
        except:
            # topics_final_aux = pd.Series(similarities)[cond_variation | cond_first_variation].index[-1]
            print('topic :' + str(topic)) #to know topic name
            print('similar_identified_topics_df_ordered.value_counts() :'+str(similar_identified_topics_df_ordered.value_counts())) #to check if a specific topic is never chosen
            print('pd.Series(similarities).apply(lambda val: (val / pd.Series(similarities).max() - 1) * 100) :'+str(pd.Series(similarities).apply(lambda val: (val / pd.Series(similarities).max() - 1) * 100))) #to check similalirty evolution
            print('cond_variation :'+str(cond_variation))
            print('cond_first_variation: '+str(cond_first_variation))
            topics_final[topic] = []


        # topics_final_words[topic] = list(
        #     chain.from_iterable([topic_words[topic] for topic in potential_topics_per_selected_topic[topics_final_aux]]))



    return topics_final


def define_top_words_aggregated_topics(model: Top2Vec, aggregated_topics_aux: dict, topic_words: np.array, word_vectors: np.array, topic_vectors: np.array, vocab: np.array, pre_defined_topic_outro_lowercased: list):

    top_words_per_pre_selected_topic={}

    for pre_selected_topic,all_topics_idx in aggregated_topics_aux.items():

        if len(all_topics_idx)==0 or pd.isnull(all_topics_idx).all():
            top_words_per_pre_selected_topic[pre_selected_topic]=[]
        else:
            if pre_selected_topic=='outro':
                pre_selected_topic_vector=np.mean(np.array([word_vectors[model.word_indexes[word]] for word in pre_defined_topic_outro_lowercased if word in vocab]),axis=0)
            else:
                pre_selected_topic_vector=word_vectors[model.word_indexes[pre_selected_topic]] #in case we want to change to compare top words with topic definition (ex.: saude) instead to the average of the aggreagted topics (next line of code)
            pre_selected_topic_vector = np.mean(np.array([topic_vectors[idx] for idx in all_topics_idx]),axis=0)

            all_topics_idx_words=np.unique(np.concatenate([topic_words[idx] for idx in all_topics_idx]))
            all_topics_idx_words_vectors=model.word_vectors[[model.word_indexes[word] for word in all_topics_idx_words]]

            cosine_similarities = cosine_similarity(pre_selected_topic_vector.reshape(1, -1), all_topics_idx_words_vectors).flatten()

            top_indices = cosine_similarities.argsort()[::-1][:50] #TOP 50

            top_words = all_topics_idx_words[top_indices]

            top_words_per_pre_selected_topic[pre_selected_topic]=top_words.tolist()


    return top_words_per_pre_selected_topic
