import sys
import os
sys.path.insert(0, os.path.join("/vol1/backup/hailu/hailu/anaconda3/lib/python3.7/site-packages/"))

import os
import gensim.corpora as corpora
import numpy as np
from gensim.models.wrappers import LdaMallet
from gensim.test.utils import datapath
import matplotlib.pyplot as plt
from gensim.models import CoherenceModel

from numpy import dot
from numpy.linalg import norm

import pandas as pd
import configparser


config = configparser.ConfigParser()
config.read('configuration.val')
corpus_path = config['file']['corpus_path']
corpus_details = config['file']['corpus_details']
model_path = config['models']['path']
mallet_home = config['mallet']['home']
mallet_path = config['mallet']['path']


def get_model_topics():
    topics = [50] + list(range(80, 161, 20))
    for num_topics in topics:
        model = LdaMallet.load(model_path + "lda_mallet_" + str(num_topics))
        print(str(num_topics) + ' topics')
        print('-----------------------------------------')
        for topic in range(num_topics):
            print(model.print_topic(topic))
        print('-----------------------------------------', end='\n\n\n')


# get_model_topics()

docs = np.load(model_path + 'preprocessed_content_topic_level.npy', allow_pickle=True).item()

content = list(docs.values())
filenames = list(docs.keys())

dictionary = corpora.Dictionary(content)
dictionary.filter_extremes(no_below=5, no_above=.5)
corpus = [dictionary.doc2bow(doc) for doc in content]

del docs

os.environ['MALLET_HOME'] = mallet_home
mallet_path = mallet_path


def create_models():
    for num_topics in range(80, 161, 20):
        print('b4 mallet',num_topics,'topics')
        prefix_path = model_path + "tmp/topics" + str(num_topics)
        model = LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, workers=12, id2word=dictionary, prefix=prefix_path)
        temp_file = datapath(model_path + "lda_mallet_" + str(num_topics))
        model.save(temp_file)
        print('Saved')
    print('Training done')


def compute_coherence_values():
    coherence_values = []
    topics = [50] + list(range(80, 161, 20))
    for num_topics in topics:
        model = LdaMallet.load(model_path + "lda_mallet_" + str(num_topics))
        print('Computing coherence for LDA model with ' + str(num_topics) + ' topics')
        coherence_model = CoherenceModel(model=model, texts=content, dictionary=dictionary, coherence='c_v')
        coherence_value = coherence_model.get_coherence()
        print(coherence_value)
        coherence_values.append(coherence_value)

    plt.bar(topics, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend("coherence_values", loc='best')
    plt.xticks(topics)
    plt.savefig(model_path + "topics_coherence_scores_artificial_80_to_160_step_20" + '.png')
    plt.show()


def read_documents():
    data = pd.read_csv(corpus_details)
    data = data.loc[data['plagiarism_type'] != 'translation']
    return data.groupby(['suspicious_document', 'source_reference']).agg(list)


# param num_topics: Use the LdaMallet trained on num_topics
# This method checks whether the source and suspicious documents are similar on topic-level
# i.e we compute cosine similarity between the suspicious doc vector and source doc vector.
# If the cosine similarity > TOPIC_LEVEL_SIMILARITY_THRESHOLD, we find the top 5 most similar topics
def get_plagiarized_documents(num_topics):
    global corpus
    data = read_documents()
    # Load the LdaMallet model corresponding to the num_topics
    model = LdaMallet.load(model_path + "lda_mallet_" + str(num_topics))

    for index, (row_index, row) in enumerate(data.iterrows()):
        suspicious_document_name, source_document_name = row.name
        suspicious_document_name = suspicious_document_name.split('.')[0] + '.txt'
        if source_document_name not in filenames or suspicious_document_name not in filenames: continue
        source_document_index = filenames.index(source_document_name)
        suspicious_document_index = filenames.index(suspicious_document_name)

        source_vector = model[corpus[source_document_index]]
        suspicious_vector = model[corpus[suspicious_document_index]]

        source_vector = np.array([value for (key, value) in source_vector]).reshape(-1)
        suspicious_vector = np.array([value for (key, value) in suspicious_vector]).reshape(-1)

        # Calculate cosine similarity between the suspicious doc vector and source doc vector
        denominator = (norm(suspicious_vector)*norm(suspicious_vector))
        similarity = dot(source_vector, suspicious_vector) / denominator

        # Find each topic contribution to similarity then sort these scores
        topic_scores = [(i, (source_vector[i] * suspicious_vector[i])/denominator) for i in range(num_topics)]
        topic_scores_sorted = sorted(topic_scores, reverse=True, key=lambda x: x[1])

        print(source_document_name, suspicious_document_name, similarity)

        if similarity > TOPIC_LEVEL_SIMILARITY_THRESHOLD:
            # If similarity > THRESHOLD, display top 5 similar topics between the source and suspicious documents
            for topic_no in range(5):
                topic = topic_scores_sorted[topic_no][0]
                print(topic, model.print_topic(topic))
        print("\n----------------------------------\n")


TOPIC_LEVEL_SIMILARITY_THRESHOLD = 0.7

'''
Computing coherence values does not converge as Topic modeling is trained on the whole corpus, So the below 
commented method takes forever to run
'''
# compute_coherence_values()

get_plagiarized_documents(80)

