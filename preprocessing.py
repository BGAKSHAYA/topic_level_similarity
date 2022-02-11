import sys
import os
sys.path.insert(0, os.path.join("/vol1/backup/hailu/hailu/anaconda3/lib/python3.7/site-packages/"))
import numpy as np
import gensim
from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords
import pandas as pd
import glob

import configparser
config = configparser.ConfigParser()
config.read('configuration.val')
corpus_path = config['file']['corpus_path']
corpus_details = config['file']['corpus_details']
model_path = config['models']['path']


def read_documents():
    data = pd.read_csv(corpus_details)
    data = data.loc[data['plagiarism_type'] != 'translation']
    return data.groupby(['suspicious_document', 'source_reference']).agg(list)


def get_document_content(filename):
    try:
        filepath = glob.glob(corpus_path + '/**/' + filename, recursive=True)[0]
        f = open(filepath, "r", encoding='utf-8-sig')
        content = f.read()
        return content
    except Exception as e:
        return None


def read_all_document_names():
    data = pd.read_csv(corpus_details)
    data = data.loc[data['plagiarism_type'] != 'translation']
    data['suspicious_document'] = [suspicious_document_name.split('.')[0] + '.txt'
                                   for suspicious_document_name in data['suspicious_document']]
    files = set()
    files.update(list(data['suspicious_document']))
    files.update(list(data['source_reference']))
    return files


def preprocess_document(doc):
    doc_tokens = gensim.utils.simple_preprocess(doc,deacc=True)
    docs_nostop=[w for w in doc_tokens if w not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    doc = [lemmatizer.lemmatize(w) for w in docs_nostop]
    return doc


# This method runs processing for all documents in the corpus
def get_all_documents():
    docs = {}

    data = read_documents()
    for index, (row_index, row) in enumerate(data.iterrows()):
        suspicious_document_name, source_document_name = row.name
        source_doc = get_document_content(source_document_name)
        suspicious_doc = get_document_content(suspicious_document_name.split('.')[0] + '.txt')

        if not source_doc or not suspicious_doc:
            print(suspicious_document_name, source_document_name, 'not found')
            continue

        if source_document_name not in docs:
            source_doc = preprocess_document(source_doc)
            docs[source_document_name] = source_doc

        if suspicious_document_name not in docs:
            suspicious_doc = preprocess_document(suspicious_doc)
            docs[suspicious_document_name] = suspicious_doc

        if index % 50 == 0:
            print('Number of processed docs:', len(docs.keys()))
            np.save(model_path + 'preprocessed_content_topic_level.npy', docs)

    print('No of Docs added', len(docs.keys()))
    return docs


# This method runs processing for remaining documents in the corpus that are not already preprocessed
def get_remaining_documents():
    docs = np.load(model_path + 'preprocessed_content_topic_level.npy', allow_pickle=True).item()

    files = read_all_document_names()
    remaining_files = files - set(docs.keys())
    count = 0
    for file in remaining_files:
        doc = get_document_content(file.split('.')[0] + '.txt')
        doc = preprocess_document(doc)
        docs[file] = doc
        count += 1
        if count % 500 == 0:
            print('Number of processed docs:', len(docs.keys()))
            np.save(model_path + 'preprocessed_content_topic_level.npy', docs)
    return docs


docs = get_all_documents()
# docs = get_remaining_documents()

np.save(model_path + 'preprocessed_content_topic_level.npy', docs)

print('Saved')
