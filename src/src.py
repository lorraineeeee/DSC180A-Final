import pickle
import pandas as pd
import numpy as np
import json
from sklearn import metrics
import gensim
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.corpus import stopwords
import re
from numpy.linalg import norm

def cleaning(sentence):
    stop_words = set(stopwords.words('english'))
    tokens = re.sub(r'[^\w\s]', '', sentence.lower()).replace("\n", " ").split(" ")
    cleaned = [token for token in tokens if token not in stop_words]
    return " ".join(cleaned)

def tfidf(data,word):
    sentence = data['sentence']
    idf = np.log(len(sentence)/sentence.str.contains(word).sum())
    result = []
    for i in range(len(sentence)):
        tf = sentence.iloc[i].count(word)/(len(sentence.iloc[i]))
        result.append(tf*idf)
    return result

def preprocessing(sentence):
    tokens = sentence.split(" ")
    return [token for token in tokens if token!="" and token != " "]

def get_vectors_per_label(model,filename):
    f = open(filename)
    seeds = json.load(f)
    vector_per_label = []
    for key, value in seeds.items():
        lst = []
        for w in value:
            lst.append(model.wv[w])
        arr = np.asarray(lst)
        total = np.average(arr, axis=0)
        vector_per_label.append(total)
    return vector_per_label

def get_vector_per_doc(model, feature):
    vector_per_doc = []
    for feat in feature:
        lst = []
        for w in feat:
            lst.append(model.wv[w])
        arr = np.asarray(lst)
        total = np.average(arr, axis=0)
        vector_per_doc.append(total)
    return vector_per_doc

def predict_word2vec(seeds, vector_per_doc, vector_per_label):
    predictions = []
    labels = list(seeds.keys())
    for doc in vector_per_doc:
        cosine = []
        for label in vector_per_label:
            cosine.append(np.dot(doc,label)/(norm(doc)*norm(label)))
        max_value = max(cosine)
        max_index = cosine.index(max_value)
        predictions.append(labels[max_index])
    return predictions  
