import pickle
import pandas as pd
import numpy as np
import json
from sklearn import metrics
#pip3 install gensim
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
from src.data.make_dataset import * 
from src.features.build_features import * 
from src.models.predict_model import * 
from src.models.train_model import *
import sys

data = make_dataset('test/testdata/df.pkl')
  
data["sentence"]=data["sentence"].apply(cleaning)
test = True
if (len(sys.argv) > 1 and sys.argv[1] != "test"):
    test = False
seedwords = ""
if (test):
    with open('config/data-params.json') as fh:
        data_params = json.load(fh)
    seedwords = data_params["test"]
f = open(seedwords)
seeds = json.load(f)
result = pd.DataFrame()
for key, value in seeds.items():
    df = pd.DataFrame()
    for w in value:
        df[w] = tfidf(data,w)
    result[key] = df.sum(axis = 1)

data["prediction"] = result.idxmax(1)

print(metrics.f1_score(data["label"], data["prediction"], average="micro"))
print(metrics.f1_score(data["label"], data["prediction"], average="macro"))

features = data["sentence"].apply(preprocessing)
model = train(data, features)
vector_per_label = get_vectors_per_label(model, seedwords)
vector_per_doc = get_vector_per_doc(model, features)

f = open(seedwords)
seeds = json.load(f)
prediction_word2vec = predict_word2vec(seeds, vector_per_doc, vector_per_label)
data["prediction_word2vec"] = prediction_word2vec
print(metrics.f1_score(data["label"], data["prediction_word2vec"], average="micro"))
print(metrics.f1_score(data["label"], data["prediction_word2vec"], average="macro"))
