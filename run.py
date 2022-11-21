import pickle
import pandas as pd
import numpy as np
import json
from sklearn import metrics
import gensim
from gensim.test.utils import common_texts
from gensim.models import Word2Vec

with open('test/testdata/df.pkl', 'rb') as f:
    data = pickle.load(f)
data["sentence"]=data["sentence"].apply(cleaning)
f = open('test/testdata/seedwords.json')
seeds = json.load(f)
result = pd.DataFrame()
for key, value in seeds.items():
    df = pd.DataFrame()
    for w in value:
        df[w] = tfidf(w)
    result[key] = df.sum(axis = 1)
data["prediction"] = result.idxmax(1)
metrics.f1_score(data["label"], data["prediction"], average="micro")
metrics.f1_score(data["label"], data["prediction"], average="macro")

features = data["sentence"].apply(preprocessing)
model = Word2Vec(sentences=features, size=100, window=5, min_count=1, workers=4)
model.save("word2vec.model")
model = Word2Vec.load("word2vec.model")
model.train(features, total_examples=len(data), epochs=20)

vector_per_label = get_vectors_per_label('test/testdata/seedwords.json')
vector_per_doc = get_vector_per_doc(features)
prediction_word2vec = predict_word2vec(vector_per_doc, vector_per_label)
data["prediction_word2vec"] = prediction_word2vec
metrics.f1_score(data["label"], data["prediction_word2vec"], average="micro")
metrics.f1_score(data["label"], data["prediction_word2vec"], average="macro")

