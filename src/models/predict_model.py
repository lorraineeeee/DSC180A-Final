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
