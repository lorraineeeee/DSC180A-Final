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

def train(data, features, window=5, min_count=1, workers=4, epochs=20):
    model = Word2Vec(sentences=features, window = window, min_count= min_count, workers=workers)
    model.save("word2vec.model")
    model = Word2Vec.load("word2vec.model")
    model.train(features, total_examples=len(data), epochs=epochs)
    return model
