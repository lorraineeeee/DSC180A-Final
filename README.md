# DSC180A-Final

## Datasets
NYT: This dataset includes news articles and 5 coarse-grained label categories, each with a set of seedwords. A pickle file provides sentences and their corresponding labels.

20News: This dataset includes news group and 7 coarse-grained label categories, each with a set of seedwords. Similarly, a pickle file provides sentences and their corresponding labels. 

## Word2Vec
Data cleaning and preprocessing: Change all sentences to be lower-cased. Remove punctuation, trailing white spaces and stop words using NLTK. 

Word2Vec Model Training: For NYT, initialize a Word2Vec vector from gensim and specify size = 200, window = 5, min_count = 1. Use 4 workers and train for 200 epochs. For 20News, specify size = 100, window = 5, min_count = 1. Use 4 workers and train for 20 epochs. 

Cosine similarity calculation: For each seedword, fetch the corresponding vector from Word2Vec model. Take the average of all vectors within a label and use that as the final word vector. For a single document, simply take the average of all word vectors within that document. Compute the cosine similarity between a document and a label using np.dot(doc,label)/(norm(doc)*norm(label), where doc represents the word vector for a document, and label represents the word vector for a label. Assign the document with the label that has the highest cosine similarity.

Micro/Macro F1 calculation: Use sklearn.metrics.f1_score to derive Micro and Macro F1 scores, respectively.

## TF-IDF
Data cleaning and preprocessing: Change all sentences to be lower-cased. Remove punctuation, trailing white spaces and stop words using NLTK.

TF-IDF calculation: For each seedword, calculate its TF-IDF value with respect to a specific document using tfidf(t,d) = tf(t,d) * idf(t), where tf(t,d)= the likelihood of the term appearing in the document, and idf(t)= log(number of documents/number of documents t appears). For each document, sum up all TF-IDF values for all seedwords within a label, and assign the document with the label that has the highest TF-IDF sum.

Micro/Macro F1 calculation: Use sklearn.metrics.f1_score to derive Micro and Macro F1 scores, respectively.
