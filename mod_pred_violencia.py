#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from unicodedata import normalize
from wordcloud import WordCloud
import tensorflow as tf
import html
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import seaborn as sns
import matplotlib.pyplot as ptl
import nltk
from nltk import SnowballStemmer


# In[ ]:


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


# In[ ]:


def armar_bag_of_words(dataset):
    corpus = []
    all_stopwords = stopwords.words('spanish')
    removeList=["no", "nunca"]
    all_stopwords = [e for e in all_stopwords if e not in removeList]
    for i, value in dataset.items():
        review = str(html.unescape(dataset[i]))
        review = cleanhtml(review)
        review = re.sub(r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+", r"\1", 
            normalize( "NFD", review), 0, re.I)
        review = normalize( 'NFC', review)
        review = re.sub('[^a-zA-Zá-ú0-9]', ' ', review)
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus.append(review)
    return corpus



# In[ ]:


def limpiar_texto(dataset):
    review = str(html.unescape(dataset))
    review = cleanhtml(review)
    review = normalize( 'NFC', review)
    review = review.replace('\n', ' ')
    review = re.sub('[^a-zA-Zá-ú0-9."",?!:]', ' ', review)
    review = review.split()
    review = [word for word in review if len(word) > 1 or word in set(['a', 'e', 'y', 'o', 'u'])]
    review = ' '.join(review)
    return review





# In[ ]:



def contar_palabras(corpus, max_p):
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features = max_p)
    X = cv.fit_transform(corpus).toarray()
    return X, cv


# In[ ]:


def tf_itf(corpus, max_p):
    from sklearn.feature_extraction.text import TfidfVectorizer 
    vectorizer = TfidfVectorizer(max_features = max_p)
    vectors = vectorizer.fit_transform(corpus)
    dense = vectors.todense()
    denselist = pd.DataFrame(dense).to_numpy()
    return denselist, vectorizer



# In[ ]:


def create_model(nodes_1, nodes_2, drop_1, drop_2):
    model = keras.models.Sequential()     
    model.add(Dense(nodes_1, activation='relu'))
    model.add(Dropout(drop_1))
    model.add(Dense(nodes_2, activation='relu'))
    model.add(Dropout(drop_2))
    model.add(Dense(8, activation='sigmoid'))   
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])   
    return model



# In[ ]:


def create_model_cnn(n_salida, n_oculto, em_input, em_dim, em_maxlen):
    from keras import layers
    model = Sequential()
    model.add(layers.Embedding(input_dim=em_input, 
                               output_dim=em_dim, 
                               input_length=em_maxlen))
    model.add(layers.GlobalMaxPool1D())
    model.add(layers.Dense(n_oculto, activation='relu'))
    model.add(layers.Dense(units=n_salida, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model



# In[ ]:

def spacy_lematizar(dataset, tipo_palabra):
    import spacy
    nlp = spacy.load('es_core_news_md')
    all_stopwords = stopwords.words('spanish')
    removeList=["no", "nunca"]
    all_stopwords = [e for e in all_stopwords if e not in removeList]
    corpus = []
    for i, value in dataset.items():
        review = str(html.unescape(dataset[i]))
        review = cleanhtml(review)
        review = re.sub(r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+", r"\1", normalize( "NFD", review), 0, re.I)
        doc = nlp(review)
        lemmas = [tok.lemma_.lower() for tok in doc if not tok in set(all_stopwords) and tok.pos_ in set(tipo_palabra) ]
        review = ' '.join(lemmas)
        corpus.append(review)
    return corpus
# In[ ]:

def spacy_steaming(dataset, tipo_palabra):
    import spacy
    nlp = spacy.load('es_core_news_md')
    all_stopwords = stopwords.words('spanish')
    spanishstemmer=SnowballStemmer("spanish")
    removeList=["no", "nunca"]
    all_stopwords = [e for e in all_stopwords if e not in removeList]
    corpus = []
    for i, value in dataset.items():
        review = str(html.unescape(dataset[i]))
        review = cleanhtml(review)
        review = re.sub(r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+", r"\1", normalize( "NFD", review), 0, re.I)
        doc = nlp(review)
        lemmas = [tok.lemma_.lower() for tok in doc if not tok in set(all_stopwords) and tok.pos_ in set(tipo_palabra) ]
        stems = [spanishstemmer.stem(token) for token in lemmas]
        review = ' '.join(stems)
        corpus.append(review)
    return corpus