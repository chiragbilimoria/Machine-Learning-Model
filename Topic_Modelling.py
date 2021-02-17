# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# working on Consumer Complaint dataset taken from CFPB website where consumer write about the issue they face from banks and finincial services

# %%
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition
import matplotlib.pyplot as plt
import numpy as np
import re
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split


# %%
nltk.download('punkt')

# %%

df=pd.read_csv('https://github.com/srivatsan88/YouTubeLI/blob/master/dataset/consumer_compliants.zip?raw=true', compression='zip', sep=',', quotechar='"')

# %%
df

# %%
#checking types of products for which complaints are lodge
df['Product'].value_counts()

# %%
#Getting name of companies against which complaints are lodge
df['Company'].value_counts()

# %%
#Creating separate dataset just to work on few selected columns
complaints_df=df[['Consumer complaint narrative','Product','Company']].rename(columns={'Consumer complaint narrative':'complaints'})

# %%
#changing pandas setting to get full value of columns
pd.set_option('display.max_colwidth', -1)
complaints_df

# %%
X_train, X_hold = train_test_split(complaints_df, test_size=0.3, random_state=111)

# %%
X_train['Product'].value_counts()

# %%
stemmer = PorterStemmer()


# %%
#creating method to tokenize text and removing words which are less than len 3 and words starting with X
def tokenize(text):
   tokens = [word for word in nltk.word_tokenize(text) if (len(word) > 3 and len(word.strip('Xx/')) > 2) ] 
   #stems = [stemmer.stem(item) for item in tokens]
   return tokens


# %%
'''Transforming text to vectors for our lda model here I am using my tokenize method to in tokenizer rather than using in build fuction of tfidf
using stopword as English I have used max_df as 75%  basically ignoring all the terms having document frequency greater than 75% 
and same applies for min df where if words occurs in less than 50 documents ignore those words.use_idf is false which means we are just using Tf and not its inverse. Norm is none hence it will behave as count vectorizer'''

vectorizer_tf = TfidfVectorizer(tokenizer=tokenize, stop_words='english', max_df=0.75, min_df=50, max_features=10000, use_idf=False, norm=None)
tf_vectors = vectorizer_tf.fit_transform(X_train.complaints)

# %%
# getting feature names
vectorizer_tf.get_feature_names()

# %%
'''Building LDA model to generate topics here I have use n_components =6 since we have 6 products so trying to see if it generate topic based on products
max_iter is 3 means it will run for 3 iterations we can trying highers values as well. Learning method is online since it is the recommended oneand with this 
I will use learning offset which downweight early iterations n jobs is set -1 to use all the processors and random state to reproduce the experiments'''
lda = decomposition.LatentDirichletAllocation(n_components=6, max_iter=3, learning_method='online', learning_offset=50, n_jobs=-1, random_state=111)

W1 = lda.fit_transform(tf_vectors)
H1 = lda.components_

# %%
# selecting top 15 words for each topics to see what customers are talking about
num_words=15

vocab = np.array(vectorizer_tf.get_feature_names())

top_words = lambda t: [vocab[i] for i in np.argsort(t)[:-num_words-1:-1]]
topic_words = ([top_words(t) for t in H1])
topics = [' '.join(t) for t in topic_words]

# %%
topics

# %%
#Getting values of topics for each row and selecting the highesh score of topic for row as Dominant topic
colnames = ["Topic" + str(i) for i in range(lda.n_components)]
docnames = ["Doc" + str(i) for i in range(len(X_train.complaints))]
df_doc_topic = pd.DataFrame(np.round(W1, 2), columns=colnames, index=docnames)
significant_topic = np.argmax(df_doc_topic.values, axis=1)
df_doc_topic['dominant_topic'] = significant_topic

# %%
#displaying values of each topics per row and seeing which is the dominant topic for the given row
df_doc_topic

# %%
#running lda for test data
WHold = lda.transform(vectorizer_tf.transform(X_hold.complaints[:5]))

# %%
#Getting values of topics for each row and selecting the highesh score of topic for row as Dominant topic for test data
colnames = ["Topic" + str(i) for i in range(lda.n_components)]
docnames = ["Doc" + str(i) for i in range(len(X_hold.complaints[:5]))]
df_doc_topic = pd.DataFrame(np.round(WHold, 2), columns=colnames, index=docnames)
significant_topic = np.argmax(df_doc_topic.values, axis=1)
df_doc_topic['dominant_topic'] = significant_topic

# %%
#displaying values of each topics per row and seeing which is the dominant topic for the given row for train data
df_doc_topic

# %%
