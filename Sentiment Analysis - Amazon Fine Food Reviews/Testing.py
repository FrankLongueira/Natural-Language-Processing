import sqlite3
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import string
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import SA_Functions as saf
import pickle
import operator

con = sqlite3.connect('data/database.sqlite')
q = pd.read_sql_query("""
	SELECT Score, Summary, Text
 	FROM Reviews
 	WHERE Score != 3
 	""", con)
 	
reviews = q['Text']
score = q['Score']

num_reviews = len(reviews)
num_train_reviews = int(0.75*num_reviews)

## The reviews are preprocessed (as of now: lowercased, stemmed)
#reviews_preprocessed, scores_posneg, lengths = saf.preprocess(reviews, score)
#saf.save_obj(reviews_preprocessed, 'reviews_preprocessed')
#saf.save_obj(scores_posneg, 'scores_posneg')
#saf.save_obj(lengths, 'lengths')
reviews_preprocessed = saf.load_obj('reviews_preprocessed')
scores_posneg = saf.load_obj('scores_posneg')
lengths = saf.load_obj('lengths')

## Grab lowercased & stemmed training reviews
#train_reviews_preprocessed = reviews_preprocessed[0:num_train_reviews]
#train_scores_posneg = scores_posneg[0:num_train_reviews]


## Count words in lower-cased & stemmed training reviews to establish positive/negative word dicts

#positive_words, negative_words = saf.counting(train_reviews_preprocessed, train_scores_posneg)
#saf.save_obj(positive_words, 'positive_words')
#saf.save_obj(negative_words, 'negative_words')
#positive_words = saf.load_obj('positive_words')
#negative_words = saf.load_obj('negative_words')

## Sort and get only top <#> words from each positive/negative dictionary
#top_count = 10000
#positive_top_words, negative_top_words = saf.topwords_sorter(positive_words, negative_words, top_count)

## Process the reviews to only take top < # > words from each positive/negative dictionary
#reviews_preprocessed_topwords = saf.keep_top(reviews_preprocessed, positive_top_words, negative_top_words)
#saf.save_obj(reviews_preprocessed_topwords, 'reviews_preprocessed_topwords')
#reviews_preprocessed_topwords = saf.load_obj('reviews_preprocessed_topwords')

reviews_train = reviews_preprocessed[0:num_train_reviews]
reviews_train_labels = scores_posneg[0:num_train_reviews]

reviews_test = reviews_preprocessed[num_train_reviews::]
reviews_test_labels = scores_posneg[num_train_reviews::]


# Apply Tf-IDf weighting scheme to the training reviews & test reviews
train_tfidf, test_tfidf = saf.tfidf_weights(reviews_train, reviews_test)
#saf.save_obj(train_tfidf, 'train_tfidf')
#saf.save_obj(test_tfidf, 'test_tfidf')
#train_tfidf = saf.load_obj('train_tfidf')
#test_tfidf = saf.load_obj('test_tfidf')

# Apply Logistic Regression model for classification on test set
prediction_LR = saf.LogReg(train_tfidf, reviews_train_labels, test_tfidf)
# Display confusion matrix for Logistic Regression
saf.confusion(prediction_LR, reviews_test_labels)

# Apply Multinomial Naive Bayes model for classification on test set
prediction_MNB = saf.MultiNB(train_tfidf, reviews_train_labels, test_tfidf)
# Display confusion matrix for Multinomial Naive Bayes
saf.confusion(prediction_MNB, reviews_test_labels)
