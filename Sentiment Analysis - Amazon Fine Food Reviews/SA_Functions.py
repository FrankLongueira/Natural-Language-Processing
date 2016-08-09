import sqlite3
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import string
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import pickle
import operator

# Function for loading in Harvard's positive/negative word lexicons
def load_lexicons():
	"""
	This function loads in Harvard's postivie/negative word lexicons.
	"""
	positive_words = np.genfromtxt('Positive_&_Negative_Words.csv', skip_header = 1, usecols = (0, ), delimiter = ',', dtype = 'str')
	negative_words = np.genfromtxt('Positive_&_Negative_Words.csv', skip_header = 1, usecols = (1, ), delimiter = ',', dtype = 'str') 
	return positive_words, negative_words
	
# Function for mapping 1-5 ratings to positive/negative labels
def partition(x):
	"""
	This function rates reviews less than 3 negative, and above 3 positive.
	"""
	if x < 3:
		return (-1)
	else:
		return (1)

# Function to tokenize & lowercase reviews
def tokenize_lower(text):
	"""
	This function tokenizes a review & makes it lower case.
	"""
	tokens = word_tokenize(text.lower())
	return tokens

# Stemming function
def stemming(tokens):
	"""
	This function stems a tokenized review.
	"""
	stemmer = PorterStemmer()
	#s1 = dict((k,1) for k in stopwords.words('english'))
	#s2 = dict((k,1) for k in string.punctuation)
	#tokens_stemmed = [stemmer.stem(i) for i in tokens if i not in s1 and i not in s2]
	tokens_stemmed = [stemmer.stem(i) for i in tokens]
	return tokens_stemmed

# Count top words
def counting(reviews_preprocessed, scores_posneg):
	positive_words = dict()
	negative_words = dict()
	position = 0
	for review in reviews_preprocessed:
		if scores_posneg[position] == 1:
			tokens = word_tokenize(review)
			for token in tokens:
				if token in positive_words:
					positive_words[token] += 1
				else:
					positive_words[token] = 1
		if scores_posneg[position] == -1:
			tokens = word_tokenize(review)
			for token in tokens:
				if token in negative_words:
					negative_words[token] += 1
				else:
					negative_words[token] = 1
		position += 1
	return positive_words, negative_words

# Python object saver/load functions
def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

# Pre-processing function for reviews
def preprocess(reviews, score):
	"""
	This function preprocesses the reviews by lowercasing & stemming.
	This function rates reviews less than 3 negative (-1), and above 3 positive (1).
	This function extracts the lengths of preprocessed reviews.
	"""
	reviews_stemmed = []
	lengths = []
	print('Preprocessing...')
	score = score.map(partition)
	for review in reviews:
		tokens = tokenize_lower(review)
		tokens_stemmed = stemming(tokens)
		reviews_stemmed.append((' '.join(tokens_stemmed)))
		lengths.append(len(tokens_stemmed))
	print('Preprocessing Finished!\n')
	return reviews_stemmed, score, lengths

# Keep only top words in pre-processed reviews
def keep_top(reviews_preprocessed, positive_top_words, negative_top_words):
	reviews_preprocessed_topwords = []
	for review in reviews_preprocessed:
		tokens = word_tokenize(review)
		new_tokens = []
		for token in tokens:
			if token in positive_top_words:
				new_tokens.append(token)
			elif token in negative_top_words:
				new_tokens.append(token)
			else:
				continue
		new_review = ' '.join(new_tokens)
		reviews_preprocessed_topwords.append(new_review)
	return reviews_preprocessed_topwords

# Sort dictionaries from lowest to highest values
def topwords_sorter(positive_words, negative_words, top_count):
	positive_top_words = dict(sorted(positive_words.iteritems(), 	key=operator.itemgetter(1), reverse=True)[:top_count])
	negative_top_words = dict(sorted(negative_words.iteritems(), key=operator.itemgetter(1), reverse=True)[:top_count])
	return positive_top_words, negative_top_words

# Tf-IDf Weighting Scheme
def tfidf_weights(preprocessed_training_reviews, preprocessed_test_reviews):
	"""
	This function applies the Tf-IDf weighting scheme to the training & test documents.
	"""
	print('Creating Tf-IDf weight vectors...')
	from sklearn.feature_extraction.text import CountVectorizer
	from sklearn.feature_extraction.text import TfidfTransformer
	
	count_vect = CountVectorizer(analyzer='word', ngram_range=(2, 2))
	tfidf_trans = TfidfTransformer()

	training_word_counts = count_vect.fit_transform(preprocessed_training_reviews)
	train_tfidf = tfidf_trans.fit_transform(training_word_counts)
	
	test_word_counts = count_vect.transform(preprocessed_test_reviews)
	test_tfidf = tfidf_trans.transform(test_word_counts)
	print('Finished creating Tf-IDf weight vectors!\n')
	return train_tfidf, test_tfidf

## Model Functions

# Logistic Regression Model
def LogReg(train_review_features, train_review_labels, test_review_features):
	"""
	This function applies a logistic regression model for classification.
	"""
	print('Applying Logistic Regression...')
	from sklearn import linear_model
	logreg = linear_model.LogisticRegression(C=1e5, class_weight = 'auto')
	logreg.fit(train_review_features, train_review_labels)
	prediction = logreg.predict(test_review_features)
	print('Finished Logistic Regression!\n')
	return prediction

# Multinomial Naive Bayes Model
def MultiNB(train_review_features, train_review_labels, test_review_features):
	"""
	This function applies a multinomial naive bayes model for classification.
	"""
	print('Applying Multinomial Naive Bayes...')
	from sklearn.naive_bayes import MultinomialNB
	model = MultinomialNB(fit_prior=True, alpha = 0.01).fit(train_review_features, 	train_review_labels)
	prediction = model.predict(test_review_features)
	print("Finished Multinomial Naive Bayes!\n")
	return prediction

# Display confusion matrices for model
def confusion(model_prediction, test_review_labels):
	"""
	This function computes confusion matrix for each model applied during testing.
	"""
	test_review_labels = np.array(test_review_labels).astype('str')
 	predicted = np.array(model_prediction).astype('str')
 	print(metrics.classification_report(test_review_labels, predicted))
 	print('\n')
