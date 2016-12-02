# Individual Programming Project - NLP - Spring 2016

# Text Categorization Using Naive-Bayes 
# By Frank Longueira

from nltk.tokenize import word_tokenize
from math import log
import string
from nltk.stem import PorterStemmer

def text_cat(train_input = 0, test_input = 0):

	# Additive Smoothing Parameter
	k = 0.056
	
	## Training
	
	# Read in training file names
	f_train = open(train_input, 'r')
	f_train_lines = f_train.read().splitlines()
	
	# Declare dictionary for storing word counts per category based on training
	d_word_categ = dict()
	
	# Declare dictionary for storing token count per category
	d_categ = dict()
	
	# Declare dictionary to count files per category to retrieve prior information
	d_priors = dict()
	
	# Declare stemmer
	stemmer = PorterStemmer()
		
	for line in f_train_lines:
		
		# Split line into file name & category
		file_cat = line.split()
		
		# Read in training file
		train_file = open(file_cat[0], 'r')
		
		# Tokenize training file
		train_file_tokenize = word_tokenize(train_file.read())
		
		# Category of this training file
		categ = file_cat[1]
		
		# Keep count of categories we are seeing
		if categ in d_priors:
			d_priors[categ] += 1.
		else:
			d_priors[categ] = 1.
		
		# Loop through words of this training file
		for token in train_file_tokenize:
			
			# Apply stemmer to token
			token = stemmer.stem(token)

			# Count number of times this word appears in this category
			if (token,categ) in d_word_categ:
				d_word_categ[(token,categ)] += 1.
			else:
				d_word_categ[(token,categ)] = 1.
				
			# Count number of tokens in this category 
			if categ in d_categ:
				d_categ[categ] += 1.
			else:
				d_categ[categ] = 1.
	
	# Store total number of training files
	num_train_files = sum(d_priors.values())
	
	# Store unique category names in list
	unique_categ = d_categ.keys()
	
	
	## Testing
	
	# Read in test file names
	f_test = open(test_input, 'r')
	f_test_lines = f_test.read().splitlines()
	
	predictions = []
	
	# Loop through each test file
	for line in f_test_lines:
		test_file = open(line,'r')
		
		# Tokenize the test file
		test_file_tokenize = word_tokenize(test_file.read())
		
		# Declare a dictionary for computational purposes
		test_d = dict()
			
		# Find vocaculary size of test file
		for test_token in test_file_tokenize:
			
			# Apply stemmer to token
			test_token = stemmer.stem(test_token)
			
			if test_token in list(string.punctuation):
				pass
			else:
				if test_token in test_d:
					test_d[test_token] += 1.
				else:
					test_d[test_token] = 1.
		
		# Vocabulary size of given test file
		vocab_size = len(test_d)
		
		# Declare dictionary for storing conditional category log probability for test file
		category_log_probabilities = dict()
		
		for category in unique_categ:
			
			# Initialize total conditional probability of article being in this category
			total_log_categ_prob = 0.
			
			# Prior probability of category based on training set
			prior = d_priors[category]/num_train_files
			
			# Normalization coefficient for additive smoothing
			normalizer = d_categ[category] + k*vocab_size

			# Calculate conditional category probabilities
			for word, count in test_d.iteritems():
				if (word, category) in d_word_categ:
					count_word_given_category = d_word_categ[(word, category)] + k
				else:
					count_word_given_category  = k


				log_categ_prob = count*log(count_word_given_category/normalizer)
				total_log_categ_prob += log_categ_prob
				
			category_log_probabilities[category] = total_log_categ_prob + log(prior)
		
		# Make decision based on MLE
		decision = max(category_log_probabilities, key = category_log_probabilities.get)
		
		# Construct string to write for each line of output file & append to list
		str = line + ' ' + decision + '\n'
		predictions.append(str)
		
	# Ask user for output file name
	output_file_name = raw_input('Enter a name for the category prediction output file: ')
	
	output_file = open(output_file_name, 'w')
	for line in predictions:
		output_file.write(line)
		
	output_file.close()
	
	return
					
		
train_input = raw_input('Please specify the name of the file containing the list of labeled training documents: ')
test_input  = raw_input('Please specify the name of the file containing the list of unlabeled test documents: ') 


text_cat(train_input, test_input)
