#Purpose of the program: to classify text using Logistic regression and Multi-layer neural network on Brown corpus
#Author: Sara Binte Zinnat
#Date: 29.03.2020

import re
import numpy as np
from nltk.corpus import brown
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression # Single layer perceptron
from sklearn.neural_network import MLPClassifier # Multi layer perceptron
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer   # word to vector
from sklearn import metrics

import multiprocessing
import time

#purpose: removing specialcharacter and lower case conversion
#input: sentences
#return: cleanSentences
def removeSpecialCharacter(sentences):
	cleanSentences = []

	for sent in sentences:
		cleanSent = []
		for token in sent:
			if (re.match('[a-zA-Z]+', token)):
				cleanSent.append(token.lower())
		cleanSentences.append(cleanSent)

	return cleanSentences

#purpose: removing stpwords
#input: sentences
#return: filteredSentences, numberOfStopWords
def removeStopWords (sentences):
	stop_words = set(stopwords.words('english'))

	numberOfStopWords = 0
	filteredSentences = []

	for sent in sentences:
		filteredSentence = []
		for token in sent:
			if token not in stop_words:
				filteredSentence.append(token)
			else:
				numberOfStopWords += 1
		filteredSentences.append(filteredSentence)

	return filteredSentences

#purpose: applying lemmatization
#input: sentences
#return: filteredSentences
def lemmatize (sentences):
	lemmatizer = WordNetLemmatizer()

	filteredSentences = []

	for sent in sentences:
		filteredSentence = []
		for token in sent:
			filteredSentence.append(lemmatizer.lemmatize(token))
		filteredSentences.append(filteredSentence)

	return filteredSentences


#purpose: generates features matrix and labels
#input: genres
#return: featuresMatrix and labels
def extractFeatures(genres):
	text = []
	category = []
	for gen in genres:
		sentences = brown.sents(categories=gen)
		sentences = removeSpecialCharacter(sentences)
		sentences = removeStopWords(sentences)
		sentences = lemmatize (sentences)

		for sent in sentences:
			t = ''
			for word in sent:
				t = t + word + ' '
			text.append(t)
			category.append(gen)

	word2vec = CountVectorizer()
	featuresMatrix = word2vec.fit_transform(text) # converts words to vector
	print ('Word2Vec Done!!!')

	le = preprocessing.LabelEncoder()
	labels = le.fit_transform(category)

	return featuresMatrix, labels


#main function
def main():
	genres=brown.categories()

	featuresMatrix, labels = extractFeatures(genres)

	print ('Feature Matrix Size: ' + str(featuresMatrix.shape))
	print ('Labels Size: ' + str(labels.shape))

	X_train, X_test, y_train, y_test = train_test_split(featuresMatrix, labels, test_size=0.10) # training 90% and testing 10%

	print()
	print('a) Using logistic regression::')
	print('Computing... Please wait...')
	model1 = LogisticRegression(random_state=0,solver='lbfgs', max_iter=10000, multi_class='auto')
	model1.fit(X_train,y_train)
	result1 = model1.predict(X_test)

	print('Accuracy: ' + str(metrics.accuracy_score(y_test, result1)))
	print('Precision: ' + str(metrics.precision_score(y_test, result1,average='macro',labels=np.unique(result1))))
	print('Recall: ' + str(metrics.recall_score(y_test, result1, average='macro',labels=np.unique(result1))))

	print()
	print('b) Using Multi-layer neural network::')
	print('Computing... Please wait...')
	print('Decreasing the tolerance value (tol=0.0001) will give better result but will increase the Computing time')
	model2 = MLPClassifier(activation='relu', solver='adam', alpha=0.00001, learning_rate='constant', learning_rate_init=0.001, max_iter=10000, tol=0.05, hidden_layer_sizes=(3,3,3)) # 3 Hidden layers with 3 hidden units
	model2.fit(X_train,y_train)
	result2 = model2.predict(X_test)

	print('Accuracy: ' + str(metrics.accuracy_score(y_test, result2)))
	print('Precision: ' + str(metrics.precision_score(y_test, result2,average='macro',labels=np.unique(result2))))
	print('Recall: ' + str(metrics.recall_score(y_test, result2, average='macro',labels=np.unique(result2))))



main()
