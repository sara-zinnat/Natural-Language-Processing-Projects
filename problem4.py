#Assignment 2 (CPSC 5310)
#Problem-04
#Purpose of the program: to classify text using Naive Bayes classifier on Brown corpus 
#Author: Sara Binte Zinnat
#ID:001217884
#Date: 05.02.2020

import nltk
import re
import numpy as np
from nltk.corpus import brown
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer   # word to vector
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics

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

	countVect = CountVectorizer()
	counts = countVect.fit_transform(text)
	tfidf_Transformer = TfidfTransformer()
	featuresMatrix = tfidf_Transformer.fit_transform(counts)

	le = preprocessing.LabelEncoder()
	labels = le.fit_transform(category)

	return featuresMatrix, labels


#main function
def main():
	genres=brown.categories() 
	
	featuresMatrix, labels = extractFeatures(genres)

	print ('Feature Matrix Size: ' + str(featuresMatrix.shape))
	print ('Labels Size: ' + str(labels.shape))

	X_train, X_test, y_train, y_test = train_test_split(featuresMatrix, labels, test_size=0.30)
	model = MultinomialNB()
	model.fit(X_train,y_train)
	result = model.predict(X_test)

	print('Accuracy: ' + str(metrics.accuracy_score(y_test, result)))
	print('Precision: ' + str(metrics.precision_score(y_test, result,average='macro',labels=np.unique(result))))
	print('Recall: ' + str(metrics.recall_score(y_test, result, average='macro',labels=np.unique(result))))

main()
