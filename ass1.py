#Purpose of the program: To count word tokens and word types from each genre/category and the vocabulary size for the whole corpus 
#Author: Sara Binte Zinnat
#Date: 24.01.2020

import nltk
import re
from nltk.corpus import brown
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

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

#purpose: counting tokens
#input: sentences
#return: numberOfWordTokens
def wordCount(sentences):
	numberOfWordTokens = 0
	for sent in sentences:
		numberOfWordTokens += len(sent)
	return numberOfWordTokens

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

	return filteredSentences, numberOfStopWords

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

#purpose: applying stemming
#input: sentences
#return: filteredSentences
def stemming (sentences):
	porter = PorterStemmer()

	filteredSentences = []

	for sent in sentences:
		filteredSentence = []
		for token in sent:
			filteredSentence.append(porter.stem(token))
		filteredSentences.append(filteredSentence)

	return filteredSentences

#purpose: counting word type
#input: sentences
#return: number of word type
def wordTypeCount (sentences):
	uniqueWord = []
	for sent in sentences:
		for token in sent:
			if token not in uniqueWord:
				uniqueWord.append(token)

	return len(uniqueWord)

#main function
def main():
	genres=brown.categories()
	vocabularySize = 0
	vocabulary = {}

	for gen in genres:
		print('For ',gen,' genre:')
		sentences = brown.sents(categories=gen)

		# removing special character
		sentences = removeSpecialCharacter(sentences)
		print('Number of word tokens (with stopwords): ', wordCount(sentences), ' and number of word type (with stopwords): ', wordTypeCount(sentences))
		sentences, numberOfStopWords = removeStopWords(sentences)
		print('Number of word tokens (without stopwords): ', wordCount(sentences), '; number of removed stopwords: ',numberOfStopWords, ' and number of word type (without stopwords): ', wordTypeCount(sentences))
		sentences = lemmatize (sentences)
		print('Lemmatization done!', ' and number of word type: ', wordTypeCount(sentences))
		sentences = stemming (sentences)
		print('Stemming done!', ' and number of word type: ', wordTypeCount(sentences))

		for sent in sentences:
			for token in sent:
				if token in vocabulary:
					vocabulary[token] += 1
				else:
					vocabulary[token] = 1

	print ('vocabulary size of whole corpus: ', len(vocabulary.keys()))

main()
