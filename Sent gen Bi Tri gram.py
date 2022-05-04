#Purpose of the program: To generate random sentences using Bi-gram and Tri-gram
#Author: Sara Binte Zinnat
#Date: 04.02.2020

import nltk
import re
import random
from nltk.corpus import brown
from nltk.corpus import stopwords

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

#purpose: to tokenize the entire brown corpus incorporating <s> and </s> for starting and ending of each sentences
#input: genres
#return: tokenized corpus
def tokenize (genres):
	corpus = []

	for gen in genres:
		sentences = brown.sents(categories=gen)

		# removing special character
		sentences = removeSpecialCharacter(sentences)
		sentences = removeStopWords(sentences)

		for sent in sentences:
			sent.insert(0,'<s>')
			sent.append('</s>')

		corpus.extend(sentences)
	return corpus

#purpose: to make an uni-gram table
#input: corpus
#return: uni-gram table
def uniGramTable(corpus):
	ugTable = {}

	for sent in corpus:
		for token in sent:
			if token in ugTable:
				ugTable[token] += 1
			else:
				ugTable[token] = 1

	return ugTable, len(ugTable)

#purpose: to make a bi-gram table
#input: corpus
#return: bi-gram table
def biGramTable(corpus):
	bgTable = {}

	bgLength = 0

	for sent in corpus:
		for i in range(0,len(sent)-1):
			if sent[i] in bgTable:
				if sent[i+1] in bgTable[sent[i]]:
					bgTable[sent[i]][sent[i+1]] += 1
				else:
					bgTable[sent[i]][sent[i+1]] = 1
					bgLength += 1
			else:
				bgTable[sent[i]] = {}
				bgTable[sent[i]][sent[i+1]] = 1
				bgLength += 1

	return bgTable, bgLength

#purpose: to make a tri-gram table
#input: corpus
#return: tri-gram table
def triGramTable(corpus):
	tgTable = {}

	tgLength = 0

	for sent in corpus:
		for i in range(0,len(sent)-2):
			if sent[i] in tgTable:
				if sent[i+1] in tgTable[sent[i]]:
					if sent[i+2] in tgTable[sent[i]][sent[i+1]]:
						tgTable[sent[i]][sent[i+1]][sent[i+2]] += 1
					else:
						tgTable[sent[i]][sent[i+1]][sent[i+2]] = 1
				else:
					tgTable[sent[i]][sent[i+1]] = {}
					tgTable[sent[i]][sent[i+1]][sent[i+2]] = 1
					tgLength += 1
			else:
				tgTable[sent[i]] = {}
				tgTable[sent[i]][sent[i+1]] = {}
				tgTable[sent[i]][sent[i+1]][sent[i+2]] = 1
				tgLength += 1

	return tgTable, tgLength

#purpose: to generate and print a random sentence using bi-gram
#input: uni-gram table and its length; bi-gram table and its length; first random token of a sentence; maximum length of a sentence
#return: a random sentence; probability of that sentence
def randomSenBigram(ugTable, ugLength, bgTable, bgLength, firstToken, maxLength):
	sen = ''
	leng = 0

	probability = 1.0

	token = firstToken
	sen = sen + token

	while token != '</s>' and leng < maxLength:
		count = 0
		tempToken = token
		for nextToken in list(bgTable[token]):
			if count < bgTable[token][nextToken] and nextToken != '</s>':
				count = bgTable[token][nextToken]
				tempToken = nextToken
		if tempToken != token:
			probability = probability * float(bgTable[token][tempToken])/ugTable[token]
			token = tempToken
			sen = sen + ' ' + token
			leng += 1
		else:
			break

	return sen, probability

#purpose: to generate and print a random sentence using tri-gram
#input: bi-gram table and its length; tri-gram table and its length; first random token of a sentence; maximum length of a sentence
#return: a random sentence; probability of that sentence
def randomSenTrigram(bgTable, bgLength, tgTable, tgLength, firstToken, maxLength):
	sen = ''
	leng = 0

	probability = 1.0

	token1 = '<s>'
	token2 = firstToken
	sen = sen + token2

	while token2 != '</s>' and leng < maxLength:
		count = 0
		tempToken = token2
		for nextToken in list(tgTable[token1][token2]):
			if count < tgTable[token1][token2][nextToken] and nextToken != '</s>':
				count = tgTable[token1][token2][nextToken]
				tempToken = nextToken
		if tempToken != token2:
			probability = probability * float(tgTable[token1][token2][tempToken])/bgTable[token1][token2]
			token1 = token2
			token2 = tempToken
			sen = sen + ' ' + token2
			leng += 1
		else:
			break

	return sen, probability


#main function
def main():
	genres=brown.categories()
	corpus = tokenize(genres)

	ugTable, ugLength = uniGramTable(corpus)
	bgTable, bgLength = biGramTable(corpus)
	tgTable, tgLength = triGramTable(corpus)

	numberOfIteration = 0
	print('************************')
	print('Please iterate more steps to see better random sentences!!')
	print('************************\n\n')
	stat = 1
	while(stat == 1):
		firstToken = random.choice(list(bgTable['<s>'])) #randomly selects the first word of a sentence
		numberOfIteration += 1
		print('\nIteration:' + str(numberOfIteration) + '; Here the first random token: ' + firstToken)
		print('Using Bi-gram:')
		sentence, probability = randomSenBigram(ugTable, ugLength, bgTable, bgLength, firstToken, 10)
		print(sentence)
		print('Probability of the sentence: ' + str(probability)+'\n')

		print('Using Tri-gram:')
		sentence, probability = randomSenTrigram(bgTable, bgLength, tgTable, tgLength, firstToken, 10)
		print(sentence)
		print('Probability of the sentence: ' + str(probability)+'\n')

		stat = int(input("Enter '1': to generate next random sentence or Enter '0' to exit: "))
		print('------------------------------------------')
main()
