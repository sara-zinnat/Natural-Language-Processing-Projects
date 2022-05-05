#Purpose of the program: To measure cosine similarity within and between the clusters
#Author: Sara Binte Zinnat
#Date: 18.03.2020
import numpy as np
import nltk
from nltk.corpus import brown
from gensim import corpora, models, similarities
from random import randint

#purpose: to fetch similarity between two sentences from matrix (using tf-idf and word2vec)
#input: s1, s2 (two sentences)
#return: cosine similarity value between s1 and s2 from matrix_tf_idf and matrix_word2vec
def similarity_sentence (s1, s2, matrix_tf_idf, matrix_word2vec):
	return matrix_tf_idf[s2][s1], matrix_word2vec[s2][s1]

#purpose: to calculate similarity between two documents (using tf-idf and word2vec)
#input: d1, d2 (two documents)
#return:
def similarity_document (d1, d2, withInCluster = False):
	d1 = [[w.lower() for w in s] for s in d1]
	d2 = [[w.lower() for w in s] for s in d2]

	if (withInCluster):
		corp = d1
	else:
		corp = []
		for s in d1:
			corp.append(s)
		for s in d2:
			corp.append(s)

	dictionary = corpora.Dictionary(corp)
	corpus = [dictionary.doc2bow(gen_doc) for gen_doc in d1]

	# TF_IDF representation
	tf_idf = models.TfidfModel(corpus)
	# MatrixSimilarity uses the cosine similarity
	sims = similarities.MatrixSimilarity(tf_idf[corpus], num_features=len(dictionary))
	query_doc_bow = [dictionary.doc2bow(gen_doc) for gen_doc in d2]
	query_doc_tf_idf = tf_idf[query_doc_bow]

	# Word2Vec representation
	model = models.Word2Vec(d1, min_count=1, size=20)
	termsim_index = models.WordEmbeddingSimilarityIndex(model.wv)
	similarity_matrix = similarities.SparseTermSimilarityMatrix(termsim_index, dictionary)
	docsim_index = similarities.SoftCosineSimilarity(corpus, similarity_matrix)
	sim_query_doc_w2v = docsim_index[query_doc_bow]

	# selecting random sentences sen1 and sen2 from d1 and d2 respectively for an example to see their cosine similarity
	sen1 = randint(0, len(d1)-1)
	sen2 = randint(0, len(d2)-1)

	print ('*'*60)
	print('************ Comparing Result ************')
	print ('* Both are ', len(d2), ' X ', len(d1),' similarity matrices')
	print ('* Here row denotes the sentences from the query document d2')
	print ('* And column denotes the sentences from the reference document d1')
	print ('* [row]X[column] denotes the cosine similarity value between sentences from d2 and d1 respectively')
	print ('*'*60)
	print ('********** TF-IDF representation **********')
	print (sims[query_doc_tf_idf])
	print ('The mean value of this similarity matrix (TF-IDF): ', np.mean(sims[query_doc_tf_idf]))
	print ('*'*60)
	print ('********** Word2Vec representation **********')
	print(sim_query_doc_w2v)
	print ('The mean value of this similarity matrix (Word2Vec): ', np.mean(sim_query_doc_w2v))
	print ('*'*60)
	print ('For an example:')
	print ('The cosine similarity between two random sentences, sen_no',sen1+1,' (from d1) and sen_no',sen2+1,' (from d2):')
	print ('TF-IDF: ', sims[query_doc_tf_idf][sen2][sen1], ' and Word2Vec: ', sim_query_doc_w2v[sen2][sen1])
	print ('*'*60)


def main ():
	genres=brown.categories()
	genres2 = genres

	# Computing the similarity within the cluster
	print ('!!! Computing the similarity within the cluster !!! STARTS')
	for gen in genres:
		print ('Computing the similarity within the cluster: ', gen)
		print ('Computing. . . Please Wait . . .')
		cluster = brown.sents(categories=gen)
		similarity_document(cluster, cluster, True)
	print ('!!! Computing the similarity within the cluster !!! ENDS')

	# Computing the similarity between clusters
	print ()
	print ()
	print ('!!! Computing the similarity between clusters !!! STARTS')
	for gen in genres:
		genres2.remove(gen)
		for gen2 in genres2:
			print (gen,', ',gen2)
			print ('Computing the similarity between clusters: (', gen, ', ', gen2, ')')
			print ('Computing. . . Please Wait . . .')
			cluster1 = brown.sents(categories=gen)
			cluster2 = brown.sents(categories=gen2)
			similarity_document(cluster1, cluster2)
	print ('!!! Computing the similarity between clusters !!! ENDS')

main()
