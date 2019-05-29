import nltk
import os
import json
from gensim.models import Word2Vec, KeyedVectors, doc2vec
import numpy as np
import pickle
from math import log, pow
from conf import *
import treetaggerwrapper
import time

corpora = corpora[4:5]

weight_file = open('POS_tagging/pos_weights', 'r')
weight_dict = {}
for line in weight_file:
	line = line.strip().split('", ')
	weight_dict[line[0][3:]] = float(line[1][:-2])

for corpus in corpora:

	print('Corpus: {}...'.format(corpus))

	newdir = 'Cross-Language-Dataset/dataset/{}/{}'.format(gran, corpus)
	vecdir = 'Cross-Language-Dataset/sumvec/{}/{}'.format(gran, corpus)
	idfvecdir = 'Cross-Language-Dataset/idfvec/{}/{}'.format(gran, corpus)
	posvecdir = 'Cross-Language-Dataset/posvec/{}/{}'.format(gran, corpus)
	posidfvecdir = 'Cross-Language-Dataset/posidfvec/{}/{}'.format(gran, corpus)

	for dir in [vecdir, idfvecdir, posvecdir, posidfvecdir]:
		if not os.path.isdir(dir):
			os.makedirs(dir)

	for lang in os.listdir(newdir):
		if lang in langs:
			for dir in [vecdir, idfvecdir, posvecdir, posidfvecdir]:
				if not os.path.isdir('{}/{}'.format(dir, lang)):
					os.makedirs('{}/{}'.format(dir, lang))

			print('Language: {}...'.format(lang))
			t0 = time.time()

			# create vec and idf_dict files
			w2v = KeyedVectors.load_word2vec_format('w2v/{}-vectors.txt'.format(lang), binary=False)
			idf_dict = {}

			for filename in os.listdir('/'.join([newdir, lang])):
				if filename != '.DS_Store':
					file = open('/'.join([newdir, lang, filename]), 'r')
					words = file.read().split()
					uniq_words = set(words)
					for word in uniq_words:
						if word not in idf_dict:
							idf_dict[word] = 0
						idf_dict[word] += 1
			total = sum([idf_dict[word] for word in idf_dict])
			idf_dict = {word: (log(total) - log(idf_dict[word])) for word in idf_dict}
			idf_dict_file = open('{}/{}/idf_dict_{}'.format(idfvecdir, lang, lang), 'wb')
			pickle.dump(idf_dict, idf_dict_file)
			idf_dict_file.close()

			t1 = time.time()
			print('Finished creating idf_dict files, time elapsed: {}'.format(t1-t0))

			# create docvec files
			idf_dict_file = open('{}/{}/idf_dict_{}'.format(idfvecdir, lang, lang), 'rb')
			idf_dict = pickle.load(idf_dict_file)
			tagger = treetaggerwrapper.TreeTagger(TAGLANG=lang, TAGDIR='/Users/no_blank/treetagger')
			w2v = KeyedVectors.load_word2vec_format('w2v/{}-vectors.txt'.format(lang), binary=False)

			univ_file = open('POS_tagging/universal_pair/{}'.format(lang), 'r')
			univ_dict = {}
			for line in univ_file:
				tag1, tag2 = line.split()
				univ_dict[tag1] = tag2

			for filename in os.listdir('/'.join([newdir, lang])):
				if filename != '.DS_Store':
					file = open('/'.join([newdir, lang, filename]), 'r')
					words = file.read().split()

					vecs = np.array([w2v[word] if word in w2v else np.zeros([100, ]) for word in words])

					tags = tagger.tag_text(words, tagonly=True)
					tags = [univ_dict[tag.split('\t')[1]] for tag in tags]

					sumvec = np.sum(vecs, 0)
					sumvec_file = open('{}/{}/{}'.format(vecdir, lang, filename), 'wb')
					pickle.dump(sumvec, sumvec_file)
					sumvec_file.close()

					idfvec = np.sum(np.array([vecs[i] * idf_dict[words[i]] for i in range(len(words))]), 0)
					idfvec_file = open('{}/{}/{}'.format(idfvecdir, lang, filename), 'wb')
					pickle.dump(idfvec, idfvec_file)
					idfvec_file.close()

					posvec = np.sum(np.array([vecs[i] * weight_dict[tags[i]] for i in range(len(words))]), 0)
					posvec_file = open('{}/{}/{}'.format(posvecdir, lang, filename), 'wb')
					pickle.dump(posvec, posvec_file)
					posvec_file.close()

					posidfvec = np.sum(np.array([vecs[i] * pow(weight_dict[tags[i]], 0.5) * pow(idf_dict[words[i]], 0.5) for i in range(len(words))]), 0)
					posidfvec_file = open('{}/{}/{}'.format(posidfvecdir, lang, filename), 'wb')
					pickle.dump(posidfvec, posidfvec_file)
					posidfvec_file.close()

			idf_dict_file.close()
			t2 = time.time()
			print('Finished creating idfvec, posvec, and posidfvec files, time elapsed: {}'.format(t2 - t1))







