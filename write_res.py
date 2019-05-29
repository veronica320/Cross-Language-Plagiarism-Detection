from conf import *
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
import csv
from pprint import pprint
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for alpha in alphas:
	if alpha != 0.5:
		weightings = ['posidf']
	else:
		weightings = ['sum', 'pos', 'idf', 'posidf']
	for src_l, trg_l in [['en', 'fr'], ['fr', 'en']]:
		for preprocess in ['fine', 'coarse']:

			c_pfm_fn = 'pfm/cpfm_{}_{}_{}*{}_{}_{}_{}'.format(src_l, trg_l, maskshape[0], maskshape[1], preprocess,
			                                                  sim_metrics, alpha)

			rec_fn = 'record/rec_{}_{}_{}*{}_{}_{}_{}.csv'.format(src_l, trg_l, maskshape[0], maskshape[1], preprocess,
			                                                      sim_metrics, alpha)
			try:
				c_pfm_file = open(c_pfm_fn, 'rb')
			except FileNotFoundError:
				continue
			c_pfm = pickle.load(c_pfm_file)


			# c_pfm['overall'] = {}
			# for weighting in weightings:
			# 	overall_pfm = []
			# 	for corpus in c_pfm:
			# 		if corpus != 'overall':
			# 			overall_pfm += c_pfm[corpus][weighting]
			# 	c_pfm['overall'][weighting] = '{:.2f}±{:.3f}'.format(np.average(overall_pfm), np.var(overall_pfm))
			# c_pfm_file.close()
			# c_pfm_file = open(c_pfm_fn, 'wb')
			# pickle.dump(c_pfm, c_pfm_file)


			w_pfm = {}
			for weighting in weightings:
				w_pfm[weighting] = []
				for corpus in corpora:
					w_pfm[weighting].append('{:.2f}±{:.3f}'.format(np.average(c_pfm[corpus][weighting]), np.var(c_pfm[corpus][weighting])))
				w_pfm[weighting].append(c_pfm['overall'][weighting])


			with open(rec_fn, 'w') as csv_file:
				writer = csv.writer(csv_file)
				writer.writerow(['method']+corpora+['overall'])
				for key, value in w_pfm.items():
					writer.writerow([key]+value)


