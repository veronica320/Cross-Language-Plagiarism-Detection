from conf import *
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
import csv
import matplotlib.pyplot as plt

weightings = ['posidf']

alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for preprocess in preprocesses[:1]:
	for src_l, trg_l in [['en', 'fr'], ['fr', 'en']]:
		plot_fn = 'plot/plot_{}_{}_{}*{}_{}'.format(src_l, trg_l, maskshape[0], maskshape[1], preprocess)
		series = {corpus:[] for corpus in corpora}
		series['overall'] = []

		for alpha in alphas:
			c_pfm_fn = 'pfm/cpfm_{}_{}_{}*{}_{}_{}_{:.1f}'.format(src_l, trg_l, maskshape[0], maskshape[1], preprocess, sim_metrics, alpha)
			c_pfm_file = open(c_pfm_fn, 'rb')
			c_pfm = pickle.load(c_pfm_file)
			series['overall'].append(float(c_pfm['overall']['posidf'].split('±')[0]))
			for corpus in corpora:
				series[corpus].append(np.average(c_pfm[corpus]['posidf']))

		print(src_l, trg_l)
		print(series['overall'])
		print(np.argmax(series['overall']))
		fig = plt.figure()
		ax = fig.add_subplot(111)
		legend = []
		for corpus in series:
			ax.plot(alphas, series[corpus], 'o-', label=corpus)
			legend.append(corpus)
		ax.legend(title='Subcorpora', loc='lower right')

		ax.set_xlabel(u'Alpha')
		ax.set_ylabel(u'Average F1 score')
		ax.set_ylim(0.1,0.9)
		plt.savefig(plot_fn)
		plt.close('all')


