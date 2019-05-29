import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from conf import *
import pickle
from pprint import pprint
import csv

c_pfm_fn = 'pfm/cpfm_{}_{}_{}*{}_{}_{}_{}'.format(src_l, trg_l, maskshape[0], maskshape[1], preprocess, sim_metrics, alpha)
c_pfm_file = open(c_pfm_fn, 'rb')
c_pfm = pickle.load(c_pfm_file)

t_test_fn = 't_test/ttest_{}_{}_{}*{}_{}_{}_{}.csv'.format(src_l, trg_l, maskshape[0], maskshape[1], preprocess, sim_metrics, alpha)

t_test_pairs = ['idf vs. sum', 'pos vs. idf', 'posidf vs. pos']

t_test_dict = {corpus:{} for corpus in corpora}
t_test_dict['overall'] = {}

for pair in t_test_pairs:
	method_x = pair.split()[0]
	method_y = pair.split()[2]
	X_overall = []
	Y_overall = []
	for corpus in corpora:
		X = c_pfm[corpus][method_x]
		Y = c_pfm[corpus][method_y]
		res = stats.ttest_rel(X, Y)
		t_test_dict[corpus][pair] = 't={:.2f}, p={:.4f}'.format(res[0], res[1])
		X_overall += c_pfm[corpus][method_x]
		Y_overall += c_pfm[corpus][method_y]
	res = stats.ttest_rel(X_overall, Y_overall)
	t_test_dict['overall'][pair] = 't={:.2f}, p={:.4f}'.format(res[0], res[1])

# pprint(t_test_dict)

t_test_dict_T = {}
for pair in t_test_pairs:
	t_test_dict_T[pair] = []
	for corpus in corpora:
		t_test_dict_T[pair].append(t_test_dict[corpus][pair])
	t_test_dict_T[pair].append(t_test_dict['overall'][pair])

# pprint(t_test_dict_T)



with open(t_test_fn, 'w') as csv_file:
	writer = csv.writer(csv_file)
	writer.writerow(['Methods']+corpora+['Overall'])
	for key, value in t_test_dict_T.items():
		writer.writerow([key]+value)