import os
import json

corpora = ['taln', 'apr', 'wiki', 'europarl', 'jrc']

gran = ['document', 'sentence', 'chunk'][0]

w2v = ['multivec']

datadir = 'Cross-Language-Dataset'

mask_set = ['masks', 'masks_sample'][1]

maskshape = (103, 203)

eval_metrics = ['r@k'][0]

langs = ['en', 'fr']

src_l = langs[0]

trg_l = langs[1]

weightings = ['sum', 'pos', 'idf', 'posidf']

preprocesses = ['coarse', 'fine']
preprocess = ['coarse', 'fine'][0]

sim_metrics = ['cosine'][0]

alpha = 0.5

alphas = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]


# c_pfm_fn = 'pfm/cpfm_{}_{}_{}*{}_{}_{}_{}'.format(src_l, trg_l, maskshape[0], maskshape[1], preprocess, sim_metrics, alpha)
#
# rec_fn = 'record/rec_{}_{}_{}*{}_{}_{}_{}.csv'.format(src_l, trg_l, maskshape[0], maskshape[1], preprocess, sim_metrics, alpha)

k = [1, 5, 10][0]



