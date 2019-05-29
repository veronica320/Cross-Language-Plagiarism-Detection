from conf import *
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
import csv

print('Validation begins...')

corpora = corpora

def cal_sim(src_vec, trg_vec, sim_metrics):
	if sim_metrics == 'cosine':
		return cosine_similarity(src_vec.reshape(1,-1), trg_vec.reshape(1,-1))

def evaluate(sim_matrix, eval_metrics, k):
	if eval_metrics == 'r@k':
		topk = np.argsort(sim_matrix, 1)[:, -k:][:, ::-1]
		r_at_k = ((topk == 0).sum())/topk.shape[0]
	return r_at_k

if os.path.isfile(c_pfm_fn):
	c_pfm_file = open(c_pfm_fn, 'rb')
	c_pfm = pickle.load(c_pfm_file)
else:
	c_pfm = {corpus:{} for corpus in corpora}


for corpus in corpora:

	print('Validating on corpus {}...'.format(corpus))
	maskdir = '{}/{}/{}-{}/'.format(datadir, mask_set, corpus, gran)

	fold = 0

	w_pfm = {weighting:[] for weighting in weightings}

	for mask_filename in os.listdir(maskdir):
		m_pfm_list = []
		sim_matrix_list = []
		if mask_filename != '.DS_Store':
			print('Validating on fold {}...'.format(fold))
			mask_file = open(maskdir + mask_filename, "r")

			sim_matrix_list = [-np.ones([maskshape[0], maskshape[1]]) for i in range(len(weightings))]

			##################

			line_id = 0
			lines = mask_file.readlines()
			line_num = len(lines)
			while(line_id < line_num):
				trg_vecs = []
				mask = json.loads(lines[line_id])
				trg_filename = '{}-{}.txt'.format(mask['3'], trg_l)
				if not os.path.isfile('{}/dataset/{}/{}/{}/{}'.format(datadir, gran, corpus, trg_l, trg_filename)):
					# print('{}/dataset/{}/{}/{}/{}'.format(datadir, gran, corpus, trg_l, trg_filename))
					line_id += 1
					continue

				if line_id % maskshape[1] == 0:
					src_filename = '{}-{}.txt'.format(mask['2'], src_l)
					if not os.path.isfile('{}/dataset/{}/{}/{}/{}'.format(datadir, gran, corpus, src_l, src_filename)):
						# print('{}/dataset/{}/{}/{}/{}'.format(datadir, gran, corpus, src_l, src_filename))
						line_id += maskshape[1]
						continue
					src_vecs = []
					for weighting in weightings:
						vecdir = '{}/{}vec/{}/{}'.format(datadir, weighting, gran, corpus)
						if preprocess == 'fine':
							vecdir = '{}/fine_{}vec/{}/{}'.format(datadir, weighting, gran, corpus)
						src_vec_file = open('{}/{}/{}'.format(vecdir , src_l, src_filename), 'rb')
						src_vec = pickle.load(src_vec_file)
						src_vecs.append(src_vec)
				i = 0
				for weighting in weightings:
					vecdir = '{}/{}vec/{}/{}'.format(datadir, weighting, gran, corpus)
					if preprocess == 'fine':
						vecdir = '{}/fine_{}vec/{}/{}'.format(datadir, weighting, gran, corpus)
					trg_vec_file = open('{}/{}/{}'.format(vecdir, trg_l, trg_filename), 'rb')
					trg_vec = pickle.load(trg_vec_file)
					trg_vecs.append(trg_vec)
					try:
						sim = cal_sim(src_vecs[i], trg_vecs[i], sim_metrics)
					except ValueError:
						pass
					sim_matrix_list[i][mask['0'], mask['1']] = sim
					i += 1

				line_id += 1

			for i in range(len(sim_matrix_list)):
				# print(sim_matrix_list[i])
				sim_matrix_list[i] = sim_matrix_list[i][sim_matrix_list[i][:, 0] >= 0, :]
				performance = evaluate(sim_matrix_list[i], eval_metrics, k)
				print('k={}, weighting={}, r={}'.format(k, weightings[i], performance))
				w_pfm[weightings[i]].append(performance)

			fold += 1

	for weighting in weightings:
		if corpus not in c_pfm:
			c_pfm[corpus] = {}
		c_pfm[corpus][weighting] = w_pfm[weighting]


	c_pfm_file = open(c_pfm_fn, 'wb')
	pickle.dump(c_pfm, c_pfm_file)
	c_pfm_file.close()