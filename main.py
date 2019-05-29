from conf import *
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
import csv

def cal_sim(src_vec, trg_vec, sim_metrics):
	if sim_metrics == 'cosine':
		return cosine_similarity(src_vec.reshape(1,-1), trg_vec.reshape(1,-1))

def evaluate(sim_matrix, eval_metrics, k):
	if eval_metrics == 'r@k':
		topk = np.argsort(sim_matrix, 1)[:, -k:][:, ::-1]
		r_at_k = ((topk == 0).sum())/topk.shape[0]
	return r_at_k

if os.path.isfile('c_pfm'):
	c_pfm_file = open('c_pfm', 'rb')
	c_pfm = pickle.load(c_pfm_file)
else:
	c_pfm = {corpus:{} for corpus in corpora}

for corpus in corpora:

	print('Validating on corpus {}...'.format(corpus))
	maskdir = '{}/{}/{}-{}/'.format(datadir, mask, corpus, gran)

	fold = 0

	w_pfm = {weighting:[] for weighting in weightings}

	for mask_filename in os.listdir(maskdir):
		m_pfm_list = []
		sim_matrix_list = []
		if mask_filename != '.DS_Store':
			print('Validating on fold {}...'.format(fold))
			mask_file = open(maskdir + mask_filename, "r")

			sim_matrix_list = [-np.ones([1000, 1000]) for i in range(len(weightings))]

			last_src = ''
			src_exists = False
			src_vecs = []

			for line in mask_file:
				trg_vecs = []
				mask = json.loads(line)

				if mask['2'] != last_src:
					last_src = mask['2']
					src_filename = '{}-{}.txt'.format(mask['2'], src_l)
					if os.path.isfile('{}/dataset/{}/{}/{}/{}'.format(datadir, gran, corpus, src_l, src_filename)):
						src_exists = True
						for weighting in weightings:
							src_vecs.append(doc2vec(src_filename, src_l, weighting))
					else:
						src_exists = False
						continue
				else:
					if not src_exists:
						continue


				if src_exists:
					trg_filename = '{}-{}.txt'.format(mask['3'], trg_l)
					if os.path.isfile('{}/dataset/{}/{}/{}/{}'.format(datadir, gran, corpus, trg_l, trg_filename)):
						i = 0
						for weighting in weightings:
							trg_vecs.append(doc2vec(trg_filename, trg_l, weighting))
							sim = cal_sim(src_vecs[i], trg_vecs[i], sim_metrics)
							sim_matrix_list[i][mask['0'], mask['1']] = sim
							i += 1


			for i in range(len(sim_matrix_list)):
				sim_matrix_list[i] = sim_matrix_list[i][sim_matrix_list[i][:, 0] >= 0, :]
				performance = evaluate(sim_matrix_list[i], eval_metrics, k)
				print('k={}, weighting={}, r={}'.format(k, weightings[i], performance))
				w_pfm[weightings[i]].append(performance)

			fold += 1
			print(w_pfm)


	for weighting in weightings:
		c_pfm[corpus][weighting] = sum(w_pfm[weighting]) / len(w_pfm[weighting])

print(c_pfm)

c_pfm_file = open(c_pfm_fn, 'wb')
pickle.dump(c_pfm, c_pfm_file)
c_pfm_file.close()


# corpus_weight = {'taln':0.62, 'wiki':1.0, 'apr':1.0, 'jrc':1.0}
# c_pfm['overall'] = {}
#
# for weighting in weightings:
# 	overall_pfm = 0.0
# 	for corpus in c_pfm:
# 		if corpus != 'overall':
# 			overall_pfm += c_pfm[corpus][weighting] * corpus_weight[corpus]
# 	c_pfm['overall'][weighting] = overall_pfm / sum(corpus_weight.values())
#
# w_pfm = {}
# for weighting in weightings:
# 	w_pfm[weighting] = []
# 	for corpus in corpora:
# 		w_pfm[weighting].append(c_pfm[corpus][weighting])
# 	w_pfm[weighting].append(c_pfm['overall'][weighting])
#
# with open('record_0410.csv', 'w') as csv_file:
# 	writer = csv.writer(csv_file)
# 	writer.writerow(['method']+corpora+['overall'])
# 	for key, value in w_pfm.items():
# 		writer.writerow([key]+value)



