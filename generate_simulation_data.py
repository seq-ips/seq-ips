import numpy as np
from numpy.random import choice
import os
import os.path as osp
import pickle
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--dataset_name', type=str, default="simulate_w3",
                        help='Number of epochs.')
    return parser.parse_args()


def query_itemitem(r_his,simi_m):
    his_i = np.array([t_data[0] for t_data in r_his])
    his_r = np.array([t_data[1] for t_data in r_his])

    simi_his = simi_m[:,his_i]
    score_his = np.exp(simi_his)
    score_his_n = score_his / score_his.sum(axis=1)[:,None]
    pred_scores = (his_r[None,:] * score_his_n).sum(axis=1)

    return pred_scores


args = parse_args()
data_name = args.dataset_name
data_dir = "data/" + data_name
if not osp.exists(data_dir):
    os.makedirs(data_dir)

K = 10
mu_alpha = 1.0 * np.random.dirichlet([20 for _ in range(K)],1)
mu_beta = 1.0 * np.random.dirichlet([100 for _ in range(K)],1)
num_user = 3000
num_t = 3000
num_item = 1000

# Generate user matrix and item matrix
user_matrix = np.random.dirichlet(mu_alpha.tolist()[0],num_user)
item_matrix = np.random.dirichlet(mu_beta.tolist()[0],num_item)

V_mean = np.dot(user_matrix,item_matrix.transpose())
V = np.zeros((num_user,num_item))

sigma = 1e-2

# Generate rating matrix
for i in range(num_user):
    for j in range(num_item):
        mean_rate = V_mean[i,j]
        if mean_rate < 0.02:
            a = 1
        alpha = ((1 - mean_rate) / sigma - 1 / mean_rate) * (mean_rate ** 2)
        beta = alpha * (1 / mean_rate - 1)
        if alpha < 0:
            V[i, j] = 0
        else:
            samples = np.random.beta(alpha,beta,1)
            n_rate = 10 * samples[0]
            V[i, j] = n_rate
    if i % 100 == 0:
        print("Generate ratings for users:%d/%d" % (i,num_user))

rate_history = [[] for _ in range(num_t)]
obs_history = [[] for _ in range(num_t)]

# Generate noisy similarity matrix for recommendation
r_simi_items = np.dot(item_matrix,item_matrix.transpose())
simi_items = np.zeros(r_simi_items.shape)
sigma_simi = 1e-2
for i1 in range(num_item):
    for i2 in range(num_item):
        mean_simi = r_simi_items[i1,i2]
        alpha = ((1 - mean_simi) / sigma_simi - 1 / mean_simi) * (mean_simi ** 2)
        beta = alpha * (1 / mean_simi - 1)
        if alpha < 0:
            simi_items[i1, i2] = 0
        else:
            samples = np.random.beta(alpha,beta,1)
            simi_items[i1, i2] = samples[0]

rank_weights = np.ones(num_item)
rank_weights[:100] = 10.0
rank_weights = rank_weights / rank_weights.sum()
num_iter = 30
with open(osp.join(data_dir,"observation_data.txt"),"w") as fout:
    # Iteratively generate user-item pair by user's rating history
    for iteration in range(num_iter):
        print("Iteration:%d/%d" % (iteration,num_iter))
        new_data = []
        for user in range(num_t):
            if iteration == 0:
                c_scores = np.ones(num_item)
                c_scores = c_scores / c_scores.sum()
                i_samples = choice(np.array(range(num_item)),1,False,p=c_scores)
                i_sample = i_samples[0]
            else:
                predict_scores = query_itemitem(rate_history[user],simi_items)
                predict_rank = np.argsort(predict_scores)[::-1]
                predict_idx = np.zeros(num_item).astype(int)
                for (idx,item) in enumerate(predict_rank):
                    predict_idx[item] = idx

                c_scores = rank_weights[predict_idx]
                c_scores = c_scores / c_scores.sum()

                while 1:
                    i_samples = choice(np.array(range(num_item)), 1, False, p=c_scores)
                    i_sample = i_samples[0]
                    if i_sample not in obs_history[user]:
                        break
            obs_history[user].append(i_sample)
            rate_history[user].append((i_sample,V[int(user),int(i_sample)]))
            new_data.append([int(user),int(i_sample),V[int(user),int(i_sample)]])
            out_line = str(user) + "," + str(int(i_sample)) + "," + str(V[int(user),int(i_sample)]) + "," + str(iteration) + "," + str(c_scores[i_sample]) + "\n"
            fout.write(out_line)
            if user % 500 == 0:
                print("Users:%d/%d" % (user,num_user))
        new_data = np.array(new_data)

# Randomly sample test data
with open(osp.join(data_dir,"random_test_data.txt"),"w") as fout:
    for user in range(num_t):
        for iteration in range(20):
            new_data = []
            while 1:
                i_sample = np.random.randint(0, num_item)
                if i_sample not in obs_history[user]:
                    break
            out_line = str(user) + "," + str(int(i_sample)) + "," + str(V[int(user),int(i_sample)]) + "\n"
            fout.write(out_line)

# Save all information if necessary
# data = {
#     "user_matrix":user_matrix,
#     "item_matrix":item_matrix,
#     "V":V,"num_user":num_user,"num_item":num_item,"num_t":num_t,
#     "V_mean":V_mean
# }
#
# with open(osp.join(data_dir,"meta_data.pkl"), 'wb') as f:
#     pickle.dump(data, f)

