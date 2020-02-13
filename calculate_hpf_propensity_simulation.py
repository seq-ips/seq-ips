import pandas as pd
from hpfrec import HPF

import argparse
import os.path as osp
import numpy as np
from utils import ndcg_at_k

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="simulate_w0",
                        help='Number of epochs.')
    return parser.parse_args()


args = parse_args()

data_dir = osp.join("data",args.dataset)
num_user = 3000
num_item = 1000

train_data_count = []
valid_data_count = []
train_list = []
train_user_col_list = [[] for _ in range(num_user)]
num_train = 20
num_valid = 5
num_test = 5
valid_user_obs_list = [[] for _ in range(num_user)]
test_user_obs_list = [[] for _ in range(num_user)]
exc_user_obs_list_valid = [[] for _ in range(num_user)]
exc_user_obs_list_test = [[] for _ in range(num_user)]
with open(osp.join(data_dir,"observation_data.txt"), 'r') as fin:
    for (line_idx, line) in enumerate(fin):
        line_info = line.replace("\n", "").split(",")

        user,item,rate,s_idx,weight = int(line_info[0]),int(line_info[1]),float(line_info[2]),int(line_info[3]),float(line_info[4])
        if s_idx < num_train:
            train_user_col_list[user].append([item,rate,weight])
            train_data_count.append([user,item,1])
            train_list.append((user,item))
            exc_user_obs_list_valid[user].append(item)
            exc_user_obs_list_test[user].append(item)
        elif s_idx < num_train + num_valid:
            valid_data_count.append([user,item,1])
            valid_user_obs_list[user].append(item)
            exc_user_obs_list_test[user].append(item)
        elif s_idx < num_train + num_valid + num_test:
            test_user_obs_list[user].append(item)


train_data_count = np.array(train_data_count)
train_data_count = pd.DataFrame(train_data_count,columns=['UserId', 'ItemId', 'Count'])

valid_data_count = np.array(valid_data_count)
valid_data_count = pd.DataFrame(valid_data_count,columns=['UserId', 'ItemId', 'Count'])

## Full function call
recommender = HPF(
    k=32, a=0.3, a_prime=0.3, b_prime=1.0,
    c=0.3, c_prime=0.3, d_prime=1.0, ncores=-1,
    stop_crit='train-llk', check_every=5, stop_thr=1e-3,
    users_per_batch=None, items_per_batch=None, step_size=lambda x: 1/np.sqrt(x+2),
    maxiter=200, reindex=True, verbose=True,
    random_seed = None, allow_inconsistent_math=False, full_llk=False,
    alloc_full_phi=False, keep_data=True, save_folder=None,
    produce_dicts=True, keep_all_objs=True, sum_exp_trick=False
)


# Fitting the model to the data
recommender.fit(train_data_count,val_set=valid_data_count)

# Report the performance
pos_prob = []
item_ranks = []
Ks = [1,5,10,20,50]
k_recalls = [[] for _ in Ks]
k_ndcgs = [[] for _ in Ks]
for u_idx in range(num_user):
    pred_scores = recommender.predict(user=[u_idx for item in range(num_item)], item = [item for item in range(num_item)])
    if len(test_user_obs_list[u_idx]) > 0:
        all_scores = pred_scores
        exp_scores = all_scores
        exc_list = exc_user_obs_list_test[u_idx]
        exc_list = np.array(exc_list)
        if len(exc_list) > 0:
            all_scores[exc_list] = -30.0
            exp_scores[exc_list] = 0
        exp_prob = exp_scores / exp_scores.sum()

        test_pos_list = test_user_obs_list[u_idx]
        all_ranks = all_scores.argsort(axis=0)[::-1]
        all_rank_array = np.zeros(num_item)
        for idx in range(num_item):
            all_rank_array[all_ranks[idx]] = idx + 1

        pos_prob.extend((exp_prob[np.array(test_pos_list)] + 1e-10).tolist())
        item_ranks.extend(all_rank_array[np.array(test_pos_list)].tolist())
        if u_idx > 0 and u_idx % 3000 == 0:
            print(str(u_idx) + "/" + str(num_user))
        pos_set = set(test_user_obs_list[u_idx])
        t_item_ranks = all_rank_array[np.array(test_pos_list)]
        rank_relevant_array = np.zeros(num_item).astype(np.int)
        rank_relevant_array[t_item_ranks.astype(int)] = 1
        t_item_ranks = t_item_ranks.tolist()
        for (k_idx,k) in enumerate(Ks):
            top_at_set = set(all_ranks[:k].tolist())
            k_recalls[k_idx].append(len(top_at_set & pos_set) * 1.0 / len(pos_set))
            k_ndcgs[k_idx].append(ndcg_at_k(rank_relevant_array.tolist(),k,1))

print("NLL:%f" % np.log(np.array(pos_prob)).mean())
for (k_idx,k) in enumerate(Ks):
    print("Recall@%d:%f" % (k,np.array(k_recalls[k_idx]).mean()))
for (k_idx,k) in enumerate(Ks):
    print("NDCG@%d:%f" % (k,np.array(k_ndcgs[k_idx]).mean()))

# Save propensity scores
with open("propensity_scores/scores_hpf_%s_%d" % (args.dataset,num_train),"w") as fout:
    for t_data in train_list:
        user,item = t_data[0],t_data[1]
        pred_p = recommender.predict(user=user, item=item)
        out_line = str(user) + "," + str(item) + "," + str(pred_p) + "\n"
        fout.write(out_line)



