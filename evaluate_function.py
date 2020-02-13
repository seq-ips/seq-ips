import torch
import numpy as np
from utils import variable2numpy,ndcg_at_k
from model import EXPO
from sklearn.metrics.pairwise import cosine_similarity


def generate_exc_list(user_rating_list,num_item):
    train_idx = []
    for u_idx in range(len(user_rating_list)):
        for t_item in user_rating_list[u_idx]:
            if t_item[0] not in train_idx:
                train_idx.append(t_item[0])
    exc_idx = []
    for i_idx in range(num_item):
        if i_idx not in train_idx:
            exc_idx.append(i_idx)
    exc_idx = np.array(exc_idx)

    return exc_idx


def evaluate_score(all_scores,exc_idx,rating_pos_idx,test_pos_list,num_item,Ks = [50]):
    exp_score = np.exp(all_scores)
    if len(exc_idx) > 0:
        all_scores[exc_idx] = -30.0
        exp_score[exc_idx] = 0.0

    if len(rating_pos_idx) > 0:
        exp_score[rating_pos_idx] = 0
        all_scores[rating_pos_idx] = -30.0
    exp_prob = exp_score / exp_score.sum()

    all_ranks = all_scores.argsort(axis=0)[::-1][:, 0]
    all_rank_array = np.zeros(num_item)
    for idx in range(num_item):
        all_rank_array[all_ranks[idx]] = idx + 1

    t_pos_prob = (exp_prob[np.array(test_pos_list)] + 1e-6).tolist()
    t_recalls = []
    t_ndcgs = []
    t_item_ranks = all_rank_array[np.array(test_pos_list)]
    rank_relevant_array = np.zeros(num_item).astype(np.int)
    rank_relevant_array[t_item_ranks.astype(int)] = 1
    t_item_ranks = t_item_ranks.tolist()
    for k in Ks:
        top_at_set = set(all_ranks[:k].tolist())
        pos_set = set(test_pos_list)
        t_recalls.append(len(top_at_set & pos_set) * 1.0 / len(pos_set))
        t_ndcgs.append(ndcg_at_k(rank_relevant_array.tolist(),k,1))

    return t_pos_prob,t_item_ranks,t_recalls,t_ndcgs


# Calculate evaluation index for exposure model
def evaluate_obs_expo(model,logger,user_rating_list,test_user_obs_list,device):
    Ks = [1,5,10,20,50]
    model.eval()
    pos_prob = []
    item_ranks = []
    recalls = [[] for _ in range(len(Ks))]
    ndcgs = [[] for _ in range(len(Ks))]
    num_user = len(user_rating_list)
    num_item = model.num_items

    exc_idx = generate_exc_list(user_rating_list, num_item)
    for u_idx in range(num_user):
        if len(test_user_obs_list[u_idx]) > 0:
            input_user_ids = np.array([u_idx]).astype(np.int64)
            input_user_ids = torch.tensor(input_user_ids).to(device)
            pred_scores = model.evaluate(input_user_ids)
            all_scores = variable2numpy(pred_scores,device)
            all_scores = all_scores[0,:][:,None]
            rating_pos_list = [t_data[0] for t_data in user_rating_list[u_idx]]
            rating_pos_idx = np.array(rating_pos_list)

            t_pos_prob,t_item_ranks,t_recalls,t_ndcgs = evaluate_score(all_scores, exc_idx, rating_pos_idx, test_user_obs_list[u_idx], num_item, Ks)
            pos_prob.extend(t_pos_prob)
            item_ranks.extend(t_item_ranks)

            for k_idx in range(len(t_recalls)):
                recalls[k_idx].append(t_recalls[k_idx])
                ndcgs[k_idx].append(t_ndcgs[k_idx])

            if u_idx > 0 and u_idx % 3000 == 0:
                logger.info(str(u_idx) + "/" + str(num_user))

    nll = np.log(np.array(pos_prob)).mean()
    mar = np.array(item_ranks).mean()
    mre = [np.array(k_recalls).mean() for k_recalls in recalls]
    mndcg = [np.array(k_ndcgs).mean() for k_ndcgs in ndcgs]
    logger.info("Negative Log Likelihood:" + str(nll))
    logger.info("Mean Average Rank:" + str(mar))
    logger.info("Mean Recall:" + str(mre))
    logger.info("Mean NDCG:" + str(mndcg))
    result = {'nll':nll,'mar':mar,'mre':mre,'mndcg':mndcg}
    return result

# Evaluate exposure model
def evaluate_exposure(model,logger,gpu_id,user_rating_list,test_user_obs_list,device):
    model.eval()
    num_user = len(user_rating_list)
    num_item = model.num_items
    latent_dim = model.latent_dim

    all_user_embedding_h = []
    for user_idx in range(num_user):
        user_r_list = user_rating_list[user_idx]
        user_item_list = [item[0] for item in user_r_list]
        user_rate_list = [item[1] for item in user_r_list]
        user_item_list = np.array(user_item_list)
        user_rate_list = np.array(user_rate_list).astype(np.float32)[:, None]

        user_item_list = torch.tensor(user_item_list)
        user_rate_list = torch.tensor(user_rate_list)
        user_item_list, user_rate_list = user_item_list.to(device),user_rate_list.to(device)

        user_embedding_t = model.cal_user_embedding(user_item_list,user_rate_list,0)
        all_user_embedding_h.append(user_embedding_t)
    all_user_embedding_h = torch.cat(all_user_embedding_h, 0)

    model_dict = model.state_dict()
    modeln = EXPO({'num_items': num_item, 'num_users': num_user, 'latent_dim': latent_dim * 2},model_dict['bias_item'])
    modeln_dict = modeln.state_dict()
    modeln_dict['embedding_user.weight'] = all_user_embedding_h
    modeln_dict['affine_output.weight'] = model_dict['affine_output.weight']
    modeln_dict['affine_output.bias'] = model_dict['affine_output.bias']
    modeln.load_state_dict(modeln_dict)
    if device == "cuda":
        modeln.cuda(gpu_id)
    modeln.eval()

    result = evaluate_obs_expo(modeln,logger,user_rating_list,test_user_obs_list,device)

    return result

# Calculate diversity of recommendation results of rating prediction model
def evaluate_diversity(model,logger,device,movie_info,num_pick=10):
    model.eval()
    num_items = model.num_items
    item_cnt = np.zeros(num_items).astype(np.float)
    all_dis = []
    for u_idx in range(model.num_users):
        b_user = u_idx * np.ones(num_items).astype(np.int)
        b_item = np.array(range(num_items)).astype(np.int)
        b_user, b_item = torch.tensor(b_user), torch.tensor(b_item)

        b_scores = model.forward(b_user, b_item)
        b_scores = variable2numpy(b_scores,device)
        sort_idx = np.argsort(b_scores[:,0])[::-1]
        r_item_genres = []
        for s_item in sort_idx[:num_pick]:
            item_cnt[s_item] += 1
            r_item_genres.append(movie_info[s_item]['genre_array'][None,:])
        r_item_genres = np.concatenate(r_item_genres,0)
        all_dis.append((1 - cosine_similarity(r_item_genres)).sum() / (num_pick * (num_pick - 1)))
    item_cnt = item_cnt
    sort_item_idx = np.argsort(item_cnt)[::-1]
    gini_den,gini_sum = 0.0,0.0
    for (s_idx, item) in enumerate(sort_item_idx.tolist()):
        gini_den += 1.0 * (2 * s_idx - num_items + 1) * item_cnt[item]
        gini_sum += 1.0 * num_items * item_cnt[item]

    # logger.info(gini_den / gini_sum)
    # logger.info(np.array(all_lrs).mean())
    return gini_den / gini_sum,np.array(all_dis).mean()

# Calculate error of rating prediction model
def evaluate_rating_error(model,logger,data_loader,device,epoch,user_weight_dict = None,batch_size=512):
    model.eval()
    test_total_loss = 0.0
    test_total_loss_2 = 0.0
    test_weight_sum = 0.0
    test_batch_num = int(np.ceil(float(data_loader.data_size()) / batch_size))
    all_pred = []
    for batch_idx in range(test_batch_num):
        batch_weight = []
        batch_data = data_loader.get_batch_data(batch_idx, batch_size)

        for u_idx in batch_data["user"]:
            if user_weight_dict is None:
                batch_weight.append(1.0)
            else:
                batch_weight.append(1 / user_weight_dict[u_idx])
        batch_weight = np.array(batch_weight)

        b_user = torch.tensor(np.array(batch_data["user"]))
        b_item = torch.tensor(np.array(batch_data["item"]))
        b_rate = np.array(batch_data["rate"]).astype(np.float32)
        b_user, b_item = b_user.to(device),b_item.to(device)
        b_scores = model.forward(b_user,b_item)
        b_scores = variable2numpy(b_scores,device)
        loss = (((b_rate - b_scores[:,0]) ** 2) * batch_weight).sum()
        loss_2 = (np.abs(b_rate - b_scores[:, 0]) * batch_weight).sum()
        all_pred.extend(b_scores[:,0].tolist())
        test_total_loss += loss
        test_total_loss_2 += loss_2
        test_weight_sum += batch_weight.sum()
    mse_loss = test_total_loss / test_weight_sum
    mae_loss = test_total_loss_2 / test_weight_sum
    # logger.info("test:%d:%f:%f:%f" % (epoch, np.array(all_pred).mean(),test_total_loss / test_weight_sum,test_total_loss_2 / test_weight_sum))

    return {"mse":mse_loss,"mae":mae_loss}

