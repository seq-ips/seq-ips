import argparse
from rec_loader import SeqUserLoader
from model import EXPOSEQ
import torch
import torch.nn.functional as F
from utils import variable2numpy,Logger
import os
import os.path as osp
import numpy as np
from evaluate_function import evaluate_exposure

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--dataset', type=str, default="simulate_w0",
                        help='Number of epochs.')
    parser.add_argument('--device', type=str, default="cpu",
                        help='Number of epochs.')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size.')
    parser.add_argument('--latent_dim', type=int, default=32,
                        help='Dimension of the latent variable.')
    parser.add_argument('--gru_input_dim', type=int, default=32,
                        help='Dimension of the input of GRU.')
    parser.add_argument('--gru_hidden_dim', type=int, default=32,
                        help='Dimension of the hidden state of GRU.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay.')
    parser.add_argument('--out_interval', type=int, default=2000,
                        help='the iteration interval to output the loss.')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='the epoch interval to save the trained model.')
    parser.add_argument('--init_model', type=str, default="",
                        help='path of inited_model')
    parser.add_argument('--num_train', type=int, default=20,
                        help='Number of items for training')
    return parser.parse_args()


def train(args):
    device = args.device

    # Set up logger
    pre_fix = "%s_seqexpo" % (args.dataset)
    result_dir = "results/" + pre_fix
    if not osp.exists(result_dir):
        os.makedirs(result_dir)
    logger = Logger(result_dir)
    logger.info(str(args))

    # Read data
    data_dir = osp.join("data", args.dataset)
    num_user = 3000
    num_item = 1000

    num_train = args.num_train
    num_valid = 5
    num_test = 5

    user_rating_list_train = [[] for _ in range(num_user)]
    user_rating_list_trainval = [[] for _ in range(num_user)]
    train_item_idx = []
    p_train_count = np.zeros(num_item).astype(np.float32)

    valid_user_obs_list = [[] for _ in range(num_user)]
    test_user_obs_list = [[] for _ in range(num_user)]

    with open(osp.join(data_dir, "observation_data.txt"), 'r') as fin:
        for (line_idx, line) in enumerate(fin):
            line_info = line.replace("\n", "").split(",")

            user, item, rate, s_idx = int(line_info[0]), int(line_info[1]), float(line_info[2]), int(line_info[3])
            if s_idx < num_train:
                user_rating_list_train[user].append((item, rate, s_idx))
                user_rating_list_trainval[user].append((item, rate, s_idx))
                if item not in train_item_idx:
                    train_item_idx.append(item)
                p_train_count[item] += 1
            elif s_idx < num_train + num_valid:
                user_rating_list_trainval[user].append((item, rate, s_idx))
                valid_user_obs_list[user].append(item)
            elif s_idx < num_train + num_valid + num_test:
                test_user_obs_list[user].append(item)

    train_loader = SeqUserLoader(user_rating_list_train,num_user,num_item)
    train_loader.shuffle()

    # Calculate the popularity to initize the bias term for fast converge
    p_train_count = (p_train_count + 1e-10) / p_train_count[train_item_idx].sum()
    p_train_logp = np.log(p_train_count)[None,:]

    # Build the model
    model_config = {'device':device,'num_users':num_user,'num_items':num_item,\
                    'latent_dim':args.latent_dim, 'gru_input_dim':args.gru_input_dim,\
                    'gru_hidden_dim':args.gru_hidden_dim}
    model = EXPOSEQ(model_config,init_bias = torch.tensor(p_train_logp.astype(np.float32)).to(device))
    if len(args.init_model) > 0:
        model.load_state_dict(torch.load(args.init_model,map_location=device))

    if device == "cuda":
        model.cuda(args.gpu_id)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)

    b_valid_result = None
    b_test_result = None
    stop_cnt = 0
    # model.eval()
    # logger.info("Valid evaluation")
    # valid_result = evaluate_exposure(model, logger, args.gpu_id, user_rating_list_train, valid_user_obs_list, device)

    total_anneal_steps = 10000
    anneal = 0.0
    update_count = 0.0
    anneal_cap = 0.05
    for epoch in range(1, args.epochs):
        logger.info("anneal:%f" % (anneal))
        model.train()
        total_cls_loss = 0.0
        total_kld_loss = 0.0
        total_loss = 0.0
        batch_num = train_loader.data_size()
        for batch_idx in range(batch_num):
            user_item_list,user_rate_list = train_loader.get_batch_data(batch_idx)
            optimizer.zero_grad()
            user_item_list = torch.tensor(user_item_list)
            user_rate_list = torch.tensor(user_rate_list)
            user_item_list, user_rate_list = user_item_list.to(device),user_rate_list.to(device)
            item_logsoftmax,user_emb_p_mu_u, user_emb_p_sigma_u, user_emb_q_mu_u, user_emb_q_sigma_u, user_emb_mu_v, user_emb_sigma_v = model.forward(user_item_list,user_rate_list,1)

            cls_loss = F.nll_loss(item_logsoftmax,user_item_list) * args.batch_size
            kld_loss_1 = torch.sum(torch.sum(0.5 * (-torch.log(user_emb_q_sigma_u) + torch.log(user_emb_p_sigma_u) + (
                        (user_emb_q_mu_u - user_emb_p_mu_u) ** 2 + user_emb_q_sigma_u) / user_emb_p_sigma_u - 1), -1))
            kld_loss_2 = torch.sum(torch.sum(0.5 * (-torch.log(user_emb_sigma_v) + (user_emb_sigma_v + user_emb_mu_v ** 2) - 1), -1))

            # Anneal logic
            if total_anneal_steps > 0:
                anneal = min(anneal_cap, 1. * update_count / total_anneal_steps)
            else:
                anneal = anneal_cap
            update_count += 1.0

            loss = cls_loss + anneal * (kld_loss_1 + kld_loss_2)
            loss.backward()
            optimizer.step()
            total_cls_loss += variable2numpy(cls_loss, device)
            total_kld_loss += variable2numpy(kld_loss_1 + kld_loss_2, device)
            total_loss += variable2numpy(loss,device)

            # if batch_idx > 0 and batch_idx % args.out_interval == 0:
            #     logger.info("train:%d:%d:%f,%f,%f" % (epoch, batch_idx, total_cls_loss / ((batch_idx + 1) * args.batch_size), total_kld_loss / ((batch_idx + 1) * args.batch_size), total_loss / ((batch_idx + 1) * args.batch_size)))
        logger.info("train:%d:%f,%f,%f" % (epoch,total_cls_loss / batch_num,total_kld_loss / batch_num,total_loss / batch_num))
        train_loader.shuffle()

        if epoch > 0 and epoch % args.save_interval == 0:
            model.eval()
            logger.info("Valid evaluation")
            valid_result = evaluate_exposure(model, logger, args.gpu_id, user_rating_list_train, valid_user_obs_list, device)
            logger.info("Test evaluation")
            test_result = evaluate_exposure(model, logger, args.gpu_id, user_rating_list_trainval, test_user_obs_list, device)
            # torch.save(model.state_dict(), osp.join(result_dir, str(epoch) + ".pkl"))
            if b_valid_result is None or valid_result['nll'] > b_valid_result['nll']:
                b_valid_result = valid_result
                b_test_result = test_result
                torch.save(model.state_dict(), osp.join(result_dir,"final_model.pkl"))
                stop_cnt = 0
            else:
                stop_cnt += 1
            logger.info(b_valid_result)
            logger.info(b_test_result)

            if stop_cnt >= 3:
                logger.info("Stop Training.")
                break


if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    train(args)


