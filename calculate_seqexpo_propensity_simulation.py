import os.path as osp
import argparse
from rec_loader import SeqUserLoader
from model import EXPOSEQ
import torch
from utils import variable2numpy
import numpy as np

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
    parser.add_argument('--init_model', type=str, default="",
                        help='path of inited_model')
    return parser.parse_args()


args = parse_args()
device = args.device

num_user = 3000
num_item = 1000
num_train = 20

# Read data
user_rating_list = [[] for _ in range(num_user)]
data_dir = osp.join("data", args.dataset)
with open(osp.join(data_dir,"observation_data.txt"), 'r') as fin:
    for (line_idx, line) in enumerate(fin):
        line_info = line.replace("\n", "").split(",")
        user,item,rate,s_idx = int(line_info[0]),int(line_info[1]),float(line_info[2]),int(line_info[3])
        if s_idx < num_train:
            user_rating_list[user].append((item, rate, s_idx))
train_loader = SeqUserLoader(user_rating_list,num_user,num_item)

# Load exposure model
model_config = {'device': device, 'num_users': num_user, 'num_items': num_item,
                'latent_dim': args.latent_dim, 'gru_input_dim': args.gru_input_dim,
                'gru_hidden_dim': args.gru_hidden_dim}
model = EXPOSEQ(model_config, init_bias=None)
init_model = "results/%s_seqexpo/final_model.pkl" % args.dataset
if len(init_model) > 0:
    model.load_state_dict(torch.load(init_model,map_location=device))
if device == "cuda":
    model.cuda(args.gpu_id)

# Calcualate the propensity scores and save
batch_num = train_loader.data_size()
model.eval()
out_file = open("propensity_scores/scores_seq_%s_%d" % (args.dataset,num_train),"w")
for batch_idx in range(batch_num):
    user_item_list_raw,user_rate_list = train_loader.get_batch_data(batch_idx)
    user_item_list = torch.tensor(user_item_list_raw)
    user_rate_list = torch.tensor(user_rate_list)
    user_item_list, user_rate_list = user_item_list.to(device),user_rate_list.to(device)

    b_scores,b_scores_bias = model.forward_evaluate(user_item_list, user_rate_list)
    b_scores_bias = variable2numpy(b_scores_bias,device)
    b_scores = variable2numpy(b_scores,device)

    for (item_idx,item) in enumerate(user_item_list_raw.tolist()):
        item_scores_idx = b_scores[item_idx,:]
        item_prob = np.exp(item_scores_idx)
        item_prob = item_prob / item_prob.sum()
        out_line = str(batch_idx) + "," + str(item) + "," + str(item_prob[item]) + "\n"
        out_file.write(out_line)
    if batch_idx % 100 == 0:
        print("%d/%d" % (batch_idx, batch_num))