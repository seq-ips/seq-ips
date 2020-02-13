import argparse
from rec_loader import UserRatingLoader
from model import GMF
import torch
from utils import variable2numpy,Logger
import os
import os.path as osp
import numpy as np
from evaluate_function import evaluate_rating_error
import random

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="simulate_w0",
                        help='Number of epochs.')
    parser.add_argument('--device', type=str, default="cpu",
                        help='Number of epochs.')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--epochs', type=int, default=250,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_dim', type=int, default=64,
                        help='Embedding size of the model.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay.')
    parser.add_argument('--evaluate_interval', type=int, default=1,
                        help='the epoch interval to save the trained model.')
    parser.add_argument('--if_weight', type=int, default=1,
                        help='If use IPS')
    parser.add_argument('--weight_type', type=int, default=1,
                        help='IPS type: 0 for pop, 1 for Poisson factorization model(PF), 2 for our model')
    return parser.parse_args()

def train(args):
    device = args.device

    num_train = 20
    num_valid = 5
    num_user = 3000
    num_item = 1000

    # Read the calculated propensity scores if necessary
    if args.weight_type == 1:
        weight_dict = {}
        with open("propensity_scores/scores_hpf_%s_%d" % (args.dataset, num_train), 'r') as fin:
            for (line_idx, line) in enumerate(fin):
                line_info = line.replace("\n", "").split(",")
                weight_dict[(int(line_info[0]), int(line_info[1]))] = float(line_info[2])
    elif args.weight_type == 2:
        weight_dict = {}
        with open("propensity_scores/scores_seq_%s_%d" % (args.dataset, num_train), 'r') as fin:
            for (line_idx, line) in enumerate(fin):
                line_info = line.replace("\n", "").split(",")
                weight_dict[(int(line_info[0]), int(line_info[1]))] = float(line_info[2])

    # Set up logger
    pre_fix = "rate_prediction_%s_%d_%d_%d" % (args.dataset, args.if_weight, args.weight_type, num_train)
    result_dir = "results/" + pre_fix
    if not osp.exists(result_dir):
        os.makedirs(result_dir)
    logger = Logger(result_dir)
    logger.info(str(args))

    # Read data
    data_dir = osp.join("data", args.dataset)
    train_user_col_list = [[] for _ in range(num_user)]
    valid_rating_list = []
    with open(osp.join(data_dir, "observation_data.txt"), 'r') as fin:
        for (line_idx, line) in enumerate(fin):
            line_info = line.replace("\n", "").split(",")

            user, item, rate, t_idx, ideal_weight = int(line_info[0]), int(line_info[1]), float(line_info[2]), int(line_info[3]), float(line_info[4])
            if len(train_user_col_list[user]) < num_train:
                train_user_col_list[user].append([item, rate])
            elif len(train_user_col_list[user]) < num_train + num_valid:
                valid_rating_list.append((user, item, rate))
    test_rating_list = []
    with open(osp.join(data_dir, "random_test_data.txt"), 'r') as fin:
        for (line_idx, line) in enumerate(fin):
            line_info = line.replace("\n", "").split(",")

            user, item, rate = int(line_info[0]), int(line_info[1]), float(line_info[2])
            test_rating_list.append((user, item, rate))

    # Calculate popularity
    p_count = np.zeros(num_item).astype(np.float32)
    for u_idx in range(len(train_user_col_list)):
        for t_data in train_user_col_list[u_idx]:
            p_count[t_data[0]] += 1
    p_count = p_count / p_count.sum()

    # Pair each training example with its propensity scores
    train_rating_list = []
    # t_rates = []
    # t_weights = []
    for user in range(len(train_user_col_list)):
        user_train_list = train_user_col_list[user]
        for (t_idx, t_data) in enumerate(user_train_list):
            item = t_data[0]
            rate = t_data[1]
            if args.weight_type == 0:
                weight = p_count[item] * num_item
            else:
                weight = weight_dict[(user,item)] * num_item

            train_rating_list.append((user, item, rate, 1 / weight))

    random.shuffle(train_rating_list)
    train_loader = UserRatingLoader(train_rating_list)
    valid_loader = UserRatingLoader(valid_rating_list)
    test_loader = UserRatingLoader(test_rating_list)

    model_config = {'num_users': num_user, 'num_items': num_item, 'latent_dim': args.num_dim}
    all_valid_results = []
    all_test_results = []

    # Repeat training for 10 times
    for t_idx in range(10):
        logger.info("Trial %d" % t_idx)

        # Create model
        model = GMF(model_config)
        if device == "cuda":
            model.cuda(args.gpu_id)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        model.eval()
        b_valid_loss = 9999.0
        b_valid_result, b_test_result = None, None
        stop_cnt = 0
        for epoch in range(1, args.epochs):
            model.train()
            total_loss = 0.0
            batch_num = int(np.ceil(float(train_loader.data_size()) / args.batch_size))
            for batch_idx in range(batch_num):
                batch_data = train_loader.get_batch_data(batch_idx, args.batch_size, args.if_weight)
                optimizer.zero_grad()
                b_user = torch.tensor(np.array(batch_data["user"]))
                b_item = torch.tensor(np.array(batch_data["item"]))
                b_rate = torch.tensor(np.array(batch_data["rate"]).astype(np.float32))
                b_weight = torch.tensor(np.array(batch_data["weight"]).astype(np.float32))
                b_user, b_item, b_rate, b_weight = b_user.to(device), b_item.to(device), b_rate.to(
                    device), b_weight.to(device)
                b_scores = model.forward(b_user, b_item)
                loss = ((((b_rate - b_scores[:, 0]) ** 2) * b_weight).sum()) / (b_weight.mean())
                loss.backward()
                optimizer.step()
                total_loss += variable2numpy(loss, device)

            logger.info("train:%d:%f" % (epoch, total_loss / len(train_rating_list)))
            train_loader.shuffle()
            if epoch > 0 and epoch % args.evaluate_interval == 0:
                model.eval()
                valid_result = evaluate_rating_error(model, logger, valid_loader, device, epoch)
                test_result = evaluate_rating_error(model, logger, test_loader, device, epoch)

                # Decide when to stop
                if valid_result["mse"] < b_valid_loss:
                    b_valid_loss = valid_result["mse"]
                    b_valid_result = valid_result
                    b_test_result = test_result
                    stop_cnt = 0
                    logger.info("The best valid loss is : " + str(b_valid_loss))
                else:
                    stop_cnt += 1
                if stop_cnt >= 3:
                    break

        logger.info(b_valid_result)
        logger.info(b_test_result)
        all_valid_results.append(b_valid_result)
        all_test_results.append(b_test_result)

    # Report the average results
    for key in all_valid_results[0].keys():
        all_valid_values = np.array([t_result[key] for t_result in all_valid_results])
        all_test_values = np.array([t_result[key] for t_result in all_test_results])
        logger.info("%s in valid,%f,%f" % (key, all_valid_values.mean(), all_valid_values.std()))
        logger.info("%s in test,%f,%f" % (key, all_test_values.mean(), all_test_values.std()))
        logger.info("Per result valid:")
        for t_result in all_valid_results:
            logger.info(t_result[key])
        logger.info("Per result test:")
        for t_result in all_test_results:
            logger.info(t_result[key])


if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    train(args)
