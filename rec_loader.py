import torch.utils.data as data
import numpy as np
import random


class SeqUserLoader(data.Dataset):
    def __init__(self, user_rating_list, num_user, num_item):
        self.user_rating_list = user_rating_list
        self.num_user = num_user
        self.num_item = num_item
        self.item_rating_dict = {}
        for temp_rating_data in self.user_rating_list:
            for t_data in temp_rating_data:
                item = t_data[0]
                if item not in self.item_rating_dict:
                    self.item_rating_dict[item] = 0
                self.item_rating_dict[item] += 1
        self.item_rating_dict[num_item] = 9999
        self.idx_list = list(range(num_user))

    def shuffle(self):
        random.shuffle(self.idx_list)

    def data_size(self):
        return self.num_user

    def get_batch_data(self,batch_idx):
        s_idx = self.idx_list[batch_idx]
        user_r_list = self.user_rating_list[s_idx]
        user_item_list = [item[0] for item in user_r_list]
        user_rate_list = [item[1] for item in user_r_list]

        return np.array(user_item_list),np.array(user_rate_list).astype(np.float32)[:,None]


class UserRatingLoader(data.Dataset):
    def __init__(self, user_rating_data):
        self.user_rating_data = user_rating_data

    def shuffle(self):
        temp_list = self.user_rating_data
        random.shuffle(temp_list)
        self.user_rating_data = temp_list

    def data_size(self):
        return len(self.user_rating_data)

    def get_batch_data(self,batch_idx,batch_size,if_weight = 0):
        batch_data = {"user":[],"item":[],"rate":[],"weight":[]}
        current_batch_size = min(self.data_size() - batch_idx * batch_size, batch_size)
        for b_inst_idx in range(current_batch_size):
            inst_idx = batch_idx * batch_size + b_inst_idx
            data_pair = self.user_rating_data[inst_idx]
            user,item,rate = data_pair[0],data_pair[1],data_pair[2]

            batch_data["user"].append(user)
            batch_data["item"].append(item)
            batch_data["rate"].append(rate)
            if len(data_pair) > 3:
                if if_weight:
                    batch_data["weight"].append(data_pair[3])
                else:
                    batch_data["weight"].append(1.0)
        return batch_data
