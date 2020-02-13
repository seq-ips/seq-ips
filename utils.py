import logging
import os.path as osp
import numpy as np


def dcg_at_k(r, k, method=0):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.


def ndcg_at_k(r, k, method=0):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def variable2numpy(var,device):
    if device == "cuda":
        return var.cpu().detach().numpy()
    else:
        return var.detach().numpy()


class Logger:
    def __init__(self, save_dir):
        if save_dir is not None:
            self.logger = logging.getLogger()
            logging.basicConfig(filename = osp.join(save_dir,"experiment.log"),format='%(asctime)s | %(message)s')
            logging.root.setLevel(level=logging.INFO)
        else:
            self.logger = None

    def info(self, msg, to_file=True):
        print(msg)
        if self.logger is not None and to_file:
            self.logger.info(msg)


