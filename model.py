import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


# Exposure model based on calcualted user latent variables
class EXPO(torch.nn.Module):
    def __init__(self, config, init_bias = None):
        super(EXPO, self).__init__()
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']

        self.bias_item = torch.nn.Parameter(init_bias)
        for p in self.parameters():
            p.requires_grad = False
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.affine_output = torch.nn.Linear(in_features=self.latent_dim, out_features=self.num_items)

    def evaluate(self, user_indices):
        user_embedding = self.embedding_user(user_indices)
        scores = self.affine_output(user_embedding) + self.bias_item

        return scores

    def init_weight(self):
        pass

# Generalized Matrix Factorization for rating prediction
class GMF(torch.nn.Module):
    def __init__(self, config):
        super(GMF, self).__init__()
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.affine_output = torch.nn.Linear(in_features=self.latent_dim, out_features=1)

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        element_product = torch.mul(user_embedding, item_embedding)
        scores = self.affine_output(element_product)

        return scores


class EXPOSEQ(torch.nn.Module):
    def __init__(self, config, init_bias = None):
        super(EXPOSEQ, self).__init__()
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']
        self.gru_input_dim = config['gru_input_dim']
        self.gru_hidden_dim = config['gru_hidden_dim']
        self.device= config['device']

        if init_bias is None:
            init_bias = np.zeros((1,self.num_items)).astype(np.float32)
            init_bias = torch.Tensor(init_bias)
            self.bias_item = torch.nn.Parameter(init_bias)

        self.bias_item = torch.nn.Parameter(init_bias)

        for p in self.parameters():
            p.requires_grad = False

        self.embedding_item_mu_v = torch.nn.Embedding(num_embeddings=self.num_items + 1,embedding_dim=self.gru_input_dim)
        self.embedding_rate_mu_v = torch.nn.Linear(in_features=1, out_features=self.gru_input_dim, bias=False)
        self.embedding_item_sigma_v = torch.nn.Embedding(num_embeddings=self.num_items + 1,embedding_dim=self.gru_input_dim)
        self.embedding_rate_sigma_v = torch.nn.Linear(in_features=1, out_features=self.gru_input_dim, bias=False)

        self.embedding_item_u_p = torch.nn.Embedding(num_embeddings=self.num_items + 1, embedding_dim=self.gru_input_dim)
        self.embedding_item_u_q = torch.nn.Embedding(num_embeddings=self.num_items + 1, embedding_dim=self.latent_dim)
        self.embedding_rate_u_p = torch.nn.Linear(in_features=1, out_features=self.gru_input_dim, bias=False)
        self.rnn = torch.nn.GRU(self.gru_input_dim * 2, self.gru_hidden_dim)
        self.W_p_mu_u = torch.nn.Linear(in_features=self.gru_hidden_dim, out_features=self.latent_dim, bias=False)
        self.W_p_sigma_u = torch.nn.Linear(in_features=self.gru_hidden_dim, out_features=self.latent_dim, bias=False)
        self.W_q_mu_u = torch.nn.Linear(in_features=self.gru_hidden_dim + 2 * self.latent_dim, out_features=self.latent_dim, bias=False)
        self.W_q_sigma_u = torch.nn.Linear(in_features=self.gru_hidden_dim + 2 * self.latent_dim, out_features=self.latent_dim, bias=False)
        self.affine_output = torch.nn.Linear(in_features=self.latent_dim * 2, out_features=self.num_items)
        self.logsoftmax = torch.nn.LogSoftmax()

        self.init_weight()

    # Obtain latent variable in three modes determined by train_mode:
    # train_mode = 0: Evaluate on the validation set, here we use the mean of
    # posterior distribution of v and mean of generative distribution of u^k
    # train_mode = 1: Train on the trainning set, here we use reparameter trick to
    # sample v,u^k from their posterior distributions
    # train_mode = 2: Evaluate on the trainning set when calculate the propensities,
    # here we use the means of posterior distributions of v and u^k
    def cal_user_embedding(self, user_items_list,user_rating_list,train_mode):
        user_item_emb_mu_v = self.embedding_item_mu_v(user_items_list)
        user_rate_emb_mu_v = self.embedding_rate_mu_v(user_rating_list)
        user_emb_mu_v = user_item_emb_mu_v + user_rate_emb_mu_v

        if train_mode > 0:
            user_emb_mu_v = user_emb_mu_v.mean(0)[None, :]
            user_items_emb_sigma_v = self.embedding_item_sigma_v(user_items_list)
            user_rating_emb_sigma_v = self.embedding_rate_mu_v(user_rating_list)
            if train_mode == 1:
                user_emb_sigma_v = torch.exp(user_items_emb_sigma_v + user_rating_emb_sigma_v)
                user_emb_sigma_v = user_emb_sigma_v.mean(0)[None, :]
                user_emb_v = self.sample_latent(user_emb_mu_v, user_emb_sigma_v)
                user_emb_v = user_emb_v.repeat(user_items_list.size()[0],1)
            else:
                user_emb_v = user_emb_mu_v.repeat(user_items_list.size()[0], 1)
        else:
            user_emb_mu_v = user_emb_mu_v.mean(0)
            user_emb_v = user_emb_mu_v

        user_item_emb_u_p = self.embedding_item_u_p(user_items_list)
        user_rate_emb_u_p = self.embedding_rate_u_p(user_rating_list)
        user_rnn_input = torch.cat((user_item_emb_u_p,user_rate_emb_u_p),1)
        user_rnn_input = user_rnn_input[:,None,:]
        h_0 = torch.zeros((1,1,self.latent_dim))
        if self.device == "cuda": h_0 = h_0.cuda()
        output, hn = self.rnn(user_rnn_input, h_0)
        h_k = torch.cat((h_0,output[:-1,:,:]),0)[:,0,:]

        user_item_emb_mu_q_u = self.embedding_item_u_q(user_items_list)
        if train_mode:
            user_emb_q_mu_u = self.W_q_mu_u(torch.cat((h_k, user_item_emb_mu_q_u, user_emb_v), -1))
            user_emb_p_mu_u = self.W_p_mu_u(h_k)
            if train_mode == 1:
                user_emb_q_sigma_u = torch.exp(self.W_q_sigma_u(torch.cat((h_k, user_item_emb_mu_q_u, user_emb_v), -1)))
                user_emb_p_sigma_u = torch.exp(self.W_p_sigma_u(h_k))
                user_emb_q_u = self.sample_latent(user_emb_q_mu_u,user_emb_q_sigma_u)
                return user_emb_q_u, user_emb_v, user_emb_p_mu_u, user_emb_p_sigma_u, user_emb_q_mu_u, user_emb_q_sigma_u, user_emb_mu_v, user_emb_sigma_v
            else:
                user_emb_q_u = user_emb_q_mu_u
                return user_emb_q_u, user_emb_v
        else:
            user_emb_p_mu_u = self.W_p_mu_u(output[-1, 0, :][None, :])
            user_emb_p_u = user_emb_p_mu_u[0, :]
            user_emb = torch.cat((user_emb_p_u, user_emb_v), 0)[None, :]
            return user_emb

    def sample_latent(self, mu, sigma2):
        # Return the latent normal sample z ~ N(mu, sigma^2)
        sigma = torch.sqrt(sigma2)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()
        if self.device == "cuda": std_z = std_z.cuda()

        return mu + sigma * Variable(std_z, requires_grad=False)  # Reparameterization trick

    def forward(self, user_items_list,user_rating_list, if_train):
        user_emb_u, user_emb_v, user_emb_p_mu_u, user_emb_p_sigma_u, user_emb_q_mu_u, user_emb_q_sigma_u,user_emb_mu_v,user_emb_sigma_v = self.cal_user_embedding(user_items_list,user_rating_list,1)
        user_emb = torch.cat((user_emb_u,user_emb_v),1)
        item_scores = self.affine_output(user_emb) + self.bias_item
        item_logsoftmax = self.logsoftmax(item_scores)
        return item_logsoftmax, user_emb_p_mu_u, user_emb_p_sigma_u, user_emb_q_mu_u, user_emb_q_sigma_u,user_emb_mu_v,user_emb_sigma_v

    def forward_evaluate(self, user_items_list,user_rating_list):
        user_emb_u, user_emb_v = self.cal_user_embedding(user_items_list,user_rating_list, 2)
        user_emb = torch.cat((user_emb_u,user_emb_v),1)
        scores_raw = self.affine_output(user_emb)
        scores = scores_raw + self.bias_item
        return scores,scores - scores_raw

    def init_weight(self):
        self.embedding_item_mu_v.weight.data.normal_(0, 1.0 / self.embedding_item_mu_v.embedding_dim)
        self.embedding_rate_mu_v.weight.data.normal_(0,1.0 / self.embedding_rate_mu_v.out_features)
        self.embedding_item_sigma_v.weight.data.normal_(0, 1.0 / self.embedding_item_sigma_v.embedding_dim)
        self.embedding_rate_sigma_v.weight.data.normal_(0,1.0 / self.embedding_rate_sigma_v.out_features)
        self.embedding_item_u_p.weight.data.normal_(0, 1.0 / self.embedding_item_u_p.embedding_dim)
        self.embedding_rate_u_p.weight.data.normal_(0, 1.0 / self.embedding_rate_u_p.in_features)
        self.embedding_item_u_q.weight.data.normal_(0, 1.0 / self.embedding_item_u_q.embedding_dim)

        torch.nn.init.eye_(self.W_p_mu_u.weight.data)
        self.W_p_sigma_u.weight.data.normal_(0, 1.0 / self.W_p_sigma_u.in_features)
        torch.nn.init.eye_(self.W_q_mu_u.weight.data)
        self.W_q_sigma_u.weight.data.normal_(0, 1.0 / self.W_q_sigma_u.in_features)

        self.affine_output.weight.data.normal_(0, 1.0 / self.affine_output.in_features)