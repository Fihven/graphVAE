"""
@File    :   model.py
@Time    :   2023/06/07 16:40:00
@Author  :   Ming 
"""

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.optimize import linear_sum_assignment


# GCNConv
class GraphConv(nn.Module):

  def __init__(self, in_dim: int, out_dim: int) -> None:
    super(GraphConv, self).__init__()
    self.in_dim = in_dim
    self.out_dim = out_dim
    self.weight = nn.Parameter(torch.FloatTensor(in_dim, out_dim))

  def forward(self, x: Tensor, adj: Tensor):
    # calculate AX
    y = torch.matmul(adj, x)
    # calculate AXW
    y = torch.matmul(y, self.weight)
    return y


class MLP_VAE(nn.Module):

  def __init__(self, input_size: int, embedding_size: int, output_size: int) -> None:
    super(MLP_VAE, self).__init__()
    self.encode_mu = nn.Linear(input_size, embedding_size)
    self.encode_logstd = nn.Linear(input_size, embedding_size)
    # transform
    self.decode_1 = nn.Linear(embedding_size + input_size, embedding_size)
    # make edge prediction (reconstruction)
    self.decode_2 = nn.Linear(embedding_size, output_size)
    self.relu = nn.ReLU()
    # initialization
    for m in self.modules():
      if isinstance(m, nn.Linear):
        m.weight.data = nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))

  def reparameterize(self, mu: Tensor, logstd: Tensor):
    if self.training:
      return mu + torch.randn_like(logstd) + torch.exp(logstd)
    else:
      return mu

  def forward(self, h: Tensor):
    """forward

        Args:
            h (Tensor): embedding, [1,input_size]
        """
    # encode, [1,embedding_size]
    __mu__ = self.encode_mu(h)
    __logstd__ = self.encode_logstd(h)
    # reparameterize, [1,embedding_size]
    z = self.reparameterize(__mu__, __logstd__)
    # decode, [1,output_size]
    y = self.decode_1(torch.cat((h, z), dim=-1))
    y = self.relu(y)
    y = self.decode_2(y)
    return y, __mu__, __logstd__


class GraphVAE(nn.Module):

  def __init__(
      self,
      input_dim: int,
      hidden_dim: int,
      latent_dim: int,
      max_num_nodes: int,
      pool: str = "sum",
  ) -> None:
    """init

        Args:
            input_dim (int): input feature dimension for node
            hidden_dim (int): hidden dim for 2-layer gcn
            latent_dim (int): dimension of the latent representation of graph
            max_num_nodes (int): max number of nodes
            pool (str, optional): pooling strategy. Defaults to "sum".
        """
    super(GraphVAE, self).__init__()
    # self.conv1 = GraphConv(input_dim, hidden_dim)
    # self.bn1 = nn.BatchNorm1d(input_dim)
    # self.conv2 = GraphConv(hidden_dim, hidden_dim)
    # self.bn2 = nn.BatchNorm1d(input_dim)
    # self.act = nn.ReLU()

    # output_dim这样设置是需要将确保元素能够铺满邻接矩阵的上三角矩阵，给定N*N的矩阵，上三角矩阵的元素是N*(N+1)/2
    # VAE输出的是边的概率，对于N个节点，最多有N*(N+1)/2条边，所以output_dim必须是N*(N+1)/2
    # //为整数除法，返回的是整数
    output_dim = max_num_nodes * (max_num_nodes + 1) // 2
    self.vae = MLP_VAE(input_dim * input_dim, latent_dim, output_dim)
    self.max_num_nodes = max_num_nodes
    self.pool = pool
    for m in self.modules():
      if isinstance(m, GraphConv):
        m.weight.data = nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))
      elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def recover_adj_lower(self, l: Tensor):
    """recover upper triangular of adjacency given a tensor of edge probability

        Args:
            l (Tensor): edge probability tensor, [1,output_dim]
        """
    adj = torch.zeros(self.max_num_nodes, self.max_num_nodes)
    # torch.triu: Return the upper triangular part of a matrix
    adj[torch.triu(torch.ones(self.max_num_nodes, self.max_num_nodes)) == 1] = l
    return adj

  def recover_full_adj_from_lower(self, lower_adj: Tensor):
    """recover adjacency matrix given a upper triangular matrix

        Args:
            lower_adj (Tensor): upper triangular matrix
        """
    # get the diagonal element of lower_adj
    diagonal_elem = torch.diag(lower_adj, 0)
    # get the diagnoal matrix of `diagnoal_elem`
    diag = torch.diag(diagonal_elem)
    # transpose the 0 and 1 dim of lower_adj
    return lower_adj + torch.transpose(lower_adj, 0, 1) - diag

  def edge_similarity_matrix(
      self,
      adj: Tensor,
      adj_recon: Tensor,
      matching_features: Tensor,
      matching_features_recon: Tensor,
      sim_func: callable,
  ):
    """node pair similarity matrix S

        Args:
            adj (Tensor): adjacency matrix
            adj_recon (Tensor): reconstruct adjacency matrix
            matching_features (Tensor): feature matrix
            matching_features_recon (Tensor): reconstruct feature matrix
            sim_func (callable): similarity function for node feature
        """
    S = torch.zeros(
        self.max_num_nodes,
        self.max_num_nodes,
        self.max_num_nodes,
        self.max_num_nodes,
    )
    # 与论文不太一致
    for i in range(self.max_num_nodes):
      for j in range(self.max_num_nodes):
        if i == j:
          for a in range(self.max_num_nodes):
            # S((i,i),(a,a))=A_{i,i} * A_hat_{a,a} * (F_i,F_a)
            S[i, i, a, a] = (adj[i, i] * adj_recon[a, a] *
                             sim_func(matching_features[i], matching_features_recon[a]))
        else:
          for a in range(self.max_num_nodes):
            for b in range(self.max_num_nodes):
              if b == a:
                continue
              # S((i,j),(a,b)) = A_{i,j} * A_{i,i} * A_{j,j} * A_hat_{a,b} * A_hat_{a,a} * A_hat_{b,b}
              S[i, j, a, b] = (adj[i, j] * adj[i, i] * adj[j, j] * adj_recon[a, b] *
                               adj_recon[a, a] * adj_recon[b, b])
    return S

  def deg_feature_similarity(self, f1: Tensor, f2: Tensor):
    """node feature similarity function

        Args:
            f1 (Tensor): feature of node 1
            f2 (Tensor): feature of node 2
        """
    return 1 / (abs(f1 - f2) + 1)

  def mpm(self, x_init: Tensor, S: Tensor, max_iters: int = 50):
    """max-pooling matching

        Args:
            x_init (Tensor): initialization of x
            S (Tensor): similarity matrix
            max_iters (int, optional): max iterations. Defaults to 50.
        """
    x = x_init
    for _ in range(max_iters):
      x_new = torch.zeros(self.max_num_nodes, self.max_num_nodes)
      for i in range(self.max_num_nodes):
        for a in range(self.max_num_nodes):
          x_new[i, a] = x[i, a] * S[i, i, a, a]
          # j is the neighbor of node i
          pooled = [torch.max(x[j, :] * S[i, j, a, :]) for j in range(self.max_num_nodes) if j != i]
          neigh_sim = sum(pooled)
          x_new[i, a] += neigh_sim
      norm = torch.norm(x_new)
      x = x_new / norm
    return x

  def permute_adj(self, adj: Tensor, curr_ind: int, target_ind: int):
    """Permute adjacency matrix.
        The target_ind (connectivity) should be permuted to the curr_ind position.
        """
    # order curr_ind according to target ind
    ind = np.zeros(self.max_num_nodes, dtype=np.int64)
    ind[target_ind] = curr_ind
    adj_permuted = torch.zeros((self.max_num_nodes, self.max_num_nodes))
    adj_permuted[:, :] = adj[ind, :]
    adj_permuted[:, :] = adj_permuted[:, ind]
    return adj_permuted

  # def pool_graph(self, x):
  #   if self.pool == "max":
  #     out, _ = torch.max(x, dim=1, keepdim=False)
  #   elif self.pool == "sum":
  #     out = torch.sum(x, dim=1, keepdim=False)
  #   return out

  def adj_recon_loss(self, adj_truth: Tensor, adj_pred: Tensor):
    return F.binary_cross_entropy_with_logits(adj_truth, adj_pred)

  def forward(self, input_features: Tensor, adj: Tensor):
    """forward

        Args:
            input_feautres (Tensor): node features, [max_num_nodes,max_num_nodes]
            adj (Tensor): adjacency matrix
        """
    # pool over all nodes
    graph_h = input_features.view(-1, self.max_num_nodes * self.max_num_nodes)
    # vae, input dim: `max_num_nodes * max_num_nodes`
    h_decode, __mu__, __logstd__ = self.vae(graph_h)
    # out: [1,output_size]
    out = F.sigmoid(h_decode)
    out_tensor = out.cpu().data
    # reconstruct adjacency from edge probability
    recon_adj_upper = self.recover_adj_lower(out_tensor)
    recon_adj_tensor = self.recover_full_adj_from_lower(recon_adj_upper)

    # set matching features be degree
    recon_node_feautres = torch.sum(recon_adj_tensor, dim=1)
    adj_data = adj.cpu().data[0]
    node_features = torch.sum(adj_data, 1)

    # edge similarity matrix
    S = self.edge_similarity_matrix(
        adj_data,
        recon_adj_tensor,
        node_features,
        recon_node_feautres,
        self.deg_feature_similarity,
    )

    # initialization strategies
    init_corr = 1 / self.max_num_nodes
    init_assignment = torch.ones(self.max_num_nodes, self.max_num_nodes) * init_corr
    # matching
    assignment = self.mpm(init_assignment, S)

    # discretization using `Hungarian Algorithm`
    # use negative of the assignment score since the alg fins min cost flow
    row_ind, col_ind = linear_sum_assignment(-assignment.numpy())

    # permute adjacency (order row index according to col index)
    adj_permuted = self.permute_adj(adj_data, row_ind, col_ind)
    # flat a adjacency matrix to a edge vector
    adj_vectorized = adj_permuted[torch.triu(torch.ones(self.max_num_nodes, self.max_num_nodes)) ==
                                  1].squeeze_()
    adj_vectorized_var = Variable(adj_vectorized)

    # reconstruction loss
    adj_recon_loss = self.adj_recon_loss(adj_vectorized_var, out[0])
    # kl loss
    loss_kl = -0.5 * torch.sum(1 + __logstd__ - __mu__.pow(2) - __logstd__.exp())
    # normalize kl loss
    loss_kl /= self.max_num_nodes * self.max_num_nodes
    loss = adj_recon_loss + loss_kl
    return loss


if __name__ == "__main__":
  test = GraphVAE(4, 16, 32, 4)
  print(test)
  # out = torch.randn((1, 10))
  # print(test.recover_adj_lower(out))
