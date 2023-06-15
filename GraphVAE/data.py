'''
@File    :   data.py
@Time    :   2023/06/08 00:22:04
@Author  :   Ming 
'''

import torch
import numpy as np
import networkx as nx
from torch.utils.data import Dataset
from typing import Literal

Feature = Literal["id", "deg", "struct"]


class GraphAdjSampler(Dataset):

  def __init__(self, G_list: list, max_num_nodes: int, features: Feature = 'id') -> None:
    super().__init__()
    self.max_num_nodes = max_num_nodes
    self.adj_all = []
    self.len_all = []
    self.feature_all = []
    for G in G_list:
      adj = nx.to_numpy_matrix(G)
      # the diagonal entries are 1 since they denote node probability
      self.adj_all.append(np.asarray(adj) + np.identity(G.number_of_nodes()))
      self.len_all.append(G.number_of_nodes())
      if features == "id":
        # [max_num_nodes,max_num_nodes]
        self.feature_all.append(np.identity(max_num_nodes))
      elif features == "deg":
        degs = np.sum(np.array(adj), 1)
        degs = np.expand_dims(np.pad(degs, [0, max_num_nodes - G.number_of_nodes()], 0), axis=1)
        # [1,max_num_nodes]
        self.feature_all.append(degs)
      elif features == "struct":
        degs = np.sum(np.array(adj), 1)
        degs = np.expand_dims(np.pad(degs, [0, max_num_nodes - G.number_of_nodes()], 'constant'), axis=1)
        clusterings = np.array(list(nx.clustering(G).values()))
        clusterings = np.expand_dims(np.pad(clusterings, [0, max_num_nodes - G.number_of_nodes()], 'constant'), axis=1)
        # [2,max_num_nodes]
        self.feature_all.append(np.hstack([degs, clusterings]))

  def __len__(self):
    return len(self.adj_all)

  def __getitem__(self, index):
    adj = self.adj_all[index]
    num_nodes = adj.shape[0]
    adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))
    adj_padded[:num_nodes, :num_nodes] = adj
    adj_decoded = np.zeros(self.max_num_nodes * (self.max_num_nodes + 1) // 2)
    node_idx = 0
    # flat adjacency matrix into a vector
    adj_vectorized = adj_padded[np.triu(np.ones((self.max_num_nodes, self.max_num_nodes))) == 1]
    return {"adj": adj_padded, "adj_decoded": adj_vectorized, "features": self.feature_all[index].copy()}
