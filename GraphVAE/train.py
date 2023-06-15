'''
@File    :   train.py
@Time    :   2023/06/08 00:31:58
@Author  :   Ming 
'''

import argparse
import torch
import networkx as nx
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from model import GraphVAE
from data import GraphAdjSampler

LR_milestones = [500, 1000]


def build_model(args, max_num_nodes):
  if args.feature_type == "id":
    input_dim = max_num_nodes
  elif args.feature_type == "deg":
    input_dim = 1
  elif args.feature_type == "struct":
    input_dim = 2
  model = GraphVAE(input_dim, 64, 256, max_num_nodes)
  return model


def train(args, dataloader, model: nn.Module):
  epoch = 1
  optimizer = optim.Adam(model.parameters(), lr=args.lr)
  scheduler = MultiStepLR(optimizer, milestones=LR_milestones, gamma=args.lr)

  model.train()
  for epoch in range(10):
    for batch_idx, data in enumerate(dataloader):
      model.zero_grad()
      features = data["features"].float()
      adj_input = data["adj"].float()
      features = Variable(features)
      adj_input = Variable(adj_input)
      loss = model(features, adj_input)
      print('Epoch: ', epoch, ', Iter: ', batch_idx, ', Loss: ', loss.item())
      loss.backward()

      optimizer.step()
      scheduler.step()


def arg_parse():
  parser = argparse.ArgumentParser(description="GraphVAE")
  io_parser = parser.add_mutually_exclusive_group(required=False)
  io_parser.add_argument('--dataset', dest='dataset', help='Input dataset.')
  parser.add_argument('--lr', dest='lr', type=float, help='Learning rate.')
  parser.add_argument('--batch_size', dest='batch_size', type=int, help='Batch size.')
  parser.add_argument('--num_workers',
                      dest='num_workers',
                      type=int,
                      help='Number of workers to load data.')
  parser.add_argument(
      '--max_num_nodes',
      dest='max_num_nodes',
      type=int,
      help='Predefined maximum number of nodes in train/test graphs. -1 if determined by \
                training data.')
  parser.add_argument('--feature',
                      dest='feature_type',
                      help='Feature used for encoder. Can be: id, deg')
  parser.set_defaults(dataset="grid",
                      feature_type="id",
                      lr=0.001,
                      batch_size=1,
                      num_workers=1,
                      max_num_nodes=-1)
  return parser.parse_args()


def main():
  prog_args = arg_parse()

  ### running log

  if prog_args.dataset == 'enzymes':
    # Not implemented
    pass
    num_graphs_raw = len(graphs)
  elif prog_args.dataset == 'grid':
    graphs = []
    for i in range(2, 5):
      for j in range(2, 5):
        graphs.append(nx.grid_2d_graph(i, j))
    num_graphs_raw = len(graphs)

  if prog_args.max_num_nodes == -1:
    max_num_nodes = max([graphs[i].number_of_nodes() for i in range(len(graphs))])
  else:
    max_num_nodes = prog_args.max_num_nodes
    graphs = [g for g in graphs if g.number_of_nodes() <= max_num_nodes]

  graphs_len = len(graphs)
  print('Number of graphs removed due to upper-limit of number of nodes: ',
        num_graphs_raw - graphs_len)
  graphs_train = graphs

  print('total graph num: {}, training set: {}'.format(len(graphs), len(graphs_train)))
  print('max number node: {}'.format(max_num_nodes))

  dataset = GraphAdjSampler(graphs_train, max_num_nodes, features=prog_args.feature_type)
  dataset_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=prog_args.batch_size,
                                               num_workers=prog_args.num_workers)
  model = build_model(prog_args, max_num_nodes)
  train(prog_args, dataset_loader, model)


if __name__ == '__main__':
  main()
