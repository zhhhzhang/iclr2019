import os.path as osp
import numpy as np
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from data import NodeDegreeFeatureDataLoader, SameFeatureDataLoader
from train import train

def prepare_config_for_dataset(config, dataset):
    if config.node_features == 'node_degree':
        get_max_node_degree
        dataset_loader = DataLoader(dataset, batch_size=config.batch_size)
        max_node_degree = 0
        min_node_degree = 1000000
        for data in dataset_loader:
            edge_index, _ = remove_self_loops(data.edge_index)
            row, col = edge_index
            out = scatter_add(torch.ones(len(data.batch))[col], row, dim=0, dim_size=len(data.batch))
            max_node_degree = max(max_node_degree, out.max())
        config.max_node_degree = max_node_degree
        config.num_features = max_node_degree + 1
    elif config.node_features == 'categorical':
        data = next(iter(dataset_loader))
        config.num_features = data.x.shape[1]
    elif config.node_features == 'same':
        config.num_features = config.same_feature_dim

def cross_validation(config):
    dataset_name = config.dataset_name
    dataset_path = osp.join(osp.dirname(os.getcwd()), '..', 'data', dataset_name)
    dataset = TUDataset(dataset_path, name=dataset_name).shuffle()

    prepare_config_for_dataset(config, dataset)

    cross_validation_batches = config.cross_validation_batches
    cross_validation_batch_size = len(dataset) // cross_validation_batches
    results = []
    train_histories = []
    test_histories = []
    for i in range(cross_validation_batches):
        start_index = i * cross_validation_batch_size
        end_index = (i + 1) * cross_validation_batch_size if i + 1 < cross_validation_batches else len(dataset)
        test_dataset = dataset[start_index:end_index] 
        train_dataset = dataset[:start_index] + dataset[end_index:]

        if config.node_features == 'categorical':
            test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
        elif config.node_features == 'node_degree':
            test_loader = NodeDegreeFeatureDataLoader(test_dataset, config.max_node_degree, batch_size=config.batch_size)
            train_loader = NodeDegreeFeatureDataLoader(train_dataset, config.max_node_degree, batch_size=config.batch_size)
        elif config.node_features == 'same':
            test_loader = SameFeatureDataLoader(test_dataset, config.same_feature_dim, batch_size=config.batch_size)
            train_loader = SameFeatureDataLoader(train_dataset, config.same_feature_dim, batch_size=config.batch_size)

        train_history, test_history = train(config, train_dataset, test_dataset) 
        train_histories.append(train_history)
        test_histories.append(test_history)
        results.append(np.max(test_history))

    avg = np.mean(results)
    std = np.std(results)
    return avg, std
