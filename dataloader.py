from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


# Sensor
class Node():
    def __init__(self, id, dist, target_node):
        self.id = id
        self.dist = dist
        self.target_node = target_node

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        return self.id != other.id

    def __lt__(self, other):
        return self.dist < other.dist

    def __le__(self, other):
        return self.dist <= other.dist

    def __gt__(self, other):
        return self.dist > other.dist

    def __ge__(self, other):
        return self.dist >= other.dist

    def __repr__(self):
        return f'Node id: {self.id} connectivaty with Node {self.target_node}: {self.dist:.4f}'


class STNN_Dataset(Dataset):
    # Downsample timesteps for training
    def __init__(self, samples_path, targets_path):
        self.samples = np.load(samples_path)
        self.targets = np.load(targets_path)
        assert self.samples.shape[0] == self.targets.shape[0]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.samples[idx], self.targets[idx]


def load_data(data_path):
    adj_mat_path = (os.path.join(data_path, r"adj_mat.npy"))
    feature_path = (os.path.join(data_path, r"node_values.npy"))

    A = np.load(os.path.normpath(adj_mat_path))
    X = np.load(os.path.normpath(feature_path))

    A = A.astype(np.float32)
    X = X.astype(np.float32)

    return A, X


def find_near_neighbors(A, target_node):

    nodes_row = A[:, target_node]
    nodes_col = A[target_node, :]
    nodes_row_idx = np.where(np.logical_and(nodes_row > 0, nodes_row < 1))[0]
    nodes_col_idx = np.where(np.logical_and(nodes_col > 0, nodes_col < 1))[0]

    node_list = [Node(target_node, nodes_row[target_node], target_node)]
    for each in nodes_row_idx:
        node = Node(each, nodes_row[each], target_node)
        node_list.append(node)
    for each in nodes_col_idx:
        node = Node(each, nodes_col[each], target_node)
        if node not in node_list:
            node_list.append(Node(each, nodes_col[each], target_node))

    node_list = sorted(node_list, reverse=True)

    return node_list


def find_near_neighbors_dynamic(A, target_node):

    nodes_row = np.max(A[:, :, target_node], axis=0)
    nodes_col = np.max(A[:, target_node, :], axis=0)
    nodes_row_idx = np.where(np.logical_and(nodes_row > 0, nodes_row < 1))[0]
    nodes_col_idx = np.where(np.logical_and(nodes_col > 0, nodes_col < 1))[0]

    node_list = [Node(target_node, nodes_row[target_node], target_node)]
    for each in nodes_row_idx:
        node = Node(each, nodes_row[each], target_node)
        node_list.append(node)
    for each in nodes_col_idx:
        node = Node(each, nodes_col[each], target_node)
        if node not in node_list:
            node_list.append(Node(each, nodes_col[each], target_node))

    node_list = sorted(node_list, reverse=True)

    return node_list


def prepare_samples_targets_list_static(A, X, num_nearby_nodes, t_in, t_out, keep_ratio=0.1, debug_flag=False,
                                        target_nodes='all'):
    traffic_feature_idx = 0  # speed must be the first feature

    num_nodes = X.shape[0]
    num_times = X.shape[1]
    num_feats = X.shape[2] + 1

    if target_nodes != 'all':
        target_node_list = target_nodes
    else:
        target_node_list = tqdm(range(num_nodes))  # all nodes

    samples, targets = [], []
    for node in target_node_list:
        target_node = node
        node_list = find_near_neighbors(A, target_node=target_node)

        # Sufficient neighbors, keep nearest k nodes only
        if len(node_list) > num_nearby_nodes:
            node_list = node_list[:num_nearby_nodes]

        # Flatten ids and distances
        node_ids = [each.id for each in node_list]
        node_connectivity = [each.dist for each in node_list]

        # Downsampling in time dimension if necessary
        if keep_ratio != 1:
            all_times = range(num_times - (t_in + t_out))
            downsample_size = min(int(num_times * keep_ratio), len(all_times))
            choosen = np.random.choice(all_times, downsample_size)
            indices = [(i, i + (t_in + t_out)) for i in choosen]
        else:  # keep all data
            indices = [(i, i + (t_in + t_out)) for i in range(num_times - (t_in + t_out))]

        # Convert data into training samples and targets
        for i, j in indices:
            sample = np.zeros((num_nearby_nodes, t_in, num_feats))
            sample[:len(node_ids), :, :num_feats - 1] = X[node_ids, i: i + t_in, :]
            sample[:len(node_ids), :, num_feats - 1] = np.array(node_connectivity).reshape(len(node_connectivity), 1)
            target = X[target_node, i + t_in: j, traffic_feature_idx]

            samples.append(sample)
            targets.append(target)

    return samples, targets


def prepare_samples_targets_list_dynamic(A, X, num_nearby_nodes, t_in, t_out, keep_ratio=0.1, debug_flag=False,
                                         target_nodes='all'):
    traffic_feature_idx = 0  # speed must be the first feature

    num_nodes = X.shape[0]
    num_times = X.shape[1]
    num_feats = X.shape[2] + 1

    samples, targets = [], []

    if target_nodes != 'all':
        node_list = target_nodes
    else:
        node_list = tqdm(range(num_nodes))  # all nodes

    for node in node_list:
        target_node = node
        node_list = find_near_neighbors_dynamic(A, target_node=target_node)

        # Sufficient neighbors, keep nearest k nodes only
        if len(node_list) > num_nearby_nodes:
            node_list = node_list[:num_nearby_nodes]

        # Flatten ids and distances
        node_ids = [each.id for each in node_list]
        # Downsampling in time dimension if necessary
        if keep_ratio != 1:
            all_times = range(num_times - (t_in + t_out))
            downsample_size = min(int(num_times * keep_ratio), len(all_times))
            choosen = np.random.choice(all_times, downsample_size)
            indices = [(i, i + (t_in + t_out)) for i in choosen]
        else:  # keep all data
            indices = [(i, i + (t_in + t_out)) for i in range(num_times - (t_in + t_out))]

        # Convert data into training samples and targets
        for i, j in indices:
            sample = np.zeros((num_nearby_nodes, t_in, num_feats))
            sample[:len(node_ids), :, :num_feats - 1] = X[node_ids, i: i + t_in, :]
            sample[:len(node_ids), :, num_feats - 1] = np.transpose(A[i:i + t_in, node_ids, target_node])
            target = X[target_node, i + t_in: j, traffic_feature_idx]

            samples.append(sample)
            targets.append(target)

    return samples, targets


# %% Convert X,A to a collection of samples, targets
def preprocess_dataset(data, t_in=12, t_out=3, num_nearby_nodes=15, keep_ratio=0.2,
                       train=True, val=True, test=True, debug=False, target_nodes='all', test_flag=False):
    # 70% train, 20% validation, 10% test
    assert isinstance(data, str)

    if target_nodes != 'all':  # Predict on node-of-interest only
        train_samples_path = os.path.join(data, rf'train_samples_{t_in}_{t_out}_{keep_ratio}_{num_nearby_nodes}_vtar{target_nodes}.npy')
        train_targets_path = os.path.join(data, rf'train_targets_{t_in}_{t_out}_{keep_ratio}_{num_nearby_nodes}_vtar{target_nodes}.npy')
        val_samples_path = os.path.join(data, rf'val_samples_{t_in}_{t_out}_{num_nearby_nodes}_vtar{target_nodes}.npy')
        val_targets_path = os.path.join(data, rf'val_targets_{t_in}_{t_out}_{num_nearby_nodes}_vtar{target_nodes}.npy')
        test_samples_path = os.path.join(data, rf'test_samples_{t_in}_{t_out}_{num_nearby_nodes}_vtar{target_nodes}.npy')
        test_targets_path = os.path.join(data, rf'test_targets_{t_in}_{t_out}_{num_nearby_nodes}_vtar{target_nodes}.npy')
    else:  # Predict on all nodes
        train_samples_path = os.path.join(data, rf'train_samples_{t_in}_{t_out}_{keep_ratio}_{num_nearby_nodes}.npy')
        train_targets_path = os.path.join(data, rf'train_targets_{t_in}_{t_out}_{keep_ratio}_{num_nearby_nodes}.npy')
        val_samples_path = os.path.join(data, rf'val_samples_{t_in}_{t_out}_{num_nearby_nodes}.npy')
        val_targets_path = os.path.join(data, rf'val_targets_{t_in}_{t_out}_{num_nearby_nodes}.npy')
        test_samples_path = os.path.join(data, rf'test_samples_{t_in}_{t_out}_{num_nearby_nodes}.npy')
        test_targets_path = os.path.join(data, rf'test_targets_{t_in}_{t_out}_{num_nearby_nodes}.npy')

    # Check if data already exists
    if os.path.isfile(train_samples_path) and os.path.isfile(train_targets_path) \
            and os.path.isfile(val_samples_path) and os.path.isfile(val_targets_path) \
            and os.path.isfile(test_samples_path) and os.path.isfile(test_targets_path):
        return train_samples_path, train_targets_path, \
               val_samples_path, val_targets_path, \
               test_samples_path, test_targets_path

    if test_flag and os.path.isfile(test_samples_path) and os.path.isfile(test_targets_path):
        return train_samples_path, train_targets_path, \
               val_samples_path, val_targets_path, \
               test_samples_path, test_targets_path

    print("Cannot find the sub-spacetime data, will prepare and save the data in proper format...")

    A, X = load_data(data)
    print('A shape:', A.shape)
    print('X shape:', X.shape)

    # Take care of static/dynamic topology
    if len(A.shape) == 2:  # fixed topology
        prepare_samples_targets_list = prepare_samples_targets_list_static
    elif len(A.shape) == 3:  # dynamic topology
        prepare_samples_targets_list = prepare_samples_targets_list_dynamic
    else:
        raise Exception('Unsupport adj matrix shape')

    # Split train/val/test
    print('Prepare train/val/test dataset')
    cut_point1 = int(X.shape[1] * 0.7)
    cut_point2 = int(X.shape[1] * 0.9)
    train_X = X[:, :cut_point1, :]
    val_X = X[:, cut_point1:cut_point2, :]
    test_X = X[:, cut_point2:, :]
    train_samples, train_targets = prepare_samples_targets_list(A, train_X, num_nearby_nodes, t_in, t_out,
                                                                keep_ratio=keep_ratio,
                                                                debug_flag=debug, target_nodes=target_nodes)
    val_samples, val_targets = prepare_samples_targets_list(A, val_X, num_nearby_nodes, t_in, t_out,
                                                            keep_ratio=1,
                                                            debug_flag=debug, target_nodes=target_nodes)
    test_samples, test_targets = prepare_samples_targets_list(A, test_X, num_nearby_nodes, t_in, t_out,
                                                              keep_ratio=1,
                                                              debug_flag=debug, target_nodes=target_nodes)

    if train:
        print('Saving train set to disk...')
        np.save(train_samples_path, np.array(train_samples))
        np.save(train_targets_path, np.array(train_targets))
    if val:
        print('Saving val set to disk...')
        np.save(val_samples_path, np.array(val_samples))
        np.save(val_targets_path, np.array(val_targets))
    if test:
        print('Saving test set to disk...')
        np.save(test_samples_path, np.array(test_samples))
        np.save(test_targets_path, np.array(test_targets))

    return train_samples_path, train_targets_path, \
           val_samples_path, val_targets_path, \
           test_samples_path, test_targets_path


def preprocess_datasets(data, t_in=12, t_out=3, num_nearby_nodes=15, keep_ratio=0.1,
                        train=True, val=True, test=True, debug=False, test_flag=False):
    assert isinstance(data, list)

    if debug:
        # Train set path
        train_samples_path = os.path.join(data[0],
                                          rf'combined_train_samples_{t_in}_{t_out}_{keep_ratio}_{num_nearby_nodes}_debug.npy')
        train_targets_path = os.path.join(data[0],
                                          rf'combined_train_targets_{t_in}_{t_out}_{keep_ratio}_{num_nearby_nodes}_debug.npy')
        # Validtion set path
        val_samples_path = os.path.join(data[0], rf'combined_val_samples_{t_in}_{t_out}_{num_nearby_nodes}_debug.npy')
        val_targets_path = os.path.join(data[0], rf'combined_val_targets_{t_in}_{t_out}_{num_nearby_nodes}_debug.npy')
        # Test set path
        test_samples_path = os.path.join(data[0], rf'combined_test_samples_{t_in}_{t_out}_{num_nearby_nodes}_debug.npy')
        test_targets_path = os.path.join(data[0], rf'combined_test_targets_{t_in}_{t_out}_{num_nearby_nodes}_debug.npy')
    else:
        # Train set path
        train_samples_path = os.path.join(data[0],
                                          rf'combined_train_samples_{t_in}_{t_out}_{keep_ratio}_{num_nearby_nodes}.npy')
        train_targets_path = os.path.join(data[0],
                                          rf'combined_train_targets_{t_in}_{t_out}_{keep_ratio}_{num_nearby_nodes}.npy')
        # Validtion set path
        val_samples_path = os.path.join(data[0], rf'combined_val_samples_{t_in}_{t_out}_{num_nearby_nodes}.npy')
        val_targets_path = os.path.join(data[0], rf'combined_val_targets_{t_in}_{t_out}_{num_nearby_nodes}.npy')
        # Test set path
        test_samples_path = os.path.join(data[0], rf'combined_test_samples_{t_in}_{t_out}_{num_nearby_nodes}.npy')
        test_targets_path = os.path.join(data[0], rf'combined_test_targets_{t_in}_{t_out}_{num_nearby_nodes}.npy')

    # Check if data already exists
    if os.path.isfile(train_samples_path) and os.path.isfile(train_targets_path) \
            and os.path.isfile(val_samples_path) and os.path.isfile(val_targets_path) \
            and os.path.isfile(test_samples_path) and os.path.isfile(test_targets_path):
        return train_samples_path, train_targets_path, \
               val_samples_path, val_targets_path, \
               test_samples_path, test_targets_path

    if test_flag and os.path.isfile(test_samples_path) and os.path.isfile(test_targets_path):
        return train_samples_path, train_targets_path, \
               val_samples_path, val_targets_path, \
               test_samples_path, test_targets_path

    print("Cannot find the sub-spacetime data, will prepare and save the data in proper format...")

    train_samples = []
    train_targets = []
    val_samples = []
    val_targets = []
    test_samples = []
    test_targets = []
    for each in data:
        A, X = load_data(each)
        print('A shape:', A.shape)
        print('X shape:', X.shape)

        # Take care of static/dynamic topology
        if len(A.shape) == 2:  # fixed topology
            prepare_samples_targets_list = prepare_samples_targets_list_static
        elif len(A.shape) == 3:  # dynamic topology
            prepare_samples_targets_list = prepare_samples_targets_list_dynamic
        else:
            raise Exception('Unsupport adj matrix shape')

        # Split train/val/test
        print('Prepare train/val/test dataset')
        cut_point1 = int(X.shape[1] * 0.7)
        cut_point2 = int(X.shape[1] * 0.9)
        train_X = X[:, :cut_point1, :]
        val_X = X[:, cut_point1:cut_point2, :]
        test_X = X[:, cut_point2:, :]
        sub_train_samples, sub_train_targets = prepare_samples_targets_list(A, train_X, num_nearby_nodes, t_in, t_out,
                                                                            keep_ratio=keep_ratio,
                                                                            debug_flag=debug)
        sub_val_samples, sub_val_targets = prepare_samples_targets_list(A, val_X, num_nearby_nodes, t_in, t_out,
                                                                        keep_ratio=1,
                                                                        debug_flag=debug)
        sub_test_samples, sub_test_targets = prepare_samples_targets_list(A, test_X, num_nearby_nodes, t_in, t_out,
                                                                          keep_ratio=1,
                                                                          debug_flag=debug)
        train_samples = train_samples + sub_train_samples
        train_targets = train_targets + sub_train_targets
        val_samples = val_samples + sub_val_samples
        val_targets = val_targets + sub_val_targets
        test_samples = test_samples + sub_test_samples
        test_targets = test_targets + sub_test_targets

    if train:
        print('Saving train set to disk...')
        np.save(train_samples_path, np.array(train_samples))
        np.save(train_targets_path, np.array(train_targets))
    if val:
        print('Saving val set to disk...')
        np.save(val_samples_path, np.array(val_samples))
        np.save(val_targets_path, np.array(val_targets))
    if test:
        print('Saving test set to disk...')
        np.save(test_samples_path, np.array(test_samples))
        np.save(test_targets_path, np.array(test_targets))

    return train_samples_path, train_targets_path, \
           val_samples_path, val_targets_path, \
           test_samples_path, test_targets_path
