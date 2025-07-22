import torch
import numpy as np
from tabular_transformer import GeneralTransformer
import json
import logging
import os
import numpy as np
import pandas as pd

CATEGORICAL = "categorical"
CONTINUOUS = "continuous"

LOGGER = logging.getLogger(__name__)

DATA_PATH = os.path.join(os.path.dirname(__file__), 'tabular_datasets')

def _load_json(path):
    with open(path) as json_file:
        return json.load(json_file)


def _load_file(filename, loader):
    local_path = os.path.join(DATA_PATH, filename)
    
    if loader == np.load:
        return loader(local_path, allow_pickle=True)
    return loader(local_path)


def _get_columns(metadata):
    categorical_columns = list()

    for column_idx, column in enumerate(metadata['columns']):
        if column['type'] == CATEGORICAL:
            categorical_columns.append(column_idx)

    return categorical_columns


def load_data(name, benchmark=False):
    data = _load_file(name + '.npz', np.load)
    meta = _load_file(name + '.json', _load_json)

    categorical_columns = _get_columns(meta)
    train = data['train']
    test = data['test']


    return train, test, (categorical_columns, meta)

def get_dataset(FLAGS, evaluation=False):
    batch_size = FLAGS.training_batch_size if not evaluation else FLAGS.eval_batch_size

    if batch_size % torch.cuda.device_count() != 0:
        raise ValueError(f'Batch sizes ({batch_size} must be divided by'
                         f'the number of devices ({torch.cuda.device_count()})')
    train, test, cols = load_data(FLAGS.data)
    cols_idx = list(np.arange(train.shape[1]))
    dis_idx = cols[0]
    con_idx = [x for x in cols_idx if x not in dis_idx]
    full_negative_con = pd.read_csv(
        "/home/graph_data/negative_continuous_beijing0030.csv").to_numpy()
    full_negative_dis = pd.read_csv(
        "/home/graph_data/negative_discrete_beijing0030.csv").to_numpy()
    full_negative_con = full_negative_con[:, :10]
    full_negative_dis = full_negative_dis[:, 10:]
    train_size = train.shape[0]
    train_con = full_negative_con[:train_size]
    test_con = full_negative_con[train_size:]
    train_dis = full_negative_dis[:train_size]
    test_dis = full_negative_dis[train_size:]
    cat_idx_ = list(np.arange(train_dis.shape[1]))[:len(cols[0])]
    transformer_con = GeneralTransformer()
    transformer_dis = GeneralTransformer()
    transformer_con.fit(train_con, [])
    transformer_dis.fit(train_dis, cat_idx_)
    train_con_data = transformer_con.transform(train_con)
    train_dis_data = transformer_dis.transform(train_dis)
    return train, train_con_data, train_dis_data, test, (transformer_con, transformer_dis, cols[1]), con_idx, dis_idx

      