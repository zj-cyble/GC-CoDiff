import layers
# from . import layers
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

import pandas as pd
import torch
# from easydict import EasyDict  # 简化 FLAGS 结构
# from model import tabularUnet  # 假设你保存为 model.py

get_act = layers.get_act
default_initializer = layers.default_init

class tabularUnet(nn.Module):
    def __init__(self, FLAGS):
        super().__init__()

        self.embed_dim = FLAGS.nf
        tdim = self.embed_dim * 4
        self.act = get_act(FLAGS)

        modules = []
        modules.append(nn.Linear(self.embed_dim, tdim))
        modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
        nn.init.zeros_(modules[-1].bias)
        modules.append(nn.Linear(tdim, tdim))
        modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
        nn.init.zeros_(modules[-1].bias)
        self.all_modules = nn.ModuleList(modules)
        self.inputs = nn.Linear(FLAGS.input_size, list(FLAGS.encoder_dim)[0])  # input layer
        self.encoder = layers.Encoder(list(FLAGS.encoder_dim), tdim, FLAGS)  # encoder
        dim_in = list(FLAGS.encoder_dim)[-1]
        dim_out = list(FLAGS.encoder_dim)[-1]
        self.bottom_block = nn.Linear(dim_in, dim_out)
        self.decoder = layers.Decoder(list(reversed(FLAGS.encoder_dim)), tdim, FLAGS)
        dim_in = list(FLAGS.encoder_dim)[0]
        dim_out = FLAGS.output_size
        self.outputs = nn.Linear(dim_in, dim_out)

    def forward(self, xx, time_cond):
        modules = self.all_modules
        m_idx =0
        temb = layers.get_timestep_embedding(time_cond, self.embed_dim)
        temb = modules[m_idx](temb)
        m_idx += 1
        temb = self.act(temb)
        temb = modules[m_idx](temb)
        m_idx += 1
        inputs = self.inputs(xx.float())
        skip_connections, encoding = self.encoder(inputs, temb)
        normalized = F.normalize(encoding, p=2, dim=1)  # L2 normalization
        similarity_matrix = normalized @ normalized.T  # shape: (815, 815)
        similarity_np = similarity_matrix.cpu().detach().numpy()
        np.save("/home/graph_data/similarity_matrix_row_sho.npy", similarity_np)
        print("similarity_np.shape",similarity_np.shape)
        encoding = self.bottom_block(encoding)
        encoding = self.act(encoding)
        x = self.decoder(skip_connections, encoding, temb)
        outputs = self.outputs(x)
        return outputs

class FLAGS:
    input_size = 18
    output_size = 18
    encoder_dim = [128, 256]
    nf = 64
    activation = 'relu'

def reorder_columns(df, continuous_cols_idx, discrete_cols_idx):

    all_idx = continuous_cols_idx + discrete_cols_idx
    col_names = df.columns[all_idx]
    return df[col_names].copy()

def main():
    discrete_cols_idx = [9,10,11,15,16,17]
    continuous_cols_idx =  [0,1,2,3,4,5,6,7,8,12,13,14]
    df = pd.read_csv("/home/data/Shoppers1.csv")  # shape = (N, new_input_size)
    dff = reorder_columns(df, continuous_cols_idx, discrete_cols_idx)
    print(dff.head())
    xx = torch.tensor(dff.values, dtype=torch.float32)
    time_cond = torch.randint(0, 1000, (xx.shape[0],))  # shape = (N,)
    model = tabularUnet(FLAGS)
    outputs = model(xx, time_cond)
    print("Output shape:", outputs.shape)

if __name__ == "__main__":
    main()
