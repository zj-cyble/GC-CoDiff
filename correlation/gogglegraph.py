import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler, LabelEncoder
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
class LearnedGraph(nn.Module):
    def __init__(self, input_dim, device):
        super(LearnedGraph, self).__init__()
        self.graph = nn.Parameter(
            torch.randn(input_dim, input_dim, requires_grad=True, device=device)
        )
        self.device = device
    def forward(self):
        adj = (self.graph + self.graph.T) / 2
        adj = torch.sigmoid(adj)
        adj = adj * (1 - torch.eye(adj.shape[0], device=self.device))
        return adj
def train_learned_graph_from_csv(csv_path, discrete_cols_idx, continuous_cols_idx, 
                                 num_epochs=1000, lr=1e-2, repeats=5, reg=1e-3, 
                                 device='cuda' if torch.cuda.is_available() else 'cpu'):
    df = pd.read_csv(csv_path).dropna().reset_index(drop=True)
    all_cols = continuous_cols_idx + discrete_cols_idx
    X = df.iloc[:, all_cols].copy()
    for idx in discrete_cols_idx:
        col = df.columns[idx]
        le = LabelEncoder()
        X[col] = le.fit_transform(df[col].astype(str))
    scaler = StandardScaler()
    X.iloc[:, :len(continuous_cols_idx)] = scaler.fit_transform(X.iloc[:, :len(continuous_cols_idx)])
    X_tensor = torch.tensor(X.values.T, dtype=torch.float32, device=device)
    input_dim = X_tensor.shape[0]
    best_loss = float('inf')
    best_adj = None

    for repeat in range(repeats):
        print(f"\n🔁 Repeat {repeat + 1}/{repeats}")
        set_seed(42 + repeat)
        model = LearnedGraph(input_dim=input_dim, device=device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            adj = model()
            X_recon = torch.matmul(adj, X_tensor)
            loss = ((X_recon - X_tensor) ** 2).mean() + reg * torch.norm(adj, p='fro')  # 加正则化
            loss.backward()
            optimizer.step()
        final_loss = loss.item()
        if final_loss < best_loss:
            best_loss = final_loss
            best_adj = model().detach().cpu().numpy()
    diag = np.diag(best_adj)
    if np.allclose(diag, 0):
        np.fill_diagonal(best_adj, 1.0)
    else:
        print("Diagonal elements are not all zero.")
    return best_adj

def reorder_columns(df, continuous_cols, discrete_cols):
    selected_cols = continuous_cols + discrete_cols
    dff = df[selected_cols].copy()
    return dff
if __name__ == "__main__":
    discrete_cols_idx = [1,3,4,5,6,7,10,11]
    continuous_cols_idx = [0,2,8,9]
    learned_adj = train_learned_graph_from_csv(
        csv_path="/home/data/shoppers1.csv",
        discrete_cols_idx=discrete_cols_idx,
        continuous_cols_idx=continuous_cols_idx,
        num_epochs=1000,
        lr=1e-2,
        device='cuda'  # 或 'cpu'
    )
    np.save("/home/graph_data/similarity_matrix_column_sho.npy", learned_adj)


