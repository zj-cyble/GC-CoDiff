import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import networkx as nx
import os
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder, StandardScaler
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import pairwise_distances
from typing import Dict
from collections import defaultdict
import os
import glob
import numpy as np
import pandas as pd


DISCRETE_COLS_IDX = [1,6,7,8,9,14] 
CONTINUOUS_COLS_IDX = [0,2,3,5,4,10,11,12,13,] 
y_idx=[14]


def load_data(file_path):
    df = pd.read_csv(file_path)
    df.fillna(method='ffill', inplace=True)
    return df


def index_to_colnames(df, col_indices):
    return [df.columns[i] for i in col_indices]

def encode_features(df, col_names):
    df_encoded = df.copy()
    for col in col_names:
        if df_encoded[col].dtype == 'object':
            df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])
    return df_encoded

def compute_column_similarity_matrix(df: pd.DataFrame, col: str) -> np.ndarray:
    col_data = df[[col]].copy()
    N = len(col_data)

    if pd.api.types.is_numeric_dtype(col_data[col]) and col_data[col].nunique() > 10:
        scaler = StandardScaler()
        X = scaler.fit_transform(col_data)

        dists = pairwise_distances(X, metric='euclidean')  # shape [N, N]
        sim_matrix = 1 / (1 + dists)

    else:
        col_array = col_data[col].astype(str).values.reshape(-1, 1)
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        X = encoder.fit_transform(col_array)

        dists = pairwise_distances(X, metric='hamming')
        sim_matrix = 1 - dists

    return sim_matrix

def build_weight_matrices(df: pd.DataFrame, selected_cols: list) -> Dict[str, np.ndarray]:
    weight_matrices = {}
    for col in selected_cols:
        print(f"🔧 正在构建列 {col} 的相似度矩阵...")
        W = compute_column_similarity_matrix(df, col)
        weight_matrices[col] = W
    return weight_matrices

def compute_frequency_weight_matrices(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    构建基于频率的有向权重矩阵 w21 (i -> j), w22 (j -> i)，
    按照公式：w_ij = #(i与j同时出现某值) / #(j中该值的出现次数)
    """
    N = len(df)
    w21 = np.zeros((N, N))
    w22 = np.zeros((N, N))

    # 遍历每一列作为实体属性
    for col in df.columns:
        value_to_indices = {}

        # 统计每个值出现在哪些行
        for idx, val in df[col].items():
            value_to_indices.setdefault(val, set()).add(idx)

        # 对每个值构造组合（行对）
        for val, indices in value_to_indices.items():
            indices = list(indices)
            count = len(indices)

            for i in indices:
                for j in indices:
                    # i → j
                    w21[i][j] += 1 / count  # 归一化：以 j 为归一化目标
                    w22[j][i] += 1 / count  # 反向边

    return w21, w22

def build_graph(df, selected_cols):
    MAX_GROUP_SIZE = 1000
    BATCH_SIZE = 500
    G = nx.Graph()

    for idx, row in df.iterrows():
        G.add_node(idx, feature=row.values)

    if not selected_cols:
        print("⚠️ 警告：selected_cols 为空，未构建任何边")
        return G

    for col in selected_cols:
        groups = df.groupby(col).groups
        for value, indices in groups.items():
            indices = list(indices)
            if len(indices) > MAX_GROUP_SIZE:
                print(f"Value {value} in column {col} has too many nodes, splitting into batches")
                random.shuffle(indices)
                num_batches = (len(indices) + BATCH_SIZE - 1) // BATCH_SIZE
                for i in range(num_batches):
                    start = i * BATCH_SIZE
                    end = min((i + 1) * BATCH_SIZE, len(indices))
                    # 直接用切片调用，不创建额外变量
                    build_edges_with_batch(G, indices[start:end], col, df)
            else:
                build_edges_with_batch(G, indices, col, df)

    return G

def get_attention_matrix(column_names, dim=10):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(column_names).toarray()
    pca = PCA(n_components=dim)
    embeddings = pca.fit_transform(tfidf)
    sim_matrix = cosine_similarity(embeddings)
    attention = np.exp(sim_matrix) / np.sum(np.exp(sim_matrix), axis=1, keepdims=True)
    return attention  # shape: [num_columns, num_columns]

def build_edges_with_batch(G, indices, col, df):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    indices_tensor = torch.tensor(indices, device=device)
    n = len(indices_tensor)
    features = df.values

    for i in range(n):
        for j in range(i + 1, n):
            idx_i = indices_tensor[i].item()
            idx_j = indices_tensor[j].item()
            feat_i = features[idx_i].reshape(1, -1)
            feat_j = features[idx_j].reshape(1, -1)
            sim = cosine_similarity(feat_i, feat_j)[0][0]
            G.add_edge(idx_i, idx_j, attr=col, weight=sim)

def convert_to_pyg_data(G, column_names, attention_matrix):
    node_features = [attr["feature"] for _, attr in G.nodes(data=True)]
    x = torch.tensor(node_features, dtype=torch.float)

    edge_list = list(G.edges(data=True))
    edge_index = torch.tensor([[u, v] for u, v, _ in edge_list], dtype=torch.long).t().contiguous()

    # 计算加权后的 edge_attr
    edge_weights = []
    colname_to_idx = {col: idx for idx, col in enumerate(column_names)}

    for u, v, attr in edge_list:
        base_weight = attr.get("weight", 1.0)
        colname = attr.get("attr", None)
        if colname is not None and colname in colname_to_idx:
            col_idx = colname_to_idx[colname]
            attn = attention_matrix[col_idx]
            # 所有列投射到该列的加权和
            weighted = sum(attn[k] * base_weight for k in range(len(attn)))
        else:
            weighted = base_weight  # fallback
        edge_weights.append(weighted)

    edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)

    if edge_index.numel() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def reorder_columns(df, continuous_cols, discrete_cols):

    selected_cols = continuous_cols + discrete_cols
    dff = df[selected_cols].copy()
    return dff

import numpy as np
import os

def compute_and_save_columnwise_similarity_blocks(df, continuous_cols, discrete_cols,save_dir,block_size=1000, ):
    df = df.copy()
    m, n = df.shape
    all_cols = continuous_cols + discrete_cols
    assert set(all_cols) == set(df.columns), "输入列必须覆盖所有列，且顺序一致"

    os.makedirs(save_dir, exist_ok=True)
    
    num_blocks = (m + block_size - 1) // block_size

    for block_i in range(num_blocks):
        for block_j in range(num_blocks):
            start_i = block_i * block_size
            end_i = min((block_i + 1) * block_size, m)
            start_j = block_j * block_size
            end_j = min((block_j + 1) * block_size, m)

            rows_i = df.iloc[start_i:end_i]
            rows_j = df.iloc[start_j:end_j]
            bi, bj = end_i - start_i, end_j - start_j  # 实际块大小

            W_block = np.zeros((bi, bj, n))

            for k, col in enumerate(df.columns):
                values_i = rows_i[col].values
                values_j = rows_j[col].values
                sim = np.zeros((bi, bj))

                if col in continuous_cols:
                    d = []
                    for i in range(bi):
                        for j in range(bj):
                            d.append((values_i[i] - values_j[j]) ** 2)
                    d = np.array(d)
                    sigma = np.std(d)
                    sigma = sigma if sigma > 1e-8 else 1.0

                    for i in range(bi):
                        for j in range(bj):
                            dist_sq = (values_i[i] - values_j[j]) ** 2
                            sim[i, j] = np.exp(-dist_sq / (2 * sigma ** 2))

                elif col in discrete_cols:
                    for i in range(bi):
                        for j in range(bj):
                            sim[i, j] = 1.0 if values_i[i] == values_j[j] else 0.0

                else:
                    raise ValueError(f"列 {col} 既不在 continuous_cols 也不在 discrete_cols 中")

                W_block[:, :, k] = sim

            filename = f"W_block_{start_i}_{end_i}_{start_j}_{end_j}.npy"
            filepath = os.path.join(save_dir, filename)
            np.save(filepath, W_block)
            print(f"Saved: {filepath}")


def print_ones_in_first_four_columns(W):
    m, _, n = W.shape  # W shape: (m, m, n)

    for k in range(min(4, n)):  # 每一层
        for j in range(m):  # 仅前4列
            for i in range(m):  # 所有行
                if W[i, j, k] == 1.0:
                    print(f"(i={i}, j={j}, k={k})")


def compute_frequency_weights(df, continuous_cols, discrete_cols):
    m = len(df)
    all_cols = continuous_cols + discrete_cols
    n = len(all_cols)
    W = np.zeros((n, n, m))

    value_counts = {
        col: df[col].astype(str).value_counts().to_dict() for col in all_cols
    }

    for c in range(m):
        row = df.iloc[c]
        for a in range(n):
            col_a = all_cols[a]
            val_a = str(row[col_a])  # 转为字符串

            for b in range(n):
                col_b = all_cols[b]
                val_b = str(row[col_b])  # 转为字符串
                if val_a == val_b:
                    count_b = value_counts[col_b].get(val_b, 1e-8)  # 防止除0
                    W[a, b, c] = 1.0 / count_b
                else:
                    W[a, b, c] = 0.0


    return W

def compute_cooccurrence_weights(df: pd.DataFrame) -> np.ndarray:
    m, n = df.shape  # m: rows, n: columns
    W = np.zeros((n, n, m), dtype=float)  # n x n x m 权重矩阵

    col_value_counts = {col: defaultdict(int) for col in range(n)}
    cooccur_counts = {(a, b): defaultdict(int) for a in range(n) for b in range(n) if a != b}

    for row in df.itertuples(index=False):
        for a in range(n):
            val_a = row[a]
            col_value_counts[a][val_a] += 1  # 统计 a 列 val_a 的出现次数
            for b in range(n):
                if a == b:
                    continue
                val_b = row[b]
                cooccur_counts[(a, b)][(val_a, val_b)] += 1

    for c in range(m):
        row = df.iloc[c]
        for a in range(n):
            val_a = row[a]
            for b in range(n):
                if a == b:
                    continue
                val_b = row[b]
                co_key = (val_a, val_b)
                co_count = cooccur_counts[(a, b)].get(co_key, 0)
                b_count = col_value_counts[b].get(val_b, 1)  # 避免除以0
                W[a, b, c] = co_count / b_count

    return W



import pandas as pd
from tqdm import tqdm
import numpy as np
import os

def load_w1_column_for_row_pair_v2(k: int, col_index: int, m: int, n: int, save_dir: str, s1: np.ndarray, block_size: int = 1000):
    wijk = np.zeros(m)
    num_blocks = (m + block_size - 1) // block_size

    found = False

    for block_i in range(num_blocks):
        start_i = block_i * block_size
        end_i = min((block_i + 1) * block_size, m)

        if not (start_i <= k < end_i):
            continue  # 当前块不包含 k

        local_k = k - start_i

        for block_j in range(num_blocks):
            start_j = block_j * block_size
            end_j = min((block_j + 1) * block_size, m)

            filename = f"W_block_{start_i}_{end_i}_{start_j}_{end_j}.npy"
            filepath = os.path.join(save_dir, filename)

            if not os.path.exists(filepath):
                print(f"[WARNING] 块文件缺失: {filename}")
                continue

            try:
                W_block = np.load(filepath)  # shape: (bi, bj, n)
                bj = end_j - start_j

                for local_j in range(bj):
                    global_j = start_j + local_j
                    w_val = W_block[local_k, local_j, col_index]
                    s_val = s1[k, global_j]
                    wijk[global_j] = w_val * s_val

                found = True

            except Exception as e:
                print(f"[ERROR] 读取文件 {filename} 失败: {e}")
                continue

    if not found:
        print(f"[WARNING] 未找到包含第 {k} 行的任何块")

    return wijk


def generate_shuffled_negative_continuous_table_v2(
    dff: pd.DataFrame,
    w2: np.ndarray,
    continuous_cols: list,
    discrete_cols: list,
    s: int,
    save_dir: str,
    block_size: int = 1000
) -> pd.DataFrame:
    dff1 = dff.copy()
    m, n = dff.shape
    assert w2.shape == (n, n, m), f"w2 shape must为 ({n}, {n}, {m})"

    col_order = continuous_cols + discrete_cols
    dff1 = dff1[col_order]
    assert list(dff1.columns) == col_order

    # Step 1: 创建 mask：a∈[0,1,2,3], b∈[4,n)
    a_range = [0, 1, 2, 3]
    b_range = list(range(4, n))
    mask = np.zeros_like(w2, dtype=bool)
    for a in a_range:
        for b in b_range:
            mask[a, b, :] = True

    masked_values = w2[mask]
    all_indices = np.argwhere(mask)
    top_s_indices = np.argpartition(masked_values, -s)[-s:]
    sorted_top_s_indices = top_s_indices[np.argsort(masked_values[top_s_indices])[::-1]]
    top_coords = all_indices[sorted_top_s_indices]  # shape: (s, 3)

    print(f"[INFO] 开始生成负例，共采样 top-{s} 条")

    s1 = np.load("/home/graph_data/similarity_matrix_row_adu.npy")   # shape: [m, m]
    for a, b, c in tqdm(top_coords):
        k = a
        i = c
        wijk = load_w1_column_for_row_pair_v2(k, col_index=b, m=m, n=n, save_dir=save_dir, s1=s1,block_size=block_size)
        j_max = np.argmax(wijk)
        dff1.iat[j_max, a] = dff.iat[c, a]
    return dff1


def generate_shuffled_negative_discrete_table_v2(
    dff: pd.DataFrame,
    w2: np.ndarray,
    save_dir: str,
    s: int = 8000,
    output_path: str = "negative_discrete.csv"
) -> pd.DataFrame:
    dff1 = dff.copy()
    m, n = dff.shape

    assert w2.shape == (n, n, m), f"w2 shape must be ({n}, {n}, {m})"

    a_range = list(range(4, n))
    b_range = [0, 1, 2, 3]

    mask = np.zeros_like(w2, dtype=bool)
    for a in a_range:
        for b in b_range:
            mask[a, b, :] = True

    masked_values = w2[mask]
    all_indices = np.argwhere(mask)

    top_s_indices = np.argpartition(masked_values, -s)[-s:]
    sorted_top_indices = top_s_indices[np.argsort(masked_values[top_s_indices])[::-1]]
    top_coords = all_indices[sorted_top_indices]  # shape: (s, 3)

    npy_files = sorted(glob.glob(os.path.join(save_dir, "W_block_*.npy")))

    for a, b, c in top_coords:
        k = a
        i = c

        wijk = None
        j_max = None

        for filepath in npy_files:
            fname = os.path.basename(filepath)
            parts = fname.replace(".npy", "").split("_")
            start_i, end_i, start_j, end_j = map(int, parts[2:6])
            if not (start_i <= i < end_i):
                continue
            i_local = i - start_i
            W_block = np.load(filepath)
            wijk = W_block[i_local, :, k]
            j_local = np.argmax(wijk)
            j_max = start_j + j_local
            break

        if j_max is not None:
            dff1.iat[j_max, a] = dff.iat[c, a]

    dff1.to_csv(output_path, index=False)
    print(f"负例保存到：{output_path}")
    return dff1


def process_table_data(file_path):
    df = load_data(file_path)
    all_cols = df.columns.tolist()
    discrete_cols = index_to_colnames(df, DISCRETE_COLS_IDX)
    continuous_cols = index_to_colnames(df, CONTINUOUS_COLS_IDX)
    dff = reorder_columns(df, continuous_cols, discrete_cols)
    df_encoded = encode_features(dff, all_cols)

    print("🔧 正在构建列级别的权重矩阵...")
    compute_and_save_columnwise_similarity_blocks(df, continuous_cols, discrete_cols,save_dir="/home/data/adult1_similarity_blocks", block_size=1000,)
    w2 = compute_cooccurrence_weights(dff)
    s2 = np.load("/home/graph_data/similarity_matrix_column_adu.npy")   # shape: [m, m]
    assert w2.shape[0] == w2.shape[1] == s2.shape[0] == s2.shape[1], "维度不一致"
    w22 = w2 * s2[:, :, np.newaxis]   # shape: [m, m, n]
    dff_neg = generate_shuffled_negative_continuous_table_v2(
    dff,
    w2=w22,
    continuous_cols=continuous_cols,
    discrete_cols=discrete_cols,
    s=130000,
    save_dir = "/home/data/adult1_similarity_blocks",
    block_size=1000
    )
    dff_neg.to_csv("/home/graph_data/negative_continuous_adult1.csv", index=False)
    dff_discrete_n = generate_shuffled_negative_discrete_table_v2(
    dff=dff,
    w2=w22,
    save_dir="/home/data/adult1_similarity_blocks",
    s=130000,
    output_path="/home/graph_data/negative_discrete_adult1.csv"
)
    return w1,w2

if __name__ == "__main__":
    csv_file_path="/home/data/adult1.csv"
    w1,w2=process_table_data(csv_file_path)

