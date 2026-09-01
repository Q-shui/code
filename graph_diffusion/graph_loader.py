# graph_diffusion/graph_loader.py
import json
import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# =========================
# 数据结构（扩散模型输入）
# =========================

@dataclass
class GraphForDiffusion:
    """
    GraphRAG 实体关系图的“计算态”表示
    """
    x: torch.Tensor              # [N, 1024] 节点特征
    edge_index: torch.Tensor     # [2, E]    邻接关系
    edge_weight: torch.Tensor    # [E]       边权重
    edge_type: List[str]         # [E]       关系类型
    node_ids: List[str]          # [N]       index -> entity_name


# =========================
# 1. 读取 GraphML（结构视图）
# =========================

def load_graphrag_graphml(graphml_path: str) -> nx.Graph:
    """
    读取 GraphRAG 生成的 graph_storage_nx_data.graphml
    """
    graph = nx.read_graphml(graphml_path)
    return graph


# =========================
# 2. 读取 VectorStore（语义视图）
# =========================

def load_vector_store_embeddings(vector_store_path: str) -> Dict[str, np.ndarray]:
    """
    读取 default__vector_store.json

    返回：
        entity_name -> embedding (1024-d)
    """
    with open(vector_store_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    embedding_dict = data["embedding_dict"]
    metadata_dict = data["metadata_dict"]

    entity_to_embedding = {}

    for text_id, meta in metadata_dict.items():
        if "entity_name" not in meta:
            continue
        entity_name = meta["entity_name"]

        if text_id not in embedding_dict:
            continue

        entity_to_embedding[entity_name] = np.asarray(
            embedding_dict[text_id],
            dtype=np.float32
        )

    return entity_to_embedding


# =========================
# 3. 构建 node index（强绑定）
# =========================

def build_node_index(
    graph: nx.Graph,
    embeddings: Dict[str, np.ndarray]
):
    """
    GraphML node_id <-> VectorStore entity_name 的唯一绑定点
    """
    node_ids = list(graph.nodes())

    missing = [n for n in node_ids if n not in embeddings]
    if missing:
        raise ValueError(
            f"[ERROR] {len(missing)} nodes exist in GraphML but not in VectorStore. "
            f"Example: {missing[:5]}"
        )

    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    return node_ids, node_to_idx


# =========================
# 4. 构建节点特征矩阵 X
# =========================

def build_node_features(
    node_ids: List[str],
    embeddings: Dict[str, np.ndarray]
) -> torch.Tensor:
    """
    X ∈ R^{N×1024}
    """
    features = np.stack(
        [embeddings[node_id] for node_id in node_ids],
        axis=0
    )
    return torch.from_numpy(features)


# =========================
# 5. 构建边结构
# =========================

def build_edges(
    graph: nx.Graph,
    node_to_idx: Dict[str, int]
):
    """
    从 GraphML 中提取：
    - edge_index
    - edge_weight
    - relation_name
    """
    src_idx, tgt_idx = [], []
    weights, relations = [], []

    for src, tgt, data in graph.edges(data=True):
        src_idx.append(node_to_idx[src])
        tgt_idx.append(node_to_idx[tgt])

        weights.append(float(data.get("weight", 1.0)))
        relations.append(data.get("relation_name", ""))

    edge_index = torch.tensor([src_idx, tgt_idx], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)

    return edge_index, edge_weight, relations


# =========================
# 6. 总入口函数（推荐）
# =========================

def load_graphrag_entity_relation_graph(
    graphml_path: str,
    vector_store_path: str
) -> GraphForDiffusion:
    """
    返回一个可直接用于扩散模型训练的图结构
    """
    graph = load_graphrag_graphml(graphml_path)
    embeddings = load_vector_store_embeddings(vector_store_path)

    node_ids, node_to_idx = build_node_index(graph, embeddings)
    x = build_node_features(node_ids, embeddings)
    edge_index, edge_weight, edge_type = build_edges(graph, node_to_idx)

    return GraphForDiffusion(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        edge_type=edge_type,
        node_ids=node_ids,
    )


# =========================
# 7. 可视化函数（加载验证）
# =========================

def visualize_subgraph(graph: nx.Graph, num_nodes: int = 80, seed: int = 42):
    random.seed(seed)
    nodes = random.sample(list(graph.nodes()), min(num_nodes, graph.number_of_nodes()))
    subgraph = graph.subgraph(nodes)

    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(subgraph, seed=seed)

    nx.draw(
        subgraph,
        pos,
        node_size=50,
        node_color="steelblue",
        edge_color="gray",
        alpha=0.7,
        with_labels=False
    )
    plt.title(f"Random Subgraph ({len(nodes)} nodes)")
    #plt.tight_layout()
    plt.savefig("graph_diffusion/subgraph.png")


def visualize_degree_distribution(graph: nx.Graph):
    degrees = [d for _, d in graph.degree()]

    plt.figure(figsize=(6, 4))
    plt.hist(degrees, bins=50, log=True)
    plt.xlabel("Node Degree")
    plt.ylabel("Frequency (log)")
    plt.title("Node Degree Distribution")
    #plt.tight_layout()
    plt.savefig("graph_diffusion/degree.png")


def visualize_node_embeddings(
    x: torch.Tensor,
    node_ids: List[str],
    max_points: int = 2000,
    seed: int = 42
):
    np.random.seed(seed)
    X = x.cpu().numpy()
    n = X.shape[0]

    if n > max_points:
        idx = np.random.choice(n, max_points, replace=False)
        X = X[idx]

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    plt.figure(figsize=(7, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], s=5, alpha=0.6)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Entity Embedding PCA Projection")
    #plt.tight_layout()
    plt.savefig("graph_diffusion/Embedding.png")


# =========================
# 8. 示例运行（可视化验证）
# =========================

if __name__ == "__main__":
    graphml_file = "./Output/multihop-rag/kg_graph/graph_storage_nx_data.graphml"
    vector_store_file = "./Output/multihop-rag/kg_graph/entities_vdb/default__vector_store.json"

    graph_data = load_graphrag_entity_relation_graph(
        graphml_file,
        vector_store_file
    )

    print("✅ Graph loaded successfully")
    print(f"Nodes: {graph_data.x.shape[0]}")
    print(f"Edges: {graph_data.edge_index.shape[1]}")
    print(f"Embedding dim: {graph_data.x.shape[1]}")

    # ====== 可视化证据 ======
    nx_graph = load_graphrag_graphml(graphml_file)

    visualize_subgraph(nx_graph)
    visualize_degree_distribution(nx_graph)
    visualize_node_embeddings(graph_data.x, graph_data.node_ids)
