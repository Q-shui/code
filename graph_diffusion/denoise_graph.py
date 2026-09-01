'''Graph diffusion denoising inference script for GraphRAG.'''

from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import torch

from graph_diffusion.graph_diffusion import (
    GraphDiffusion,
    NoiseSchedule,
    adjacency_to_edge_index,
    random_noise_graph,
    enforce_symmetry,
)
from graph_diffusion.graph_loader import (
    load_graphrag_entity_relation_graph,
    load_graphrag_graphml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Denoise a GraphRAG graph with the trained diffusion model.')
    parser.add_argument('--graphml-path', default='./Output/multihop-rag/kg_graph/graph_storage_nx_data.graphml')
    parser.add_argument('--vector-store-path', default='./Output/multihop-rag/kg_graph/entities_vdb/default__vector_store.json')
    parser.add_argument('--checkpoint-path', default='./graph_diffusion/checkpoints/diffusion.pt')
    parser.add_argument('--output-graphml', default='./graph_diffusion/denoised_graph.graphml')
    parser.add_argument('--output-npy', default=None)
    parser.add_argument('--denoise-steps', type=int, default=None)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--start-density', type=float, default=0.1)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--add-start', type=float, default=None)
    parser.add_argument('--add-end', type=float, default=None)
    parser.add_argument('--remove-start', type=float, default=None)
    parser.add_argument('--remove-end', type=float, default=None)
    parser.add_argument('--batch', type=int, default=1)
    return parser.parse_args()


def load_diffusion_model(checkpoint_path: str, in_feats: int, device: torch.device) -> tuple[GraphDiffusion, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    meta = checkpoint.get('args', {})
    hidden_dim = meta.get('hidden_dim', 512)
    model = GraphDiffusion(in_feats, hidden_dim).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model, meta


def build_schedule(args: argparse.Namespace, meta: dict) -> NoiseSchedule:
    total_steps = args.denoise_steps if args.denoise_steps is not None else meta.get('diffusion_steps', 100)
    return NoiseSchedule(
        T=total_steps,
        add_start=args.add_start if args.add_start is not None else meta.get('add_start', 0.01),
        add_end=args.add_end if args.add_end is not None else meta.get('add_end', 0.35),
        remove_start=args.remove_start if args.remove_start is not None else meta.get('remove_start', 0.0),
        remove_end=args.remove_end if args.remove_end is not None else meta.get('remove_end', 0.6),
    )


def iterative_denoise(
    model: GraphDiffusion,
    schedule: NoiseSchedule,
    x: torch.Tensor,
    start_density: float,
    threshold: float,
    device: torch.device,
) -> torch.Tensor:
    num_nodes = x.size(0)
    current = random_noise_graph(num_nodes, density=start_density, device=device)
    with torch.no_grad():
        for t in reversed(range(schedule.T + 1)):
            edge_index = adjacency_to_edge_index(current)
            if edge_index.size(1) == 0:
                edge_index = adjacency_to_edge_index(random_noise_graph(num_nodes, density=0.05, device=device))
            logits = model(x, edge_index, torch.tensor([t], device=device))
            current = torch.sigmoid(logits)
            current = enforce_symmetry(current)
            current.fill_diagonal_(0)
    return (current > threshold).float()


def write_graphml(original_path: str, node_ids: list[str], adj: torch.Tensor, output_path: str) -> None:
    base = load_graphrag_graphml(original_path)
    denoised = nx.Graph()
    denoised.add_nodes_from(base.nodes(data=True))
    for i in range(adj.size(0)):
        for j in range(i + 1, adj.size(1)):
            weight = float(adj[i, j])
            if weight > 0:
                denoised.add_edge(node_ids[i], node_ids[j], weight=weight)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(denoised, output_path)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    graph_data = load_graphrag_entity_relation_graph(args.graphml_path, args.vector_store_path)
    x = graph_data.x.to(device)
    model, meta = load_diffusion_model(args.checkpoint_path, x.size(1), device)
    schedule = build_schedule(args, meta)
    adj_hat = iterative_denoise(model, schedule, x, args.start_density, args.threshold, device)
    write_graphml(args.graphml_path, graph_data.node_ids, adj_hat, args.output_graphml)
    if args.output_npy:
        torch.save(adj_hat, args.output_npy)
    print(f'Denoised graph written to {args.output_graphml}')


if __name__ == '__main__':
    main()
