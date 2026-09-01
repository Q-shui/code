'''Training routine for the GraphRAG graph diffusion denoising agent.'''

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from graph_diffusion.graph_diffusion import (
    GraphDiffusion,
    NoiseSchedule,
    forward_diffuse,
    build_dense_adj,
    adjacency_to_edge_index,
)
from graph_diffusion.graph_loader import load_graphrag_entity_relation_graph

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train the GraphRAG diffusion denoiser.')
    parser.add_argument('--graphml-path', default='./Output/multihop-rag/kg_graph/graph_storage_nx_data.graphml')
    parser.add_argument('--vector-store-path', default='./Output/multihop-rag/kg_graph/entities_vdb/default__vector_store.json')
    parser.add_argument('--checkpoint-path', default='./graph_diffusion/checkpoints/diffusion.pt')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--steps-per-epoch', type=int, default=50)
    parser.add_argument('--diffusion-steps', type=int, default=100)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--add-start', type=float, default=0.01)
    parser.add_argument('--add-end', type=float, default=0.35)
    parser.add_argument('--remove-start', type=float, default=0.0)
    parser.add_argument('--remove-end', type=float, default=0.60)
    return parser.parse_args()

def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    graph_data = load_graphrag_entity_relation_graph(args.graphml_path, args.vector_store_path)
    x = graph_data.x.to(device)
    num_nodes = x.size(0)
    adj_clean = build_dense_adj(num_nodes, graph_data.edge_index).to(device)

    schedule = NoiseSchedule(
        T=args.diffusion_steps,
        add_start=args.add_start,
        add_end=args.add_end,
        remove_start=args.remove_start,
        remove_end=args.remove_end,
    )

    model = GraphDiffusion(x.size(1), args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs * args.steps_per_epoch))

    Path(args.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for step in range(1, args.steps_per_epoch + 1):
            t = torch.randint(0, schedule.T + 1, (1,), device=device)
            noisy_adj = forward_diffuse(adj_clean, int(t.item()), schedule)
            edge_index = adjacency_to_edge_index(noisy_adj)
            if edge_index.size(1) == 0:
                continue

            logits = model(x, edge_index, t)
            loss = F.binary_cross_entropy_with_logits(logits, adj_clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
        scheduler.step()

        if losses:
            mean_loss = sum(losses) / len(losses)
            if mean_loss < best_loss:
                best_loss = mean_loss
            print(
                f'Epoch {epoch:02d}/{args.epochs} | '
                f'loss {mean_loss:.4f} | '
                f'best {best_loss:.4f} | '
                f't avg {schedule.T / 2:.2f}'
            )
        else:
            print(f'Epoch {epoch:02d}: skipped (no edges were sampled).')

    torch.save(
        {
            'model_state': model.state_dict(),
            'args': vars(args),
            'node_order': graph_data.node_ids,
        },
        args.checkpoint_path,
    )
    print(f'Saved diffusion checkpoint to {args.checkpoint_path}')


if __name__ == '__main__':
    train(parse_args())
