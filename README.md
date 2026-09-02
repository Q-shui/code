# Diffusion-Inspired Graph Refinement for Robust Reasoning in GraphRAG

This repository contains the paper-oriented experimental code for DGRAG ablation studies, extended from the DIGIMON / GraphRAG unified framework. On top of the original GraphRAG construction, retrieval, question answering, and evaluation pipeline, this project adds graph-diffusion-based edge pruning, denoised graph generation, structural metric analysis, random-pruning and metadata-pruning baselines, and scripts for producing paper-ready result tables.

## Overview

The project has two main workflows:

1. GraphRAG pipeline: load a dataset, build chunks, construct the entity-relation graph, build vector indexes and community information, then run QA and automatic evaluation.
2. DGRAG denoising: load a graph diffusion checkpoint on a pre-built GraphRAG graph, predict which edges should be kept or removed, and export the denoised GraphML graph together with structural metrics.

The default method configuration is `Option/Method/GGraphRAG.yaml`, and the global configuration is `Option/Config2.yaml`. The main GraphRAG entry point is `main.py`; graph diffusion training and inference scripts are under `train/graph_diffusion/`.

## Repository Structure


## Environment Setup

Create the Conda environment:

```bash
conda env create -f experiment.yml
conda activate digimon
```

Or create it with a custom name:

```bash
conda env create -f experiment.yml -n dgrag
conda activate dgrag
```

The environment includes PyTorch, Transformers, LlamaIndex, OpenAI SDK, NetworkX, pandas, scikit-learn, graspologic, and other dependencies. Graph diffusion training and inference use CUDA by default when available. If no GPU is available, replace `--device cuda` with `--device cpu` in the relevant commands.

## Configuration

The global configuration file is `Option/Config2.yaml`. The main fields are:

```yaml
llm:
  api_type: openai
  base_url: https://api.openai.com/v1
  model: gpt-4o-mini
  api_key: YOUR_API_KEY

embedding:
  api_type: openai
  base_url: https://api.openai.com/v1
  model: text-embedding-3-small
  api_key: YOUR_API_KEY
  dimensions: 1024

data_root: ./Data
working_dir: ./Output_denoised
exp_name: graphrag_gpt_4o_mini_multihop-rag-summary
```

Do not commit real API keys to the repository. Use a private local configuration or inject credentials through environment/config management before running experiments.

Method-specific configurations are stored in `Option/Method/`. Common options include:

- `GGraphRAG.yaml`: global GraphRAG with community information.
- `RAPTOR.yaml`: tree-based retrieval.
- `LightRAG.yaml`: relation vector retrieval and relation-to-chunk retrieval.
- `HippoRAG.yaml`, `KGP.yaml`, `ToG.yaml`, `Dalk.yaml`, `GR.yaml`, `LGraphRAG.yaml`: additional graph retrieval configurations.

Configuration merging is handled by `Option/Config2.py`: the method configuration, project default configuration, and root configuration are merged, and later-loaded fields overwrite earlier ones. The command-line `dataset_name` is added to the final configuration and appended to `working_dir` to form the dataset-level output directory.

## Data Format

Each dataset should be placed under `Data/<dataset_name>/` and contain at least two JSONL files:

```text
Data/<dataset_name>/
├── Corpus.json
└── Question.json
```

Each row in `Corpus.json` represents one document. The code reads:

- `title`: document title.
- `context`: document content.

Each row in `Question.json` represents one QA item. The code reads:

- `question`: question text.
- `answer`: ground-truth answer.
- Any additional fields are preserved in the output result file.

Datasets visible in the current repository include `quality`, `multihop-rag`, `multihop-rag-summary`, `mix`.

## Run the GraphRAG Pipeline

Run from the repository root:

```bash
python main.py -opt Option/Method/GGraphRAG.yaml -dataset_name quality
```

You can replace both the method and dataset:

```bash
python main.py -opt Option/Method/RAPTOR.yaml -dataset_name multihop-rag
python main.py -opt Option/Method/LightRAG.yaml -dataset_name mix
```

The main pipeline performs the following steps:

1. Load the corpus from `Data/<dataset_name>/Corpus.json`.
2. Build chunks, graph structures, and vector indexes.
3. Query each question in `Question.json`.
4. Write generated answers to `Results/results.json`.
5. Run evaluation and write metrics to `Results/metrics.json`.

The output directory is determined by `working_dir` and `exp_name` in `Option/Config2.yaml`. For example:

```text
Output_denoised/quality/graphrag_gpt_4o_mini_quality/
├── Configs/
├── Results/
└── Metrics/
```

`main.py` currently evaluates at most the first 100 questions by default. To evaluate the full dataset, modify `dataset_len` in `wrapper_query()`.

## Train the Graph Diffusion Denoiser

The graph diffusion training entry point is:

```bash
python train/graph_diffusion/train_diffusion.py \
  --graph_path Output/quality/kg_graph/graph_storage_nx_data.graphml \
  --vector_store_path Output/quality/kg_graph/entities_vdb/default__vector_store.json \
  --checkpoint_dir train/graph_diffusion/checkpoints_quality \
  --checkpoint_name graph_diffusion_prune_v2.pt \
  --epochs 20 \
  --steps_per_epoch 100 \
  --device cuda
```

Important arguments:

- `--graph_path`: GraphML graph produced by the original GraphRAG pipeline.
- `--vector_store_path`: entity vector index JSON.
- `--checkpoint_dir`: output directory for model checkpoints.
- `--max_nodes_per_batch`: maximum number of nodes for sampled subgraphs.
- `--sampling_strategy`: supports `edge` and `node`.
- `--num_steps`: number of diffusion timesteps.
- `--label_threshold`: threshold used to construct edge labels from confidence features.
- `--negative_ratio`: negative edge sampling ratio.

After training, the script writes:

```text
train/graph_diffusion/checkpoints_quality/
├── graph_diffusion_prune_v2.pt
└── graph_diffusion_prune_v2.history.json
```

## Single Denoising Inference Run

Use a trained checkpoint to prune edges in one GraphRAG graph:

```bash
python train/graph_diffusion/denoise_graph.py \
  --checkpoint_path train/graph_diffusion/checkpoints_quality/graph_diffusion_prune_v2.pt \
  --graph_path Output/quality/kg_graph/graph_storage_nx_data.graphml \
  --vector_store_path Output/quality/kg_graph/entities_vdb/default__vector_store.json \
  --output_mask_path ablation_denoise/quality/t085_r1_s42/mask.json \
  --output_graph_path ablation_denoise/quality/t085_r1_s42/graph_denoised.graphml \
  --threshold 0.85 \
  --refinement_steps 1 \
  --edge_batch_size 8192 \
  --seed 42 \
  --device cuda
```

Generated files:

- `mask.json`: kept and removed edges, including model scores and edge metadata.
- `graph_denoised.graphml`: denoised graph.
- `graph_denoised.metrics.json`: structural changes between the original and denoised graphs.

This project is extended from the following work:

```bibtex
@article{zhou2025depth,
  title={In-depth Analysis of Graph-based RAG in a Unified Framework},
  author={Zhou, Yingli and Su, Yaodong and Sun, Youran and Wang, Shu and Wang, Taotao and He, Runyuan and Zhang, Yongwei and Liang, Sicong and Liu, Xilin and Ma, Yuchi and others},
  journal={arXiv preprint arXiv:2503.04338},
  year={2025}
}
```
