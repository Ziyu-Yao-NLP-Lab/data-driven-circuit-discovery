# Data-driven Circuit Discovery (DCD)

Paper: https://arxiv.org/pdf/2605.09129v1
Authors: Daking Rai, Mor Geva, and Ziyu Yao
Thread: https://x.com/DakingRai/status/2055317587882283188

DCD is a method for discovering **multiple** circuits inside a language model by
clustering per-example attribution vectors. We show that existing
hypothesis-driven circuit-discovery methods (EAP, EAP-IG, edge activation
patching) tend to find **dataset-specific** rather than **task-general**
circuits; DCD makes that structure explicit by recovering one circuit per
cluster of examples.

This repository contains the code to reproduce the three experiment families
in the paper:

1. `experiments/general-task-circuits/` — **Section 4:** hypothesis-driven
   methods do not discover general task circuits (minor variation in the
   discovery dataset produces low-overlap circuits with low cross-task
   faithfulness).
2. `experiments/dataset-specific-circuits/` — **Section 5:** hypothesis-driven
   methods can mix distinct mechanisms in a single circuit (a single circuit
   discovered on a multi-task dataset achieves high faithfulness on the
   discovery distribution but conflates the underlying mechanisms).
3. `experiments/dcd/` — **the proposed DCD method**: per-example EAP-IG
   attribution → clustering → per-cluster circuits → faithfulness evaluation.

## Installation

```bash
git clone --recurse-submodules <this-repo-url>
cd data-driven-circuit-discovery

python -m venv .venv && source .venv/bin/activate     # or conda
pip install -r requirements.txt

# Install the bundled EAP / MIB-circuit-track dependency
pip install -e third_party/MIB-circuit-track/EAP-IG
pip install -e third_party/MIB-circuit-track
```

If you cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

## Repository layout

```
src/
├── dcd/                  # core DCD library (clustering, sparsification, representatives)
├── circuit_discovery/    # faithfulness evaluation + per-example attribution
├── data/                 # per-task data generators (IOI, sequence-completion, arithmetic, entity-binding)
└── utils/                # model loading, graph utils, metrics, EAP patches

experiments/
├── dcd/                          # 01 → 05 DCD pipeline + curated analysis/ scripts
├── dataset-specific-circuits/    # Section 5: single circuits mix distinct mechanisms
└── general-task-circuits/        # Section 4: methods don't find a general task circuit

configs/{family}/{task}/{model}.yaml   # one config per experiment-task-model combination
third_party/MIB-circuit-track/         # git submodule (EAP + MIB utilities)
```

All scripts assume they are launched from the repo root. Paths inside
configs (`cache_dir: models`, etc.) are resolved relative to the launch
directory.

## Models

The pipeline is model-agnostic; we report results for:

| Short name                | HuggingFace ID                          |
|---------------------------|-----------------------------------------|
| `gpt2`                    | `gpt2`                                  |
| `qwen2.5-7b-instruct`     | `Qwen/Qwen2.5-7B-Instruct`              |
| `llama-3.1-8b-instruct`   | `meta-llama/Llama-3.1-8B-Instruct`      |

Weights are downloaded by `transformer_lens` / `transformers` on first use
into the directory given by `model.cache_dir` in each config (default:
`./models`). The `models/` directory is gitignored.

## Reproducing experiments

### DCD (proposed method)

```bash
python experiments/dcd/01_create_data.py     --config configs/dcd/ioi/gpt2.yaml
python experiments/dcd/02_run_attribution.py --config configs/dcd/ioi/gpt2.yaml
python experiments/dcd/03_run_clustering.py  --config configs/dcd/ioi/gpt2.yaml \
                                             --clustering-config configs/dcd/clustering_grid.yaml --stage preprocess
python experiments/dcd/03_run_clustering.py  --config configs/dcd/ioi/gpt2.yaml \
                                             --clustering-config configs/dcd/clustering_grid.yaml --stage cluster
python experiments/dcd/03_run_clustering.py  --config configs/dcd/ioi/gpt2.yaml \
                                             --clustering-config configs/dcd/clustering_grid.yaml --stage select
python experiments/dcd/04_create_circuits.py    --config configs/dcd/ioi/gpt2.yaml
python experiments/dcd/05_evaluate_faithfulness.py --config configs/dcd/ioi/gpt2.yaml
```

Swap the config to reproduce other (task, model) combinations:
`configs/dcd/{arithmetic,entity-binding,ioi,sequence-completion,all-tasks}/<model>.yaml`.

### Section 4 — Hypothesis-driven methods do not discover general task circuits

```bash
python experiments/general-task-circuits/01_create_data.py           --config configs/general-task-circuits/ioi/gpt2.yaml
python experiments/general-task-circuits/02_run_circuit_discovery.py --config configs/general-task-circuits/ioi/gpt2.yaml
python experiments/general-task-circuits/03_evaluate.py              --config configs/general-task-circuits/ioi/gpt2.yaml
```

### Section 5 — Hypothesis-driven methods can mix distinct mechanisms in a single circuit

```bash
python experiments/dataset-specific-circuits/01_create_data.py           --config configs/dataset-specific-circuits/gpt2.yaml
python experiments/dataset-specific-circuits/02_run_circuit_discovery.py --config configs/dataset-specific-circuits/gpt2.yaml
python experiments/dataset-specific-circuits/03_evaluate.py              --config configs/dataset-specific-circuits/gpt2.yaml
```

### Paper figures

After completing the DCD pipeline, the curated scripts under
`experiments/dcd/analysis/` reproduce cluster-purity and faithfulness-curve
figures from the paper. Each script reads from `results/` and writes
plots back next to the source data.

## Citation

```bibtex
@article{rai2026data,
  title={Data-driven Circuit Discovery for Interpretability of Language Models},
  author={Rai, Daking and Geva, Mor and Yao, Ziyu},
  journal={arXiv preprint arXiv:2605.09129},
  year={2026}
}
```
