# DASE7506 MP1 — Small Language Model Challenge

Train a small GPT from scratch on WikiText-2, improve its architecture or
training, and minimize reproducible test bits per byte (BPB) within the
course resource limits.

## Final model

| Property | Value |
|---|---|
| Architecture | GELU FFN + RoPE + LayerNorm + dropout 0.25 |
| Parameters | 10,509,440 |
| Training tokens | 163,840,000 (20,000 steps × 32 × 256) |
| Validation BPB (best @ 19500) | 1.4865 |
| **Test BPB (FP32, CPU)** | **1.5074** |
| Checkpoint | `runs/final-model/checkpoint_step19500.pt` |

Baseline (provided): 1,088,256 parameters, **2.1013** test BPB.

## Ablations (all at the final model's scale)

| Experiment | FFN | Pos | dropout | Val BPB | Test BPB |
|---|---|---|---|---|---|
| **final-model** | **GELU** | **RoPE** | **0.25** | **1.4865** | **1.5074** |
| ablation-no-gelu | SwiGLU | RoPE | 0.25 | 1.5044 | 1.5317 |
| ablation-no-rope | GELU | absolute | 0.25 | 1.5203 | 1.5453 |
| ablation-no-dropout | GELU | RoPE | 0 | 1.6160 | 1.6392 |

Each ablation removes exactly one component from the final model. Full
logs, metrics and test scores are in `evidence/`.

## Resource limits (frozen final predictor)

| Limit | Bound | Measured |
|---|---|---|
| CPU scoring time | ≤ 5× baseline (87.9 s) | 34.71 s |
| Peak evaluation RAM | ≤ 4 GiB | ≈ 2.23 GiB |
| Inference assets | ≤ 64 MiB | 41 MB |

## Reproduction

All commands run from the repository root.

### Install

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\Activate.ps1
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

### Train the final model

```bash
python train.py \
    --implementation student_no_swiglu \
    --config configs/large_gelu.json \
    --device cuda \
    --seed 17 \
    --steps 20000 \
    --batch-size 32 \
    --eval-every 500 \
    --run-dir runs/final-model
```

(`--device cpu` works if no GPU is available; training is much slower.)

### Evaluate the frozen checkpoint (ranked, CPU + FP32)

```bash
python evaluate.py \
    --checkpoint runs/final-model/checkpoint_step19500.pt \
    --device cpu \
    --precision fp32 \
    --split test
```

The reported score is the `bpb` field of the output JSON.

### Reproduce an ablation

```bash
# no RoPE
python train.py --implementation student_no_rope --config configs/large_gelu.json \
    --device cuda --seed 17 --steps 20000 --batch-size 32 --eval-every 500 \
    --run-dir runs/ablation-no-rope-gelu

# no dropout
python train.py --implementation student --config configs/large_gelu_no_dropout.json \
    --device cuda --seed 17 --steps 20000 --batch-size 32 --eval-every 500 \
    --run-dir runs/ablation-no-dropout-gelu
```

## Files

| File | Role |
|---|---|
| `student.py` | Final model factory (GELU + RoPE + dropout) |
| `student_no_rope.py` | Ablation: no RoPE |
| `student_no_swiglu.py` | Same as `student.py` (kept for naming symmetry with the old SwiGLU variant) |
| `rope.py` | Rotary position embedding, no parameters |
| `train.py` | Training loop (modified: weight_decay 0.35, saves intermediate checkpoints) |
| `evaluate.py`, `common.py` | Provided scorer and helpers — unchanged |
| `configs/large_gelu.json` | Final model config |
| `configs/large_gelu_no_dropout.json` | No-dropout ablation config |
| `configs/baseline.json` | Provided baseline config |
| `evidence/` | Per-experiment metrics, test JSON, logs and commands |
| `runs/` | Final and ablation checkpoints used for the reported scores |

## AI assistance disclosure

Substantive AI assistance (ChatGPT / Claude) was used for:
- debugging environment and training issues;
- suggesting architectural directions (RoPE, GELU vs SwiGLU, dropout schedules) that were then implemented, ablated and verified by the author;
- writing helper scripts for running repeated ablations and parsing logs.

All model code, training decisions, ablations and reported numbers were
reviewed and validated by the author. No external training text, no
pretrained weights, and no test-based tuning were used.

## Data attribution

WikiText-2 was introduced by Merity et al., *Pointer Sentinel Mixture Models*
(https://arxiv.org/abs/1609.07843). The upstream dataset is distributed under
CC BY-SA 3.0 and the GNU Free Documentation License. The `wikitext-2-raw-v1`
splits supplied with the course preserve revision
`b08601e04326c79dfdd32d625aee71d232d685c3`; the BPE-2048 tokenizer is fitted
only on the training split. Dataset hashes are in `data/manifest.json`.
