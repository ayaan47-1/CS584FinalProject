# PatchTST: Long-Term Time Series Forecasting with Patch-Based Transformers

A from-scratch PyTorch implementation of **PatchTST** (Nie et al., ICLR 2023) for long-term time series forecasting on electricity transformer datasets.

## Key Innovation

PatchTST treats time series as images: subseries-level *patches* are tokens (like image patches in ViT), reducing self-attention complexity from O(L²) to O((L/S)²) while capturing local temporal patterns.

## Results

**PatchTST + RevIN vs. DLinear**

| Dataset | h=192 | h=336 | h=720 |
|---------|-------|-------|-------|
| ETTh1   | ❌ loses | ❌ loses | ❌ loses |
| ETTh2   | ✅ **+1.7%** | ✅ **+6.8%** | ✅ **+23%** |

RevIN (Reversible Instance Normalization) is critical on datasets with distributional shift (ETTh2); without it, DLinear wins everywhere.

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook src/patchtst.ipynb
```

Results are cached to `results.json` after each experiment run — subsequent runs skip re-training.

## Contents

- `src/patchtst.ipynb` — Full implementation + experiments
- `doc/report.pdf` — 10-page research report
- `presentation/` — Slides + 4-up handout PDF
- `sources/` — Reference papers (PatchTST, DLinear, RevIN)
- `data/` — ETTh1, ETTh2 datasets
- `results.json` — Cached experiment results

## Architecture

- **Patch embedding** — sliding window with learnable positional encoding
- **Transformer encoder** — 3 layers, 8 heads, 128 dims, pre-norm
- **Channel independence** — each variable processed separately
- **RevIN wrapper** — per-instance normalization for distributional shift
- **Custom MHSA** — MPS-compatible Multi-Head Self-Attention

## Key Findings

1. **RevIN is dataset-dependent** — transformative on ETTh2 (shift), marginal on ETTh1 (stationary)
2. **Without RevIN, Transformer capacity hurts** — DLinear outperforms on all 8 combinations
3. **Larger stride wins** — stride=16 (non-overlapping patches) outperforms stride=4 and stride=8
4. **Look-back window saturates** — seq_len=336 (paper default) is optimal; seq_len=512 overfits

## References

- Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers," ICLR 2023
- Zeng et al., "Are Transformers Effective for Time Series Forecasting?" AAAI 2023
- Kim et al., "Reversible Instance Normalization for Accurate Time-Series Forecasting," ICLR 2022

**Author:** Ayaan Khan (A20505209) — CS584 Machine Learning, Spring 2026
