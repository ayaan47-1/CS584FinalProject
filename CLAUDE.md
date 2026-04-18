# CS584 Project: PatchTST Implementation

**Student:** Ayaan Khan (A20505209)
**Course:** CS584 - Machine Learning, Spring 2026
**Due:** 2026-04-20

## Project Overview

Implementing PatchTST (Nie et al., ICLR 2023) from scratch in PyTorch — a patch-based Transformer for long-term time series forecasting. Two key innovations: **patching** (subseries-level tokens, reducing attention complexity from O(L²) to O((L/S)²)) and **channel independence** (each variate processed independently through a shared backbone).

## Goals

1. Implement PatchTST architecture from scratch in PyTorch (Jupyter notebook)
2. Reproduce main paper results on ETTh1, ETTh2, and Weather datasets
3. Compare against DLinear baseline
4. Ablation studies on patch length, stride, and look-back window size
5. 8+ page report + 10-15 minute presentation

## Architecture Components

- **Patch embedding layer** — segments input into subseries patches
- **Learnable positional encodings**
- **Transformer encoder** — multi-head self-attention + feed-forward
- **Flattening + linear prediction head**

## Datasets

| Dataset | Freq | Features | Steps | Split |
|---------|------|----------|-------|-------|
| ETTh1 | hourly | 7 | 17,420 | 12/4/4 months |
| ETTh2 | hourly | 7 | 17,420 | 12/4/4 months |
| Weather | 10-min | 21 | 52,696 | 70/10/20% |

Source: https://github.com/zhouhaoyi/ETDataset

**Prediction horizons:** {96, 192, 336, 720} steps
**Metrics:** MSE and MAE

## Development Notes

- Solo project — no team members
- Primary deliverable is a Jupyter notebook
- Per-feature standard scaling fit on training set only (no data leakage)
- Sliding window data pipeline for train/val/test

## Commit Style

```
<type>: <description>
```
Types: feat, fix, refactor, docs, test, chore

Do NOT add Claude co-author attribution to commits.
