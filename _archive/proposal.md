# PatchTST: Long-term Time Series Forecasting with Patch-based Transformers

**Name:** Ayaan Khan
**Student ID:** A20505209
**Course:** CS584 - Machine Learning, Spring 2026
**Date:** March 30, 2026

---

## Problem Statement

Long-term time series forecasting (LTSF) is a fundamental problem in machine learning with applications in energy planning, weather prediction, traffic management, and healthcare. Traditional Transformer-based approaches for LTSF suffer from two critical issues: (1) the quadratic complexity of self-attention limits the length of the look-back window, and (2) point-wise attention over individual time steps fails to capture local semantic patterns in temporal data. Recent work has even questioned whether Transformers are effective for time series at all, with simple linear models outperforming complex architectures. This project addresses whether a well-designed Transformer architecture can overcome these limitations and achieve state-of-the-art forecasting performance.

## Approach

This project implements PatchTST, a Transformer-based architecture for long-term time series forecasting proposed by Nie et al. (ICLR 2023). PatchTST introduces two key innovations:

1. **Patching:** The input time series is segmented into subseries-level patches (analogous to image patches in Vision Transformers). Each patch serves as a token for the Transformer encoder. This reduces the input sequence length from L to approximately L/S (where S is the patch stride), lowering the self-attention complexity from O(L^2) to O((L/S)^2) while preserving local temporal information within each patch. This also enables significantly longer look-back windows.

2. **Channel Independence:** Each variate (channel) of a multivariate time series is processed independently through a shared Transformer backbone, rather than mixing all variates into a single token. This reduces overfitting, improves generalization, and simplifies the model architecture.

The implementation will build the model from scratch in PyTorch within a Jupyter notebook, following the supervised learning variant described in the paper. The model consists of: a patch embedding layer, learnable positional encodings, a standard Transformer encoder (multi-head self-attention + feed-forward layers), and a flattening + linear prediction head. The project will reproduce the paper's main results on benchmark datasets and perform ablation studies on key hyperparameters (patch length, stride, and look-back window size).

## Data

The project will use the following publicly available benchmark datasets from the ETT (Electricity Transformer Temperature) collection:

- **ETTh1 / ETTh2**: Hourly recordings from two electricity transformers, each containing 7 features (6 power load features + oil temperature) over ~2 years (17,420 time steps). Train/val/test split: 12/4/4 months.
- **Weather** (secondary benchmark): 21 meteorological indicators recorded every 10 minutes for the year 2020 (52,696 time steps). Train/val/test split: 70/10/20%.

These datasets are standard benchmarks in the LTSF literature, enabling direct comparison with published results. All datasets are freely available from the ETDataset repository (https://github.com/zhouhaoyi/ETDataset). Forecasting will be evaluated at prediction horizons of {96, 192, 336, 720} time steps.

**Evaluation Metrics:** Mean Squared Error (MSE) and Mean Absolute Error (MAE), consistent with the original paper and prior LTSF benchmarks.

## Team Member Responsibilities

This is a solo project. All components will be completed by a single team member:

- **Data preprocessing and exploration:** Download datasets, implement sliding window data pipeline, normalization (per-feature standard scaling on training set), and PyTorch DataLoader construction.
- **Model implementation:** Build PatchTST architecture from scratch in PyTorch — patching module, patch embedding, Transformer encoder, and linear prediction head.
- **Training and evaluation:** Train the model on ETTh1/ETTh2 and Weather datasets across all four prediction horizons. Compute MSE and MAE. Compare against a linear baseline (DLinear).
- **Ablation studies:** Analyze the effect of patch length, stride, and look-back window size on forecasting performance.
- **Report and presentation:** Write the 8+ page report and prepare the 10-15 minute presentation with architecture diagrams and result tables/plots.

## References

1. Y. Nie, N. H. Nguyen, P. Sinthong, and J. Kalagnanam, "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers," in *Proc. International Conference on Learning Representations (ICLR)*, 2023. Available: https://arxiv.org/abs/2211.14730

2. A. Zeng, M. Chen, L. Zhang, and Q. Xu, "Are Transformers Effective for Time Series Forecasting?" in *Proc. AAAI Conference on Artificial Intelligence*, 2023. Available: https://arxiv.org/abs/2205.13504

3. H. Zhou, S. Zhang, J. Peng, S. Zhang, J. Li, H. Xiong, and W. Zhang, "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting," in *Proc. AAAI*, 2021. Available: https://arxiv.org/abs/2012.07436

4. T. Zhou, Z. Ma, Q. Wen, X. Wang, L. Sun, and R. Jin, "FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting," in *Proc. ICML*, 2022. Available: https://arxiv.org/abs/2201.12740

5. A. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," in *Proc. ICLR*, 2021. Available: https://arxiv.org/abs/2010.11929
