# ⚡ Kaggle Grandmaster Competitive ML & NLP Toolkit

**Author:** Luis Chandrasekhar Ballesteros López-Tello ([@ChandraBalote](https://github.com/ChandraBalote) | [Kaggle @chandrasaekhar](https://www.kaggle.com/chandrasaekhar))

A battle-tested, modular Python framework implementing top Kaggle Grandmaster practices for **Tabular, NLP, and Hybrid Multimodal Competitions**:

1. **Leak-Free Cross-Validation & Out-Of-Fold (OOF) Stacking Engine** (`StratifiedKFold`, `GroupKFold`, `MultilabelStratifiedKFold`).
2. **Adversarial Validation (`Train vs. Test` Shift Detector)**: Automatically identifies covariate shift and computes sample weights for training instances closest to the test distribution.
3. **High-Order Feature Engineering for Tabular & Fraud Detection**:
   - Out-of-fold Smoothed Target Encoding, Frequency Encoding, Group Aggregations (`mean`, `std`, `min`, `max`, `z-score`), and Non-Linear Interaction Ratios.
4. **3-View Discourse-Aware NLP Feature Extractor**:
   - Multi-granular TF-IDF (`word` + `char_wb`) + Pretrained Transformer (`DeBERTa-v3` / `RoBERTa`) Head/Tail/Pivot pooling with INT8 dynamic quantization.
5. **Nested Log-Odds & Rank-Averaged Meta-Ensembling**:
   - Combines GBDT families (`LightGBM`, `CatBoost`, `XGBoost`), Linear/Kernel models, and Neural Networks in log-odds space with Nelder-Mead / Dirichlet simplex optimization and leakage-free threshold tuning.
