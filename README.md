# Kaggle Tabular & NLP Modeling Utilities

**Author:** Luis Chandrasekhar Ballesteros López-Tello ([@ChandraBalote](https://github.com/ChandraBalote) · [Kaggle @chandrasaekhar](https://www.kaggle.com/chandrasaekhar))

Modular Python utilities for cross-validation, leak-free feature encoding, and ensembling in structured tabular and text classification problems:

1. **Cross-Validation & Out-of-Fold Tracking**
   - Consistent `StratifiedKFold` and `GroupKFold` fold generators with out-of-fold `.npy` caching so every model family shares identical validation splits.

2. **Adversarial Distribution Checks**
   - Quick `train` vs. `test` classifier and Kolmogorov–Smirnov diagnostics to verify whether covariate shift is present before choosing a validation scheme.

3. **Nested Out-of-Fold Target & Frequency Encoding**
   - Fast vectorized `numpy.bincount` nested target encoding (`inner` stratified split inside each outer training fold) with configurable empirical Bayes smoothing (`smooth`) and multi-scale numeric discretization keys.

4. **Multi-Window Text Pooling**
   - Word/character TF-IDF pipelines paired with Head, Tail, and Contrast-Pivot encoder pooling for long-form document classification.

5. **Log-Odds & Rank Ensembling**
   - Nested cross-validated logistic stacking, rank-logit blending, and step-regularized hill-climbing for ROC-AUC and log-loss objectives.
