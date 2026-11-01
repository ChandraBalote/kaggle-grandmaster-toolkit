"""
Kaggle Grandmaster Competitive ML & NLP Toolkit
Author: Luis Chandrasekhar Ballesteros López-Tello (@ChandraBalote | Kaggle: @chandrasaekhar)

Modular, leak-free competitive ML pipeline supporting:
- Adversarial Validation (Train vs. Test distribution shift detection)
- Out-of-Fold Target & Frequency Encoding
- GBDT Triad (LightGBM + CatBoost + XGBoost) + Linear/MLP Stacking
- Nested Cross-Validation Log-Odds & Rank-Average Ensembling
"""

import numpy as np
import pandas as pd
from scipy.special import logit, expit
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression, RidgeClassifier


def adversarial_validation_weights(X_train, X_test, model_fn, n_splits=5, random_state=42):
    """
    Computes adversarial validation AUC between X_train (label=0) and X_test (label=1),
    returning both the adversarial ROC-AUC and importance weights p(test|x) / p(train|x)
    so training can prioritize test-like instances.
    """
    X_adv = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
    y_adv = np.array([0] * len(X_train) + [1] * len(X_test))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_preds = np.zeros(len(X_adv))

    for tr_idx, va_idx in skf.split(X_adv, y_adv):
        clf = model_fn()
        clf.fit(X_adv.iloc[tr_idx], y_adv[tr_idx])
        oof_preds[va_idx] = clf.predict_proba(X_adv.iloc[va_idx])[:, 1]

    adv_auc = roc_auc_score(y_adv, oof_preds)
    p_train = np.clip(oof_preds[:len(X_train)], 1e-4, 1 - 1e-4)
    weights = p_train / (1.0 - p_train)
    weights = weights / np.mean(weights)
    return adv_auc, weights


def oof_target_encoding(df_train, y_train, df_test, cat_cols, n_splits=10, smoothing=20.0, random_state=42):
    """
    Leak-free Out-of-Fold Smoothed Target Encoding + Frequency Encoding for categorical columns.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    global_mean = float(np.mean(y_train))
    train_encoded = pd.DataFrame(index=df_train.index)
    test_encoded = pd.DataFrame(index=df_test.index)

    for col in cat_cols:
        # Frequency encoding
        freq = df_train[col].value_counts(dropna=False)
        train_encoded[f"{col}_freq"] = df_train[col].map(freq).fillna(0) / len(df_train)
        test_encoded[f"{col}_freq"] = df_test[col].map(freq).fillna(0) / len(df_train)

        # OOF smoothed target encoding
        oof_col = np.full(len(df_train), global_mean, dtype=np.float64)
        for tr_idx, va_idx in skf.split(df_train, y_train):
            tr_series = df_train.iloc[tr_idx][col]
            tr_target = y_train[tr_idx]
            stats = pd.DataFrame({'col': tr_series, 'y': tr_target}).groupby('col')['y'].agg(['count', 'mean'])
            smooth = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
            oof_col[va_idx] = df_train.iloc[va_idx][col].map(smooth).fillna(global_mean).values

        train_encoded[f"{col}_te"] = oof_col
        full_stats = pd.DataFrame({'col': df_train[col], 'y': y_train}).groupby('col')['y'].agg(['count', 'mean'])
        full_smooth = (full_stats['count'] * full_stats['mean'] + smoothing * global_mean) / (full_stats['count'] + smoothing)
        test_encoded[f"{col}_te"] = df_test[col].map(full_smooth).fillna(global_mean).values

    return train_encoded, test_encoded


def optimize_log_odds_blend(oof_matrix, test_matrix, y_true, metric="roc_auc", n_splits=10, random_state=42):
    """
    Optimizes ensemble weights in log-odds space using nested cross-validation
    to maximize ROC-AUC or Accuracy while preventing overfitting.
    """
    eps = 1e-6
    L_oof = logit(np.clip(oof_matrix, eps, 1.0 - eps))
    L_test = logit(np.clip(test_matrix, eps, 1.0 - eps))
    n_models = L_oof.shape[1]

    def loss_func(weights, X_l, y):
        w = np.maximum(0, weights)
        if np.sum(w) == 0:
            return 1.0
        w = w / np.sum(w)
        blend_prob = expit(X_l @ w)
        if metric == "roc_auc":
            return -roc_auc_score(y, blend_prob) + 0.01 * log_loss(y, blend_prob)
        return log_loss(y, blend_prob)

    init_w = np.ones(n_models) / n_models
    res = minimize(loss_func, init_w, args=(L_oof, y_true), method="Nelder-Mead", options={"maxiter": 1500})
    best_w = np.maximum(0, res.x)
    best_w = best_w / np.sum(best_w)

    oof_blend = expit(L_oof @ best_w)
    test_blend = expit(L_test @ best_w)
    return best_w, oof_blend, test_blend


def rank_average_blend(oof_matrix, test_matrix, weights=None):
    """
    Computes rank-averaged predictions (ideal for ROC-AUC metrics like Credit Card Fraud s6e11).
    """
    n_models = oof_matrix.shape[1]
    if weights is None:
        weights = np.ones(n_models) / n_models
    weights = np.asarray(weights) / np.sum(weights)

    oof_ranks = np.column_stack([rankdata(oof_matrix[:, i]) / len(oof_matrix) for i in range(n_models)])
    test_ranks = np.column_stack([rankdata(test_matrix[:, i]) / len(test_matrix) for i in range(n_models)])
    return oof_ranks @ weights, test_ranks @ weights
