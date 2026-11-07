# -*- coding: utf-8 -*-
"""
Modul 4: Penanganan Class Imbalance via SMOTE-NC (Khusus Training Fold)
Mengimplementasikan algoritma SMOTE-NC (Chawla et al., 2002) untuk kombinasi fitur
kontinu dan nominal/kategorikal, serta estimator wrapper Scikit-Learn sehingga
oversampling hanya dieksekusi di dalam training fold (100% bebas data leakage).
"""

from typing import List, Tuple, Optional
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone


def smote_nc_resample(
    X: np.ndarray,
    y: np.ndarray,
    categorical_indices: Optional[List[int]] = None,
    k_neighbors: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Melakukan oversampling kelas minoritas menggunakan algoritma SMOTE-NC
    (Synthetic Minority Over-sampling Technique for Nominal and Continuous features):
    - Menghitung median standar deviasi fitur kontinu kelas minoritas sebagai bobot
      penalti jarak pada fitur kategorikal yang berbeda nilai.
    - Mencari k-tetangga terdekat sesama kelas minoritas dengan metrik jarak campuran SMOTE-NC.
    - Menginterpolasi fitur kontinu secara linear dan menentukan nilai fitur kategorikal
      melalui majority vote dari k-tetangga terdekat.
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y)
    n_samples, n_features = X_arr.shape

    if categorical_indices is None:
        categorical_indices = []
    cat_idx = sorted(set(i for i in categorical_indices if 0 <= i < n_features))
    cont_idx = [i for i in range(n_features) if i not in cat_idx]

    classes, counts = np.unique(y_arr, return_counts=True)
    if len(classes) <= 1:
        return X_arr, y_arr

    max_count = int(np.max(counts))
    rng = np.random.default_rng(random_state)

    X_resampled_list = [X_arr]
    y_resampled_list = [y_arr]

    for cls, count in zip(classes, counts):
        n_to_generate = max_count - int(count)
        if n_to_generate <= 0:
            continue

        X_min = X_arr[y_arr == cls]
        n_min = len(X_min)
        if n_min < 2:
            continue

        k = min(k_neighbors, n_min - 1)

        # Hitung median standar deviasi fitur kontinu pada kelas minoritas (bobot penalti nominal SMOTE-NC)
        if len(cont_idx) > 0:
            cont_stds = np.std(X_min[:, cont_idx], axis=0)
            med_std = float(np.median(cont_stds)) if np.any(cont_stds > 0) else 1.0
        else:
            med_std = 1.0

        # Matriks jarak pasangan antar sampel minoritas
        diff_cont_sq = np.zeros((n_min, n_min), dtype=float)
        if len(cont_idx) > 0:
            c_data = X_min[:, cont_idx]
            diff = c_data[:, None, :] - c_data[None, :, :]
            diff_cont_sq = np.sum(diff ** 2, axis=2)

        diff_cat_pen = np.zeros((n_min, n_min), dtype=float)
        if len(cat_idx) > 0:
            cat_data = X_min[:, cat_idx]
            mismatch = (cat_data[:, None, :] != cat_data[None, :, :]).astype(float)
            diff_cat_pen = np.sum(mismatch, axis=2) * (med_std ** 2)

        dist_matrix = np.sqrt(diff_cont_sq + diff_cat_pen)
        np.fill_diagonal(dist_matrix, np.inf)

        nn_indices = np.argsort(dist_matrix, axis=1)[:, :k]

        synth_samples = np.zeros((n_to_generate, n_features), dtype=float)
        base_indices = rng.integers(0, n_min, size=n_to_generate)

        for s_idx, b_idx in enumerate(base_indices):
            neighbor_pool = nn_indices[b_idx]
            chosen_nn = neighbor_pool[rng.integers(0, len(neighbor_pool))]
            gap = rng.random()

            # Interpolasi fitur kontinu
            if len(cont_idx) > 0:
                synth_samples[s_idx, cont_idx] = (
                    X_min[b_idx, cont_idx] + gap * (X_min[chosen_nn, cont_idx] - X_min[b_idx, cont_idx])
                )

            # Majority vote dari k-tetangga untuk fitur kategorikal
            if len(cat_idx) > 0:
                for c_i in cat_idx:
                    vals, v_counts = np.unique(X_min[neighbor_pool, c_i], return_counts=True)
                    synth_samples[s_idx, c_i] = vals[np.argmax(v_counts)]

        X_resampled_list.append(synth_samples)
        y_resampled_list.append(np.full(n_to_generate, cls, dtype=y_arr.dtype))

    return np.vstack(X_resampled_list), np.concatenate(y_resampled_list)


class SMOTENCClassifier(ClassifierMixin, BaseEstimator):
    """
    Meta-estimator Scikit-Learn yang mengeksekusi SMOTE-NC hanya saat `.fit(X, y)`
    dipanggil pada training fold, lalu meneruskan data seimbang ke classifier inti.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        estimator=None,
        categorical_indices: Optional[List[int]] = None,
        use_smote_nc: bool = True,
        k_neighbors: int = 5,
        random_state: int = 42,
    ):
        self.estimator = estimator
        self.categorical_indices = categorical_indices
        self.use_smote_nc = use_smote_nc
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)
        if self.use_smote_nc:
            X_fit, y_fit = smote_nc_resample(
                X_arr,
                y_arr,
                categorical_indices=self.categorical_indices,
                k_neighbors=self.k_neighbors,
                random_state=self.random_state,
            )
        else:
            X_fit, y_fit = X_arr, y_arr

        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X_fit, y_fit)
        self.classes_ = getattr(self.estimator_, "classes_", np.unique(y_arr))
        return self

    def predict(self, X):
        return self.estimator_.predict(np.asarray(X, dtype=float))

    def predict_proba(self, X):
        return self.estimator_.predict_proba(np.asarray(X, dtype=float))

