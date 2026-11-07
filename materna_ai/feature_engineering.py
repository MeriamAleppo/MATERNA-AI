# -*- coding: utf-8 -*-
"""
Modul 2 & 9.3:
- Automated Hemodynamic & Longitudinal Trajectory Tracking
- ACRFS (Aggregated Coefficient Ranking Feature Selection: Pearson + Mutual Information + Chi2)
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.preprocessing import MinMaxScaler
from .config import ALL_FEATURES, FEATURE_CLINICAL_META, TIER1_FEATURES, TRAJECTORY_FEATURES, TIER2_BIOMARKER_FEATURES


def compute_hemodynamic_trajectory_summary(patient: Dict[str, Any]) -> Dict[str, Any]:
    """
    Menghitung indikator hemodinamik & dinamika longitudinal kunjungan ANC:
    - MAP saat ini = (SBP + 2*DBP)/3
    - MAP Trimester 1 = (SBP_T1 + 2*DBP_T1)/3
    - Delta MAP (kenaikan tensi dibanding trimester sebelumnya)
    - Weight Gain Velocity (laju penambahan berat badan per minggu)
    - Status klinis masing-masing indikator untuk deteksi dini lonjakan resistensi vaskular
    """
    sbp = float(patient.get("systolic_bp", 120))
    dbp = float(patient.get("diastolic_bp", 78))
    map_now = round((sbp + 2.0 * dbp) / 3.0, 2)
    pulse_pressure = int(round(sbp - dbp))

    sbp_t1 = float(patient.get("sbp_trimester1", max(95, sbp - 10)))
    dbp_t1 = float(patient.get("dbp_trimester1", max(60, dbp - 6)))
    map_t1 = round((sbp_t1 + 2.0 * dbp_t1) / 3.0, 2)

    delta_sbp = round(sbp - sbp_t1, 1)
    delta_dbp = round(dbp - dbp_t1, 1)
    delta_map = round(map_now - map_t1, 2)
    wg_vel = round(float(patient.get("weight_gain_kg_per_week", 0.45)), 2)

    if map_now >= 106:
        map_status = "Kritis (>=106 mmHg - Hipertensi Berat/Perfusi Kritis)"
    elif map_now >= 95:
        map_status = "Waspada (>=95 mmHg - Resistensi Vaskular Meningkat)"
    else:
        map_status = "Normal (<95 mmHg - Normotensif)"

    if delta_map >= 15:
        trajectory_status = "Lonjakan Tajam (Delta MAP >= +15 mmHg)"
    elif delta_map >= 10:
        trajectory_status = "Kenaikan Moderat (Delta MAP +10 s/d +14.9 mmHg)"
    else:
        trajectory_status = "Stabil (Delta MAP < +10 mmHg)"

    if wg_vel >= 0.75:
        wg_status = "Retensi Cairan Cepat (>=0.75 kg/minggu - Waspada Edema Endotel)"
    elif wg_vel >= 0.55:
        wg_status = "Kenaikan Di Atas Rata-rata (0.55 - 0.74 kg/minggu)"
    else:
        wg_status = "Fisiologis Normal (<0.55 kg/minggu)"

    return {
        "map_mmhg": map_now,
        "map_trimester1": map_t1,
        "pulse_pressure": pulse_pressure,
        "delta_sbp": delta_sbp,
        "delta_dbp": delta_dbp,
        "delta_map": delta_map,
        "weight_gain_kg_per_week": wg_vel,
        "map_status": map_status,
        "trajectory_status": trajectory_status,
        "weight_velocity_status": wg_status,
        "vascular_surge_alert": bool(map_now >= 95 or delta_map >= 12 or wg_vel >= 0.75),
    }


def compute_acrfs_feature_ranking(
    df: pd.DataFrame,
    feature_cols: List[str] = None,
    target_col: str = "preeclampsia_diagnosis",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Menghitung seleksi dan pemeringkatan fitur menggunakan metode ACRFS
    (Aggregated Coefficient Ranking Feature Selection):
    1. Pearson Correlation Coefficient |r|
    2. Mutual Information (MI) Score
    3. Chi-Square (Chi2) Statistic (pada fitur yang dinormalisasi ke non-negatif [0, 1])
    Skor komposit ACRFS dihitung dari rata-rata ketiga metrik yang telah dinormalisasi (Min-Max [0, 1]).
    """
    if feature_cols is None:
        feature_cols = [c for c in ALL_FEATURES if c in df.columns]

    X = df[feature_cols].copy()
    X = X.fillna(X.median())
    y = df[target_col].values

    # 1. Absolute Pearson Correlation |r|
    pearson_scores = np.array([abs(np.corrcoef(X[c].values, y)[0, 1]) for c in feature_cols])
    pearson_scores = np.nan_to_num(pearson_scores, nan=0.0)

    # 2. Mutual Information (MI)
    mi_scores = mutual_info_classif(X.values, y, random_state=random_state)
    mi_scores = np.nan_to_num(mi_scores, nan=0.0)

    # 3. Chi-Square (Chi2) pada matriks non-negatif MinMax [0, 1]
    X_minmax = MinMaxScaler().fit_transform(X.values)
    chi2_stats, _ = chi2(X_minmax, y)
    chi2_stats = np.nan_to_num(chi2_stats, nan=0.0)

    def _norm(arr: np.ndarray) -> np.ndarray:
        ptp = np.max(arr) - np.min(arr)
        if ptp <= 1e-12:
            return np.zeros_like(arr)
        return (arr - np.min(arr)) / ptp

    pearson_norm = _norm(pearson_scores)
    mi_norm = _norm(mi_scores)
    chi2_norm = _norm(chi2_stats)

    acrfs_score = (pearson_norm + mi_norm + chi2_norm) / 3.0

    rows = []
    for idx, col in enumerate(feature_cols):
        if col in TIER1_FEATURES:
            tier_group = "Tier 1 (Klinis Dasar)"
        elif col in TRAJECTORY_FEATURES:
            tier_group = "Trajectory ANC"
        elif col in TIER2_BIOMARKER_FEATURES:
            tier_group = "Tier 2 (Biomarker & Doppler)"
        else:
            tier_group = "Lainnya"

        meta = FEATURE_CLINICAL_META.get(col, {"label": col})
        rows.append({
            "feature": col,
            "clinical_label": meta.get("label", col),
            "tier_group": tier_group,
            "pearson_abs": round(float(pearson_scores[idx]), 4),
            "mutual_info": round(float(mi_scores[idx]), 4),
            "chi2_stat": round(float(chi2_stats[idx]), 4),
            "pearson_norm": round(float(pearson_norm[idx]), 4),
            "mi_norm": round(float(mi_norm[idx]), 4),
            "chi2_norm": round(float(chi2_norm[idx]), 4),
            "acrfs_score": round(float(acrfs_score[idx]), 4),
        })

    rank_df = pd.DataFrame(rows).sort_values("acrfs_score", ascending=False).reset_index(drop=True)
    rank_df["rank"] = np.arange(1, len(rank_df) + 1)
    return rank_df
