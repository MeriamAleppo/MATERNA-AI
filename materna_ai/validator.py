# -*- coding: utf-8 -*-
"""
Modul 7: Automated Clinical Data Auditing, Boundary Validator, & Smart Imputation
Mengecek logika obstetrik, konsistensi hemodinamik, batas fisiologis ekstrem,
serta melakukan imputasi cerdas (median per gestational age untuk numerik, modus untuk kategorikal)
dan deduplikasi rekam medis.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from .config import PHYSIOLOGICAL_BOUNDS, DIPSTICK_MAP


def compute_map(systolic_bp: float, diastolic_bp: float) -> float:
    """Komputasi otomatis Mean Arterial Pressure: (Sistolik + 2*Diastolik) / 3."""
    return round((float(systolic_bp) + 2.0 * float(diastolic_bp)) / 3.0, 2)


def validate_and_audit_patient_input(patient: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Memvalidasi satu rekaman pasien secara real-time (saat input di Streamlit / API / CLI):
    1. Validasi logika obstetrik:
       - Jika gravida == 1 -> otomatis nullipara = 1, parity = 0, preg_interval_yrs = 0.0, prev_pe_history = 0.
       - Jika nullipara == 1 dan gravida == 1 -> preg_interval_yrs = 0.0.
       - Jika parity >= gravida -> dikoreksi menjadi max(0, gravida - 1).
    2. Validasi hemodinamik:
       - Memastikan diastolic_bp < systolic_bp (dan minimal selisih tekanan nadi >= 15 mmHg).
       - Menghitung ulang map_mmhg, pulse_pressure, delta_sbp, delta_dbp, delta_map secara otomatis.
    3. Boundary check nilai fisiologis ekstrem (LILA 15-45 cm, Tensi 60-250 mmHg, dll.).
    """
    cleaned = dict(patient)
    audit_warnings: List[str] = []

    # 1. Boundary check nilai fisiologis ekstrem
    for field, (low, high, unit, label) in PHYSIOLOGICAL_BOUNDS.items():
        if field in cleaned and cleaned[field] is not None and not pd.isna(cleaned[field]):
            val = float(cleaned[field])
            if val < low or val > high:
                clamped = float(np.clip(val, low, high))
                audit_warnings.append(
                    f"[Boundary Validator] {label} ({val} {unit}) di luar rentang fisiologis "
                    f"[{low}–{high} {unit}]. Dikoreksi otomatis ke {clamped} {unit}."
                )
                cleaned[field] = int(clamped) if isinstance(low, int) and isinstance(high, int) else round(clamped, 2)

    # 2. Validasi logika obstetrik
    gravida = int(cleaned.get("gravida", 1))
    nullipara = int(cleaned.get("nullipara", 1 if gravida == 1 else 0))
    parity = int(cleaned.get("parity", 0 if nullipara == 1 else max(1, gravida - 1)))
    preg_interval = float(cleaned.get("preg_interval_yrs", 0.0))

    if gravida <= 1:
        gravida = 1
        if nullipara != 1 or parity != 0 or preg_interval != 0.0:
            audit_warnings.append(
                "[Obstetric Rule] Gravida == 1 terdeteksi: otomatis mengatur Nullipara = 1 (Ya), "
                "Paritas = 0, dan Jarak Kehamilan = 0.0 tahun."
            )
        nullipara = 1
        parity = 0
        preg_interval = 0.0
        if int(cleaned.get("prev_pe_history", 0)) == 1:
            audit_warnings.append(
                "[Obstetric Rule] Gravida == 1 (kehamilan pertama): Riwayat Preeklamsia sebelumnya disesuaikan ke 0."
            )
            cleaned["prev_pe_history"] = 0
    else:
        if nullipara == 1 and parity > 0:
            audit_warnings.append(
                "[Obstetric Rule] Nullipara == 1 tidak konsisten dengan Paritas > 0: Paritas disesuaikan ke 0 "
                "dan Jarak Kehamilan = 0.0 tahun."
            )
            parity = 0
            preg_interval = 0.0
        elif nullipara == 0 and parity == 0:
            parity = max(1, gravida - 1)
            audit_warnings.append(
                f"[Obstetric Rule] Multipara (Gravida={gravida}) dengan Paritas=0 disesuaikan menjadi Paritas={parity}."
            )
        if parity >= gravida:
            parity = max(0, gravida - 1)
            audit_warnings.append(
                f"[Obstetric Rule] Paritas tidak boleh >= Gravida ({gravida}): disesuaikan menjadi {parity}."
            )
        if nullipara == 1:
            preg_interval = 0.0

    cleaned["gravida"] = gravida
    cleaned["nullipara"] = nullipara
    cleaned["parity"] = parity
    cleaned["preg_interval_yrs"] = round(preg_interval, 1)

    # 3. Validasi hemodinamik (Diastolik < Sistolik)
    sbp = int(round(float(cleaned.get("systolic_bp", 120))))
    dbp = int(round(float(cleaned.get("diastolic_bp", 78))))
    if dbp >= sbp:
        audit_warnings.append(
            f"[Hemodynamic Validator] Tensi Diastolik ({dbp} mmHg) >= Sistolik ({sbp} mmHg). "
            f"Nilai ditukar/dikoreksi agar Diastolik < Sistolik."
        )
        if sbp >= 60 and dbp > sbp:
            sbp, dbp = dbp, sbp
        if dbp >= sbp:
            dbp = max(40, sbp - 25)
    elif (sbp - dbp) < 12:
        dbp = max(40, sbp - 18)
        audit_warnings.append(
            f"[Hemodynamic Validator] Selisih tekanan nadi terlalu sempit (<12 mmHg). "
            f"Diastolik dikalibrasi ke {dbp} mmHg."
        )

    cleaned["systolic_bp"] = sbp
    cleaned["diastolic_bp"] = dbp
    cleaned["map_mmhg"] = compute_map(sbp, dbp)
    cleaned["pulse_pressure"] = sbp - dbp

    # Validasi Tensi Baseline Trimester 1 & Kalkulasi Longitudinal Trajectory
    sbp_t1 = int(round(float(cleaned.get("sbp_trimester1", max(95, sbp - 10)))))
    dbp_t1 = int(round(float(cleaned.get("dbp_trimester1", max(60, dbp - 6)))))
    if dbp_t1 >= sbp_t1:
        sbp_t1, dbp_t1 = max(sbp_t1, dbp_t1), min(sbp_t1, dbp_t1)
        if dbp_t1 >= sbp_t1:
            dbp_t1 = max(45, sbp_t1 - 22)
        audit_warnings.append(
            "[Hemodynamic Validator] Tensi Trimester 1 tidak konsisten (DBP_T1 >= SBP_T1), telah dikoreksi."
        )

    cleaned["sbp_trimester1"] = sbp_t1
    cleaned["dbp_trimester1"] = dbp_t1
    map_t1 = compute_map(sbp_t1, dbp_t1)
    cleaned["delta_sbp"] = sbp - sbp_t1
    cleaned["delta_dbp"] = dbp - dbp_t1
    cleaned["delta_map"] = round(cleaned["map_mmhg"] - map_t1, 2)

    # 4. Sinkronisasi proteinuria dipstick & ordinal
    if "proteinuria_dipstick" in cleaned and "proteinuria_ordinal" not in cleaned:
        cleaned["proteinuria_ordinal"] = int(DIPSTICK_MAP.get(str(cleaned["proteinuria_dipstick"]), 0))
    elif "proteinuria_ordinal" in cleaned:
        cleaned["proteinuria_ordinal"] = int(np.clip(int(cleaned["proteinuria_ordinal"]), 0, 3))

    # Komorbid gabungan (untuk skema 12 variabel inti Tier 1)
    cleaned["comorbid_htn_dm"] = int(
        bool(cleaned.get("family_htn_history", 0)) or bool(cleaned.get("gest_diabetes", 0))
    )

    return cleaned, audit_warnings


def preprocess_and_clean_dataset(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Membersihkan dataset secara menyeluruh sesuai Kriteria 3e & Modul 7:
    - Deduplikasi berdasarkan `patient_id`
    - Smart Imputation:
      * Numerik: Median per Gestational Age (`gestational_age_weeks`) dengan fallback global median
      * Kategorikal: Modus (nilai terbanyak)
    - Pengecekan logika obstetrik (jika gravida == 1 -> nullipara = 1, parity = 0, preg_interval_yrs = 0)
    - Validasi konsistensi hemodinamik (Diastolik < Sistolik) & boundary check fisiologis
    - Komputasi ulang MAP, Pulse Pressure, dan Delta Longitudinal ANC
    """
    stats: Dict[str, Any] = {}
    stats["initial_rows"] = int(len(df_raw))
    stats["duplicate_rows"] = int(df_raw.duplicated(subset=["patient_id"]).sum())

    df = df_raw.drop_duplicates(subset=["patient_id"], keep="first").copy().reset_index(drop=True)
    stats["after_dedup_rows"] = int(len(df))
    stats["missing_before"] = {k: int(v) for k, v in df.isna().sum().to_dict().items()}

    # 1. Imputasi Kategorikal dengan Modus
    cat_cols = ["proteinuria_dipstick", "faskes_origin"]
    cat_imputed_count = 0
    for c in cat_cols:
        if c in df.columns and df[c].isna().any():
            cat_imputed_count += int(df[c].isna().sum())
            mode_val = df[c].mode(dropna=True).iloc[0]
            df[c] = df[c].fillna(mode_val)
    stats["categorical_imputed"] = cat_imputed_count

    # 2. Smart Imputation Numerik: Median per Gestational Age (dengan fallback Global Median)
    num_missing_cols = ["lila_cm", "urine_pcr", "uric_acid_mg_dl", "utapi_doppler"]
    imputation_details = {}
    for col in num_missing_cols:
        if col in df.columns:
            n_miss = int(df[col].isna().sum())
            imputation_details[col] = n_miss
            if n_miss > 0:
                ga_medians = df.groupby("gestational_age_weeks")[col].transform("median")
                df[col] = df[col].fillna(ga_medians).fillna(df[col].median())
                df[col] = df[col].round(2 if col != "uric_acid_mg_dl" else 1)
    stats["numeric_imputed_by_ga_median"] = imputation_details
    stats["missing_after"] = int(df.isna().sum().sum())

    # 3. Validasi Logika Obstetrik
    grav1_mask = df["gravida"] <= 1
    inconsistent_obstetric = int(
        (
            grav1_mask
            & (
                (df["nullipara"] != 1)
                | (df["parity"] != 0)
                | (df["preg_interval_yrs"] > 0)
                | (df["prev_pe_history"] != 0)
            )
        ).sum()
    )
    df.loc[grav1_mask, "gravida"] = 1
    df.loc[grav1_mask, "nullipara"] = 1
    df.loc[grav1_mask, "parity"] = 0
    df.loc[grav1_mask, "preg_interval_yrs"] = 0.0
    stats["obstetric_inconsistencies_fixed"] = inconsistent_obstetric

    # 4. Validasi Hemodinamik & Boundary Check Fisiologis
    inconsistent_bp = int((df["systolic_bp"] <= df["diastolic_bp"]).sum())
    if inconsistent_bp > 0:
        bad_mask = df["systolic_bp"] <= df["diastolic_bp"]
        df.loc[bad_mask, "diastolic_bp"] = df.loc[bad_mask, "systolic_bp"] - 20
    stats["inconsistent_bp_fixed"] = inconsistent_bp

    boundary_fixes = 0
    for col, (low, high, _, _) in PHYSIOLOGICAL_BOUNDS.items():
        if col in df.columns:
            out_mask = (df[col] < low) | (df[col] > high)
            boundary_fixes += int(out_mask.sum())
            df[col] = df[col].clip(low, high)
    stats["boundary_outliers_clipped"] = boundary_fixes

    # 5. Kalkulasi Ulang Fitur Hemodinamik & Trajectory Longitudinal
    df["map_mmhg"] = np.round((df["systolic_bp"] + 2.0 * df["diastolic_bp"]) / 3.0, 2)
    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
    map_t1 = np.round((df["sbp_trimester1"] + 2.0 * df["dbp_trimester1"]) / 3.0, 2)
    df["delta_sbp"] = df["systolic_bp"] - df["sbp_trimester1"]
    df["delta_dbp"] = df["diastolic_bp"] - df["dbp_trimester1"]
    df["delta_map"] = np.round(df["map_mmhg"] - map_t1, 2)

    # 6. Encoding Kategorikal
    df["proteinuria_ordinal"] = df["proteinuria_dipstick"].map(DIPSTICK_MAP).fillna(0).astype(int)
    faskes_map = {"Puskesmas_Pembantu": 0, "Puskesmas_Induk": 1, "RSUD_Rujukan": 2}
    df["faskes_code"] = df["faskes_origin"].map(faskes_map).fillna(0).astype(int)
    df["comorbid_htn_dm"] = ((df["family_htn_history"] == 1) | (df["gest_diabetes"] == 1)).astype(int)

    return df, stats
