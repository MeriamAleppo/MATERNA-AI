# -*- coding: utf-8 -*-
"""
Konfigurasi Terpusat MATERNA-AI:
Definisi fitur Dual-Engine (Tier 1 Faskes Primer vs Tier 2 RSUD Rujukan),
batas fisiologis validasi klinis, label fitur Bahasa Indonesia, dan nilai referensi normal.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
PRESENTATION_DIR = BASE_DIR / "presentation"
REPORT_DIR = BASE_DIR / "report"
FIGURES_DIR = REPORT_DIR / "figures"

# 12 Variabel Klinis Dasar Inti untuk Faskes Primer (Puskesmas Pembantu / Posyandu)
TIER1_CORE_12_FEATURES = [
    "maternal_age",
    "gestational_age_weeks",
    "gravida",
    "nullipara",
    "preg_interval_yrs",
    "pre_pregnancy_bmi",
    "lila_cm",
    "systolic_bp",
    "diastolic_bp",
    "map_mmhg",
    "prev_pe_history",
    "comorbid_htn_dm",
]

# Fitur Klinis Dasar Tier 1 Lengkap (15 fitur pada skema dataset)
TIER1_FEATURES = [
    "gestational_age_weeks",
    "maternal_age",
    "gravida",
    "parity",
    "nullipara",
    "preg_interval_yrs",
    "prev_pe_history",
    "family_htn_history",
    "gest_diabetes",
    "pre_pregnancy_bmi",
    "lila_cm",
    "systolic_bp",
    "diastolic_bp",
    "map_mmhg",
    "pulse_pressure",
]

# Fitur Pelacakan Dinamika Longitudinal ANC (Trajectory Tracking)
TRAJECTORY_FEATURES = [
    "sbp_trimester1",
    "dbp_trimester1",
    "delta_sbp",
    "delta_dbp",
    "delta_map",
    "weight_gain_kg_per_week",
    "edema_progression",
]

# Fitur Engine Tier 1 + Longitudinal Trajectory (Digunakan pada Mode Bidan / Faskes Primer)
TIER1_ENGINE_FEATURES = TIER1_FEATURES + TRAJECTORY_FEATURES

# Fitur Biomarker Organ Target & Doppler (Khusus RSUD Rujukan / Mode Tier 2)
TIER2_BIOMARKER_FEATURES = [
    "proteinuria_ordinal",
    "urine_pcr",
    "platelet_count_k_ul",
    "serum_creatinine_mg_dl",
    "sgot_ast_u_l",
    "sgpt_alt_u_l",
    "uric_acid_mg_dl",
    "utapi_doppler",
]

# Seluruh 30 Fitur Prediktor (Mode Tier 2 RSUD Rujukan)
ALL_FEATURES = TIER1_FEATURES + TRAJECTORY_FEATURES + TIER2_BIOMARKER_FEATURES

# Fitur Kategorikal/Diskrit untuk SMOTE-NC
CATEGORICAL_FEATURE_NAMES = [
    "nullipara",
    "prev_pe_history",
    "family_htn_history",
    "gest_diabetes",
    "edema_progression",
    "proteinuria_ordinal",
]

SUBTYPE_MAP = {
    0: "Normal / Non-Preeklamsia",
    1: "Late-Onset Preeclampsia (>= 34 Minggu)",
    2: "Early-Onset Preeclampsia (< 34 Minggu)",
}

SUBTYPE_SHORT_MAP = {
    0: "Normal",
    1: "Late_Onset_PE",
    2: "Early_Onset_PE",
}

DIPSTICK_MAP = {
    "Negatif": 0,
    "0": 0,
    "+1": 1,
    "+2": 2,
    "+3": 3,
}

INV_DIPSTICK_MAP = {0: "Negatif", 1: "+1", 2: "+2", 3: "+3"}

# Batas Fisiologis untuk Automated Clinical Data Auditing & Boundary Validator
PHYSIOLOGICAL_BOUNDS = {
    "maternal_age": (14, 52, "Tahun", "Usia Ibu"),
    "gestational_age_weeks": (12, 42, "Minggu", "Usia Kehamilan"),
    "gravida": (1, 12, "Kali", "Gravida"),
    "parity": (0, 10, "Kali", "Paritas"),
    "preg_interval_yrs": (0.0, 20.0, "Tahun", "Jarak Kehamilan"),
    "pre_pregnancy_bmi": (14.0, 50.0, "kg/m²", "IMT Pra-Kehamilan"),
    "lila_cm": (15.0, 45.0, "cm", "Lingkar Lengan Atas (LILA)"),
    "systolic_bp": (60, 250, "mmHg", "Tensi Sistolik"),
    "diastolic_bp": (40, 160, "mmHg", "Tensi Diastolik"),
    "map_mmhg": (50.0, 190.0, "mmHg", "Mean Arterial Pressure (MAP)"),
    "sbp_trimester1": (60, 220, "mmHg", "Sistolik Trimester 1"),
    "dbp_trimester1": (40, 140, "mmHg", "Diastolik Trimester 1"),
    "weight_gain_kg_per_week": (0.0, 2.5, "kg/minggu", "Laju Penambahan BB"),
    "proteinuria_ordinal": (0, 3, "Skala 0-3", "Proteinuria Dipstick"),
    "urine_pcr": (0.01, 5.0, "mg/mg", "Rasio Protein/Kreatinin Urin"),
    "platelet_count_k_ul": (20, 600, "10³/µL", "Hitung Trombosit"),
    "serum_creatinine_mg_dl": (0.2, 5.0, "mg/dL", "Kreatinin Serum"),
    "sgot_ast_u_l": (5, 500, "U/L", "Enzim SGOT (AST)"),
    "sgpt_alt_u_l": (5, 500, "U/L", "Enzim SGPT (ALT)"),
    "uric_acid_mg_dl": (1.5, 15.0, "mg/dL", "Asam Urat Serum"),
    "utapi_doppler": (0.3, 3.5, "PI", "Indeks Doppler Arteri Uterina (UtAPI)"),
}

# Label ramah pengguna & ambang klinis untuk XAI (SHAP & LIME)
FEATURE_CLINICAL_META = {
    "gestational_age_weeks": {"label": "Usia Kehamilan", "unit": "minggu", "cutoff": 34, "direction": "context"},
    "maternal_age": {"label": "Usia Ibu", "unit": "tahun", "cutoff": 35, "direction": "high"},
    "gravida": {"label": "Gravida (Kehamilan ke-)", "unit": "", "cutoff": 3, "direction": "high"},
    "parity": {"label": "Paritas", "unit": "", "cutoff": 0, "direction": "low"},
    "nullipara": {"label": "Status Nullipara", "unit": "", "cutoff": 1, "direction": "binary"},
    "preg_interval_yrs": {"label": "Jarak Kehamilan", "unit": "tahun", "cutoff": 5.0, "direction": "high"},
    "prev_pe_history": {"label": "Riwayat Preeklamsia", "unit": "", "cutoff": 1, "direction": "binary"},
    "family_htn_history": {"label": "Riwayat Hipertensi Keluarga", "unit": "", "cutoff": 1, "direction": "binary"},
    "gest_diabetes": {"label": "Komorbid Diabetes Gestasional", "unit": "", "cutoff": 1, "direction": "binary"},
    "pre_pregnancy_bmi": {"label": "IMT Pra-Kehamilan", "unit": "kg/m²", "cutoff": 25.0, "direction": "high"},
    "lila_cm": {"label": "Lingkar Lengan Atas (LILA)", "unit": "cm", "cutoff": 27.5, "direction": "high"},
    "systolic_bp": {"label": "Tensi Sistolik", "unit": "mmHg", "cutoff": 140, "direction": "high"},
    "diastolic_bp": {"label": "Tensi Diastolik", "unit": "mmHg", "cutoff": 90, "direction": "high"},
    "map_mmhg": {"label": "Mean Arterial Pressure (MAP)", "unit": "mmHg", "cutoff": 95.0, "direction": "high"},
    "pulse_pressure": {"label": "Pulse Pressure", "unit": "mmHg", "cutoff": 50, "direction": "high"},
    "sbp_trimester1": {"label": "Sistolik Baseline T1", "unit": "mmHg", "cutoff": 120, "direction": "high"},
    "dbp_trimester1": {"label": "Diastolik Baseline T1", "unit": "mmHg", "cutoff": 80, "direction": "high"},
    "delta_sbp": {"label": "Lonjakan Sistolik (ΔSBP)", "unit": "mmHg", "cutoff": 15, "direction": "high"},
    "delta_dbp": {"label": "Lonjakan Diastolik (ΔDBP)", "unit": "mmHg", "cutoff": 10, "direction": "high"},
    "delta_map": {"label": "Lonjakan MAP (ΔMAP)", "unit": "mmHg", "cutoff": 12.0, "direction": "high"},
    "weight_gain_kg_per_week": {"label": "Laju Kenaikan BB", "unit": "kg/mgg", "cutoff": 0.65, "direction": "high"},
    "edema_progression": {"label": "Progresivitas Edema", "unit": "skala", "cutoff": 1, "direction": "high"},
    "proteinuria_ordinal": {"label": "Proteinuria Dipstick", "unit": "+", "cutoff": 1, "direction": "high"},
    "urine_pcr": {"label": "Rasio Protein/Kreatinin Urin", "unit": "mg/mg", "cutoff": 0.30, "direction": "high"},
    "platelet_count_k_ul": {"label": "Hitung Trombosit", "unit": "10³/µL", "cutoff": 150, "direction": "low"},
    "serum_creatinine_mg_dl": {"label": "Kreatinin Serum", "unit": "mg/dL", "cutoff": 1.0, "direction": "high"},
    "sgot_ast_u_l": {"label": "Enzim Hepar SGOT (AST)", "unit": "U/L", "cutoff": 40, "direction": "high"},
    "sgpt_alt_u_l": {"label": "Enzim Hepar SGPT (ALT)", "unit": "U/L", "cutoff": 40, "direction": "high"},
    "uric_acid_mg_dl": {"label": "Asam Urat Serum", "unit": "mg/dL", "cutoff": 5.5, "direction": "high"},
    "utapi_doppler": {"label": "Doppler Arteri Uterina (UtAPI)", "unit": "PI", "cutoff": 1.25, "direction": "high"},
}
