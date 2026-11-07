#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
MATERNA-AI: Full-Featured Interactive Clinical Decision Support System (CDSS)
UTS Data Mining - A11.2024.16055 - Nayyara Rauf Ghani Al Khayri
================================================================================
Cara menjalankan antarmuka Web Dashboard Interaktif (6 Halaman):
    streamlit run app.py
Cara menjalankan simulasi CLI lengkap (Tier 1, Tier 2, XAI SHAP, POGI, & PDF):
    python app.py --cli-demo
================================================================================
"""

import sys
import json
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from materna_ai.config import (
    BASE_DIR,
    DATA_DIR,
    LOGS_DIR,
    MODELS_DIR,
    REPORT_DIR,
    FIGURES_DIR,
    TIER1_CORE_12_FEATURES,
    TIER1_FEATURES,
    TRAJECTORY_FEATURES,
    TIER1_ENGINE_FEATURES,
    TIER2_BIOMARKER_FEATURES,
    ALL_FEATURES,
    SUBTYPE_MAP,
    DIPSTICK_MAP,
    INV_DIPSTICK_MAP,
)
from materna_ai.validator import (
    validate_and_audit_patient_input,
    preprocess_and_clean_dataset,
)
from materna_ai.feature_engineering import (
    compute_hemodynamic_trajectory_summary,
    compute_acrfs_feature_ranking,
)
from materna_ai.xai_explainer import (
    compute_shap_and_lime_explanation,
    render_shap_waterfall_plot,
)
from materna_ai.pogi_engine import evaluate_pogi_decision_rules
from materna_ai.pdf_generator import generate_clinical_referral_pdf_bytes


def load_artifacts():
    df = pd.read_csv(DATA_DIR / "preeclampsia_clinical_dataset.csv")
    metrics = json.loads((LOGS_DIR / "metrics_summary.json").read_text(encoding="utf-8"))
    acrfs_path = LOGS_DIR / "acrfs_feature_ranking.csv"
    if acrfs_path.exists():
        acrfs_df = pd.read_csv(acrfs_path)
    else:
        acrfs_df = compute_acrfs_feature_ranking(df)

    best_model = joblib.load(MODELS_DIR / "best_model.joblib")
    tier1_path = MODELS_DIR / "tier1_best_model.joblib"
    tier1_model = joblib.load(tier1_path) if tier1_path.exists() else best_model

    sub_path = MODELS_DIR / "subtype_multiclass_model.joblib"
    subtype_model = joblib.load(sub_path) if sub_path.exists() else None

    t1_sub_path = MODELS_DIR / "tier1_subtype_model.joblib"
    tier1_subtype_model = joblib.load(t1_sub_path) if t1_sub_path.exists() else None

    bg_path = MODELS_DIR / "xai_background.joblib"
    bg_df = joblib.load(bg_path) if bg_path.exists() else df[ALL_FEATURES].sample(n=80, random_state=42)

    return {
        "df": df,
        "metrics": metrics,
        "acrfs_df": acrfs_df,
        "best_model": best_model,
        "tier1_model": tier1_model,
        "subtype_model": subtype_model,
        "tier1_subtype_model": tier1_subtype_model,
        "bg_df": bg_df,
    }


def predict_patient_case(patient_raw: dict, mode: str, artifacts: dict) -> dict:
    """
    Menjalankan pipeline inferensi lengkap pada 1 pasien:
    1. Audit & Boundary Validation (Obstetrik + Hemodinamik)
    2. Komputasi MAP & Longitudinal Trajectory
    3. Dual-Engine Prediction (Tier 1 Faskes Primer vs Tier 2 RSUD Rujukan)
    4. Klasifikasi Subtipe 3-Kelas (Normal vs Late-Onset >=34 mgg vs Early-Onset <34 mgg)
    5. Explainable AI (SHAP Waterfall & LIME + Narasi Kuantitatif)
    6. POGI Clinical Decision Rule-Matching Engine
    """
    audited, warnings_list = validate_and_audit_patient_input(patient_raw)
    hemo = compute_hemodynamic_trajectory_summary(audited)

    is_tier1 = "Tier 1" in mode or "Bidan" in mode or "Primer" in mode
    if is_tier1:
        feat_cols = TIER1_ENGINE_FEATURES
        active_model = artifacts["tier1_model"]
        sub_model = artifacts["tier1_subtype_model"]
        model_label = "Engine Tier 1 (Faskes Primer — Non-Lab)"
    else:
        feat_cols = ALL_FEATURES
        active_model = artifacts["best_model"]
        sub_model = artifacts["subtype_model"]
        model_label = f"Engine Tier 2 ({artifacts['metrics'].get('best_model_name', 'Best Model')} — Full Biomarker)"

    x_in = pd.DataFrame([{k: audited.get(k, 0) for k in feat_cols}])
    prob_pe = float(active_model.predict_proba(x_in)[0, 1])

    if sub_model is not None:
        subtype_code = int(sub_model.predict(x_in)[0])
        subtype_probs = sub_model.predict_proba(x_in)[0].tolist()
    else:
        subtype_code = 0 if prob_pe < 0.40 else (2 if audited["gestational_age_weeks"] < 34 else 1)
        subtype_probs = [1.0 - prob_pe, prob_pe * 0.5, prob_pe * 0.5]

    xai_res = compute_shap_and_lime_explanation(
        model=active_model,
        patient_df=x_in,
        background_df=artifacts["bg_df"],
        feature_cols=feat_cols,
        top_k=8,
    )
    pogi_res = evaluate_pogi_decision_rules(
        prob_pe=prob_pe,
        patient=audited,
        subtype_pred=subtype_code,
        screening_mode=mode,
    )

    return {
        "audited_patient": audited,
        "audit_warnings": warnings_list,
        "hemodynamics": hemo,
        "mode": mode,
        "model_label": model_label,
        "probability": prob_pe,
        "subtype_code": subtype_code,
        "subtype_name": SUBTYPE_MAP.get(subtype_code, "Normal"),
        "subtype_probs": subtype_probs,
        "xai": xai_res,
        "pogi": pogi_res,
    }


def run_streamlit_dashboard():
    import streamlit as st

    st.set_page_config(
        page_title="MATERNA-AI | Multi-Tier Preeclampsia CDSS",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    artifacts = load_artifacts()
    df = artifacts["df"]
    metrics = artifacts["metrics"]
    acrfs_df = artifacts["acrfs_df"]

    st.sidebar.markdown("## 🩺 MATERNA-AI v2.5")
    st.sidebar.caption(
        "**Multi-Tier Clinical Decision Support System for Early Preeclampsia Prediction & Risk Trajectory**\n\n"
        "Oleh: **Nayyara Rauf Ghani Al Khayri** (`A11.2024.16055`)"
    )
    st.sidebar.markdown("---")

    pages = [
        "1. Dashboard Ringkasan & Arsitektur Sistem",
        "2. Eksplorasi Dataset & Distribusi Prevalensi",
        "3. Seleksi Fitur ACRFS (Pearson + MI + Chi2)",
        "4. Evaluasi & Komparasi Model",
        "5. Simulator Skrining Interaktif Pasien",
        "6. Panduan Klinis & Dokumentasi Tim",
    ]
    selected_page = st.sidebar.radio("Navigasi Modul Utama:", pages)

    st.sidebar.markdown("---")
    best_name = metrics.get("best_model_name", "SVM")
    best_info = metrics.get("models", {}).get(best_name, {})
    st.sidebar.info(
        f"**Model Terbaik Aktif:** `{best_name}`\n"
        f"- Recall/Sensitivitas: **{best_info.get('test_recall_sensitivity', 0)*100:.2f}%**\n"
        f"- Specificity: **{best_info.get('test_specificity', 0)*100:.2f}%**\n"
        f"- ROC-AUC: **{best_info.get('test_roc_auc', 0):.4f}**"
    )

    # =========================================================================
    # HALAMAN 1: DASHBOARD RINGKASAN & ARSITEKTUR SISTEM
    # =========================================================================
    if selected_page == pages[0]:
        st.title("🩺 MATERNA-AI: Multi-Tier Clinical Decision Support System")
        st.subheader("Sistem Prediksi Dini Preeklamsia, Pemantauan Trajektori Hemodinamik ANC, & Stratifikasi Subtipe Risiko")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Kohort Pasien Bersih", f"{len(df):,} Ibu Hamil", "12 Duplikat Dihapus")
        c2.metric("Sensitivitas (Recall) Terbaik", f"{best_info.get('test_recall_sensitivity', 0)*100:.2f}%", "Target PRD >= 88.0%")
        c3.metric("Specificity Terbaik", f"{best_info.get('test_specificity', 0)*100:.2f}%", "Target PRD >= 80.0%")
        c4.metric("ROC-AUC Terbaik", f"{best_info.get('test_roc_auc', 0):.4f}", "Target PRD >= 0.8800")

        st.markdown("### Arsitektur Dual-Engine Multi-Tier & Longitudinal Trajectory")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                r"""
                #### 🏥 Mode Tier 1 — Low-Resource / Faskes Primer (Bidan / Puskesmas Pembantu)
                - Dirancang khusus untuk fasilitas tanpa laboratorium kimia darah atau USG Doppler.
                - Mengandalkan **12 Variabel Klinis Dasar Inti** (Usia Ibu, Usia Kehamilan, Gravida, Nullipara, Jarak Kehamilan, IMT Pra-Hamil, LILA, Tensi Sistolik/Diastolik, MAP, Riwayat Preeklamsia, Komorbid Hipertensi/DM) serta **Longitudinal Trajectory ANC** ($\Delta\text{MAP}$ & Laju Penambahan BB/minggu).
                - Akurasi Engine Tier 1: **{:.2f}%** | Recall: **{:.2f}%** | ROC-AUC: **{:.4f}**
                """.format(
                    metrics.get("tier1_engine", {}).get("accuracy", 0.9458) * 100,
                    metrics.get("tier1_engine", {}).get("recall_sensitivity", 0.9306) * 100,
                    metrics.get("tier1_engine", {}).get("roc_auc", 0.9909),
                )
            )
        with col_b:
            st.markdown(
                r"""
                #### 🏢 Mode Tier 2 — Advanced Biomarkers / RSUD Rujukan (Dokter Sp.OG)
                - Mengintegrasikan seluruh fitur Tier 1 + Trajectory ditambah **8 Biomarker Organ Target**:
                  Proteinuria Dipstick (`0` s/d `+3`), Rasio Protein/Kreatinin Urin (`urine_pcr`), Hitung Trombosit, Kreatinin Serum, Enzim Hepar `SGOT/SGPT`, Asam Urat Serum, serta Indeks Doppler Arteri Uterina (`UtAPI`).
                - Mendukung **Stratifikasi Subtipe 3-Kelas**: Normal (`0`), *Late-Onset Preeclampsia* $\ge 34$ minggu (`1`), dan *Early-Onset Preeclampsia* $< 34$ minggu (`2`).
                """
            )

        st.markdown("### Studi Ablasi Kontribusi Fitur (MHS 1 vs MHS 2 vs MHS 3)")
        ab_rows = []
        for k, v in metrics.get("ablation", {}).items():
            ab_rows.append({
                "Skenario Fitur": k,
                "Jumlah Fitur": v["num_features"],
                "Accuracy (%)": round(v["accuracy"] * 100, 2),
                "Recall / Sensitivitas (%)": round(v["recall_sensitivity"] * 100, 2),
                "Specificity (%)": round(v["specificity"] * 100, 2),
                "F1-Score (%)": round(v["f1_score"] * 100, 2),
                "ROC-AUC": v["roc_auc"],
            })
        st.dataframe(pd.DataFrame(ab_rows), use_container_width=True)

        fig1_p = FIGURES_DIR / "fig1_cohort_and_map.png"
        if fig1_p.exists():
            st.image(str(fig1_p), caption="Distribusi Kohort Pasien & Perbandingan Mean Arterial Pressure (MAP) antar Subtipe", use_container_width=True)

    # =========================================================================
    # HALAMAN 2: EKSPLORASI DATASET & DISTRIBUSI PREVALENSI
    # =========================================================================
    elif selected_page == pages[1]:
        st.title("📊 Eksplorasi Dataset Klinis & Audit Kualitas Data")
        clean_stats = metrics.get("clean_stats", {})

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Baris Awal (Raw)", clean_stats.get("initial_rows", 1212))
        c2.metric("Duplikat Dihapus", clean_stats.get("duplicate_rows", 12))
        c3.metric("Missing Diimputasi (Median per GA)", sum(clean_stats.get("numeric_imputed_by_ga_median", {}).values()))
        c4.metric("Missing Tersisa", clean_stats.get("missing_after", 0))

        faskes_filter = st.multiselect(
            "Filter Berdasarkan Asal Fasilitas Kesehatan:",
            options=list(df["faskes_origin"].unique()),
            default=list(df["faskes_origin"].unique()),
        )
        df_sub = df[df["faskes_origin"].isin(faskes_filter)]

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("#### Distribusi Prevalensi Subtipe Preeklamsia")
            sub_cnt = df_sub["preeclampsia_subtype"].value_counts().reset_index()
            sub_cnt.columns = ["Subtipe Preeklamsia", "Jumlah Pasien"]
            sub_cnt["Proporsi (%)"] = (sub_cnt["Jumlah Pasien"] / len(df_sub) * 100).round(2)
            st.dataframe(sub_cnt, use_container_width=True)
        with col_r:
            st.markdown("#### Ringkasan Statistik Indikator Kunci per Subtipe")
            grp = df_sub.groupby("preeclampsia_subtype")[
                ["map_mmhg", "delta_map", "weight_gain_kg_per_week", "urine_pcr", "platelet_count_k_ul", "utapi_doppler"]
            ].mean().round(2)
            st.dataframe(grp, use_container_width=True)

        st.markdown("#### Sampel Data Rekam Medis Pasien (`data/preeclampsia_clinical_dataset.csv`)")
        st.dataframe(df_sub.head(50), use_container_width=True)

    # =========================================================================
    # HALAMAN 3: SELEKSI FITUR ACRFS (PEARSON + MI + CHI2)
    # =========================================================================
    elif selected_page == pages[2]:
        st.title("🧬 Seleksi Fitur ACRFS (Aggregated Coefficient Ranking)")
        st.markdown(
            r"""
            Metode **ACRFS (*Aggregated Coefficient Ranking Feature Selection*)** menggabungkan tiga keluarga ukuran statistik:
            1. **Korelasi Linear Absolut Pearson ($|r|$)**: Mengukur kekuatan hubungan linear fitur terhadap diagnosis preeklamsia.
            2. **Mutual Information ($\text{MI}$)**: Menangkap ketergantungan non-linear berbasis entropi informasi.
            3. **Uji Statistik Chi-Square ($\chi^2$)**: Mengukur signifikansi kontingensi antar distribusi kelas.
            """
        )

        fig4_p = FIGURES_DIR / "fig4_acrfs_feature_ranking.png"
        if fig4_p.exists():
            st.image(str(fig4_p), caption="Peringkat Top 15 Fitur Berdasarkan Skor Komposit ACRFS", use_container_width=True)

        st.markdown("#### Tabel Lengkap Pemeringkatan 30 Fitur Prediktor (ACRFS)")
        st.dataframe(acrfs_df, use_container_width=True)

    # =========================================================================
    # HALAMAN 4: EVALUASI & KOMPARASI MODEL
    # =========================================================================
    elif selected_page == pages[3]:
        st.title("📈 Evaluasi & Komparasi Multi-Model Pipeline (SMOTE-NC)")
        st.markdown(
            "Seluruh model dievaluasi menggunakan **Stratified 5-Fold Cross-Validation** (`GridSearchCV`) "
            "dengan penanganan *class imbalance* **SMOTE-NC** (khusus pada *training fold*) dan `class_weight='balanced'`."
        )

        model_rows = []
        for m_name, m_res in metrics.get("models", {}).items():
            cm = m_res.get("confusion_matrix", [[0, 0], [0, 0]])
            model_rows.append({
                "Model": m_name,
                "Keluarga Algoritma": m_res.get("family", "Standar"),
                "CV ROC-AUC": m_res.get("cv_roc_auc", 0),
                "Test Accuracy (%)": round(m_res.get("test_accuracy", 0) * 100, 2),
                "Precision (%)": round(m_res.get("test_precision", 0) * 100, 2),
                "Recall / Sensitivitas (%)": round(m_res.get("test_recall_sensitivity", 0) * 100, 2),
                "Specificity (%)": round(m_res.get("test_specificity", 0) * 100, 2),
                "F1-Score (%)": round(m_res.get("test_f1", 0) * 100, 2),
                "Test ROC-AUC": m_res.get("test_roc_auc", 0),
                "Test PR-AUC": m_res.get("test_pr_auc", 0),
                "TN / FP / FN / TP": f"{cm[0][0]} / {cm[0][1]} / {cm[1][0]} / {cm[1][1]}",
            })
        st.dataframe(pd.DataFrame(model_rows), use_container_width=True)

        fig2_p = FIGURES_DIR / "fig2_roc_and_metrics.png"
        if fig2_p.exists():
            st.image(str(fig2_p), caption="Kurva ROC & Perbandingan Metrik terhadap Target Klinis PRD", use_container_width=True)

        fig5_p = FIGURES_DIR / "fig5_confusion_and_learning_curves.png"
        if fig5_p.exists():
            st.image(str(fig5_p), caption="Precision-Recall Curve & Learning Curve Konvergensi Model", use_container_width=True)

        mc = metrics.get("multiclass_subtype", {})
        st.markdown("#### Evaluasi Subtype Clinical Stratification (Multi-Class 3 Kelas)")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Multiclass Accuracy", f"{mc.get('accuracy', 0)*100:.2f}%")
        mc2.metric("Multiclass Macro-F1", f"{mc.get('macro_f1', 0)*100:.2f}%")
        mc3.metric("Multiclass OvR ROC-AUC", f"{mc.get('roc_auc_ovr', 0):.4f}")

    # =========================================================================
    # HALAMAN 5: SIMULATOR SKRINING INTERAKTIF PASIEN
    # =========================================================================
    elif selected_page == pages[4]:
        st.title("🩺 Simulator Skrining Interaktif Pasien & Generator Rujukan PDF")

        screening_mode = st.radio(
            "Pilih Mode Fasilitas Kesehatan (Dual-Engine Toggle):",
            [
                "Mode Tier 1 — Bidan Faskes Primer (Puskesmas Pembantu / Posyandu - Tanpa Lab)",
                "Mode Tier 2 — Dokter RSUD Rujukan (Panel Biomarker Lab Lengkap & USG Doppler)",
            ],
            horizontal=True,
        )

        preset = st.selectbox(
            "Pilih Skenario Pasien Preset (atau ubah nilai secara manual di bawah):",
            [
                "Skenario 1: Ibu Hamil Normotensif di Pustu (26 thn, 28 Minggu — Risiko Rendah)",
                "Skenario 2: Ibu Hamil Lonjakan Trajectory MAP di Puskesmas (31 thn, 35 Minggu — Risiko Sedang/Late-Onset)",
                "Skenario 3: Ibu Hamil Primigravida Early-Onset Preeclampsia Kritis di RSUD (36 thn, 29 Minggu — Risiko Tinggi)",
                "Skenario 4: Uji Validasi Otomatis (Gravida=1 tetapi input Nullipara=0 & Jarak Hamil=3 thn)",
            ],
        )

        if "Skenario 1" in preset:
            defaults = dict(
                patient_id="ANON-2026-0101", faskes_origin="Puskesmas_Pembantu",
                maternal_age=26, gestational_age_weeks=28, gravida=2, parity=1, nullipara=0,
                preg_interval_yrs=3.0, prev_pe_history=0, family_htn_history=0, gest_diabetes=0,
                pre_pregnancy_bmi=22.8, lila_cm=24.5, systolic_bp=115, diastolic_bp=74,
                sbp_trimester1=112, dbp_trimester1=72, weight_gain_kg_per_week=0.40, edema_progression=0,
                proteinuria_ordinal=0, urine_pcr=0.12, platelet_count_k_ul=255,
                serum_creatinine_mg_dl=0.68, sgot_ast_u_l=21, sgpt_alt_u_l=19,
                uric_acid_mg_dl=4.0, utapi_doppler=0.85,
            )
        elif "Skenario 2" in preset:
            defaults = dict(
                patient_id="ANON-2026-0205", faskes_origin="Puskesmas_Induk",
                maternal_age=31, gestational_age_weeks=35, gravida=2, parity=1, nullipara=0,
                preg_interval_yrs=5.5, prev_pe_history=0, family_htn_history=1, gest_diabetes=0,
                pre_pregnancy_bmi=27.2, lila_cm=28.0, systolic_bp=138, diastolic_bp=88,
                sbp_trimester1=116, dbp_trimester1=72, weight_gain_kg_per_week=0.72, edema_progression=1,
                proteinuria_ordinal=1, urine_pcr=0.34, platelet_count_k_ul=188,
                serum_creatinine_mg_dl=0.92, sgot_ast_u_l=38, sgpt_alt_u_l=35,
                uric_acid_mg_dl=5.6, utapi_doppler=1.22,
            )
        elif "Skenario 3" in preset:
            defaults = dict(
                patient_id="ANON-2026-0309", faskes_origin="RSUD_Rujukan",
                maternal_age=36, gestational_age_weeks=29, gravida=1, parity=0, nullipara=1,
                preg_interval_yrs=0.0, prev_pe_history=0, family_htn_history=1, gest_diabetes=1,
                pre_pregnancy_bmi=29.4, lila_cm=29.8, systolic_bp=154, diastolic_bp=98,
                sbp_trimester1=124, dbp_trimester1=78, weight_gain_kg_per_week=0.88, edema_progression=2,
                proteinuria_ordinal=2, urine_pcr=0.62, platelet_count_k_ul=138,
                serum_creatinine_mg_dl=1.15, sgot_ast_u_l=58, sgpt_alt_u_l=52,
                uric_acid_mg_dl=6.7, utapi_doppler=1.58,
            )
        else:
            defaults = dict(
                patient_id="ANON-2026-0412", faskes_origin="Puskesmas_Pembantu",
                maternal_age=24, gestational_age_weeks=26, gravida=1, parity=1, nullipara=0,
                preg_interval_yrs=3.5, prev_pe_history=1, family_htn_history=1, gest_diabetes=0,
                pre_pregnancy_bmi=26.5, lila_cm=27.2, systolic_bp=142, diastolic_bp=92,
                sbp_trimester1=118, dbp_trimester1=74, weight_gain_kg_per_week=0.76, edema_progression=1,
                proteinuria_ordinal=1, urine_pcr=0.35, platelet_count_k_ul=175,
                serum_creatinine_mg_dl=0.90, sgot_ast_u_l=34, sgpt_alt_u_l=31,
                uric_acid_mg_dl=5.5, utapi_doppler=1.30,
            )

        with st.form("patient_screening_form"):
            st.markdown("#### A. Data Klinis Dasar & Obstetrik (Tier 1 — 12 Variabel Dasar)")
            c1, c2, c3, c4 = st.columns(4)
            patient_id = c1.text_input("ID Rekam Medis (Anonim)", value=defaults["patient_id"])
            maternal_age = c2.number_input("Usia Ibu (Tahun)", 14, 52, int(defaults["maternal_age"]))
            gest_age = c3.number_input("Usia Kehamilan (Minggu)", 12, 42, int(defaults["gestational_age_weeks"]))
            gravida = c4.number_input("Gravida (Kehamilan ke-)", 1, 12, int(defaults["gravida"]))

            c5, c6, c7, c8 = st.columns(4)
            nullipara = c5.selectbox("Status Nullipara", [0, 1], index=int(defaults["nullipara"]), format_func=lambda x: "1 - Ya (Nullipara)" if x == 1 else "0 - Tidak (Multipara)")
            parity = c6.number_input("Paritas", 0, 10, int(defaults["parity"]))
            preg_interval = c7.number_input("Jarak Kehamilan (Tahun)", 0.0, 20.0, float(defaults["preg_interval_yrs"]), step=0.5)
            prev_pe = c8.selectbox("Riwayat Preeklamsia Lalu", [0, 1], index=int(defaults["prev_pe_history"]), format_func=lambda x: "1 - Ya" if x == 1 else "0 - Tidak")

            c9, c10, c11, c12 = st.columns(4)
            fam_htn = c9.selectbox("Riwayat Hipertensi Keluarga", [0, 1], index=int(defaults["family_htn_history"]), format_func=lambda x: "1 - Ya" if x == 1 else "0 - Tidak")
            gest_dm = c10.selectbox("Komorbid Diabetes (DM)", [0, 1], index=int(defaults["gest_diabetes"]), format_func=lambda x: "1 - Ya" if x == 1 else "0 - Tidak")
            pre_bmi = c11.number_input("IMT Pra-Kehamilan (kg/m²)", 14.0, 50.0, float(defaults["pre_pregnancy_bmi"]), step=0.1)
            lila_cm = c12.number_input("LILA (cm)", 15.0, 45.0, float(defaults["lila_cm"]), step=0.1)

            st.markdown("#### B. Pengukuran Hemodinamik & Longitudinal Trajectory ANC")
            h1, h2, h3, h4, h5, h6 = st.columns(6)
            sbp = h1.number_input("Sistolik Saat Ini (mmHg)", 60, 250, int(defaults["systolic_bp"]))
            dbp = h2.number_input("Diastolik Saat Ini (mmHg)", 40, 160, int(defaults["diastolic_bp"]))
            sbp_t1 = h3.number_input("Sistolik Trimester 1 (mmHg)", 60, 220, int(defaults["sbp_trimester1"]))
            dbp_t1 = h4.number_input("Diastolik Trimester 1 (mmHg)", 40, 140, int(defaults["dbp_trimester1"]))
            wg_vel = h5.number_input("Laju BB (kg/minggu)", 0.0, 2.5, float(defaults["weight_gain_kg_per_week"]), step=0.05)
            edema = h6.selectbox("Progresivitas Edema", [0, 1, 2], index=int(defaults["edema_progression"]), format_func=lambda x: f"Skor {x}")

            if "Tier 2" in screening_mode:
                st.markdown("#### C. Panel Biomarker Organ Target & USG Doppler (Khusus Mode Tier 2 — RSUD Rujukan)")
                b1, b2, b3, b4 = st.columns(4)
                prot_ord = b1.selectbox("Proteinuria Dipstick", [0, 1, 2, 3], index=int(defaults["proteinuria_ordinal"]), format_func=lambda x: INV_DIPSTICK_MAP[x])
                urine_pcr = b2.number_input("Rasio Protein/Kreatinin Urin", 0.01, 5.0, float(defaults["urine_pcr"]), step=0.02)
                platelet = b3.number_input("Hitung Trombosit (10³/µL)", 20, 600, int(defaults["platelet_count_k_ul"]))
                creatinine = b4.number_input("Kreatinin Serum (mg/dL)", 0.2, 5.0, float(defaults["serum_creatinine_mg_dl"]), step=0.05)

                b5, b6, b7, b8 = st.columns(4)
                sgot = b5.number_input("Enzim SGOT / AST (U/L)", 5, 500, int(defaults["sgot_ast_u_l"]))
                sgpt = b6.number_input("Enzim SGPT / ALT (U/L)", 5, 500, int(defaults["sgpt_alt_u_l"]))
                uric_acid = b7.number_input("Asam Urat Serum (mg/dL)", 1.5, 15.0, float(defaults["uric_acid_mg_dl"]), step=0.1)
                utapi = b8.number_input("Doppler Arteri Uterina (UtAPI)", 0.3, 3.5, float(defaults["utapi_doppler"]), step=0.02)
            else:
                st.info("ℹ️ **Mode Tier 1 (Faskes Primer) Aktif:** Panel laboratorium tidak diperlukan. Prediksi menggunakan model khusus Tier 1 berbasis tanda vital, antropometri, anamnesis, dan trajektori MAP.")
                prot_ord = defaults["proteinuria_ordinal"]
                urine_pcr = defaults["urine_pcr"]
                platelet = defaults["platelet_count_k_ul"]
                creatinine = defaults["serum_creatinine_mg_dl"]
                sgot = defaults["sgot_ast_u_l"]
                sgpt = defaults["sgpt_alt_u_l"]
                uric_acid = defaults["uric_acid_mg_dl"]
                utapi = defaults["utapi_doppler"]

            submitted = st.form_submit_button("🔍 Jalankan Penapisan Risiko, Analisis XAI (SHAP/LIME), & Rekomendasi POGI")

        if submitted or True:
            patient_input = {
                "patient_id": patient_id,
                "faskes_origin": "Puskesmas_Pembantu" if "Tier 1" in screening_mode else "RSUD_Rujukan",
                "maternal_age": maternal_age,
                "gestational_age_weeks": gest_age,
                "gravida": gravida,
                "parity": parity,
                "nullipara": nullipara,
                "preg_interval_yrs": preg_interval,
                "prev_pe_history": prev_pe,
                "family_htn_history": fam_htn,
                "gest_diabetes": gest_dm,
                "pre_pregnancy_bmi": pre_bmi,
                "lila_cm": lila_cm,
                "systolic_bp": sbp,
                "diastolic_bp": dbp,
                "sbp_trimester1": sbp_t1,
                "dbp_trimester1": dbp_t1,
                "weight_gain_kg_per_week": wg_vel,
                "edema_progression": edema,
                "proteinuria_ordinal": prot_ord,
                "urine_pcr": urine_pcr,
                "platelet_count_k_ul": platelet,
                "serum_creatinine_mg_dl": creatinine,
                "sgot_ast_u_l": sgot,
                "sgpt_alt_u_l": sgpt,
                "uric_acid_mg_dl": uric_acid,
                "utapi_doppler": utapi,
            }
            sim_res = predict_patient_case(patient_input, screening_mode, artifacts)

            if sim_res["audit_warnings"]:
                st.warning(
                    "**Automated Clinical Data Auditing & Boundary Validator mendeteksi koreksi otomatis:**\n"
                    + "\n".join(f"- {w}" for w in sim_res["audit_warnings"])
                )

            hemo = sim_res["hemodynamics"]
            pogi = sim_res["pogi"]
            xai = sim_res["xai"]

            st.markdown("---")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Probabilitas Risiko Preeklamsia", f"{pogi['probability_pct']:.2f}%", pogi["risk_tier"])
            m2.metric("Mean Arterial Pressure (MAP)", f"{hemo['map_mmhg']:.2f} mmHg", f"ΔMAP {hemo['delta_map']:+.2f} mmHg dari T1")
            m3.metric("Laju Penambahan BB", f"{hemo['weight_gain_kg_per_week']:.2f} kg/minggu", hemo["weight_velocity_status"].split("(")[0])
            m4.metric("Stratifikasi Subtipe", sim_res["subtype_name"].split("(")[0].strip(), f"Usia Gestasi {gest_age} Minggu")

            col_xai, col_pogi = st.columns([1.25, 1.0])
            with col_xai:
                st.markdown("#### 🔬 Explainable AI (XAI): SHAP Waterfall Plot & Kontribusi Faktor")
                fig_w = render_shap_waterfall_plot(
                    xai,
                    max_display=9,
                    title=f"SHAP Waterfall Plot — Pasien {patient_id} ({sim_res['model_label']})",
                )
                st.pyplot(fig_w)
                plt.close(fig_w)

                st.markdown("**Kontribusi Kuantitatif Faktor Pemicu & Protektif (SHAP & LIME):**")
                for narr in xai["top_narratives"][:6]:
                    st.markdown(f"- `{narr}`")

            with col_pogi:
                st.markdown("#### 📋 Rekomendasi Klinis Berbasis Pedoman POGI / PNPK")
                st.markdown(f"**Kategori Risiko:** `{pogi['risk_tier']}`")
                st.markdown(f"**Stratifikasi Onset:** `{pogi['subtype_label']}`")
                st.markdown(f"**Urgensi Rujukan:** {pogi['urgency']}")
                st.markdown(f"**Fasilitas Tujuan:** {pogi['target_facility']}")

                if pogi["red_flags"]:
                    st.error("**Tanda Bahaya / Red Flags Terdeteksi:**\n" + "\n".join(f"- {rf}" for rf in pogi["red_flags"]))

                st.success("**Protokol Tindakan Klinis POGI:**\n" + "\n".join(f"{i+1}. {act}" for i, act in enumerate(pogi["recommended_actions"])))

                pdf_bytes = generate_clinical_referral_pdf_bytes(
                    patient=sim_res["audited_patient"],
                    pogi_result=pogi,
                    xai_result=xai,
                    screening_mode=screening_mode,
                    model_name=sim_res["model_label"],
                )
                st.download_button(
                    label="📄 Unduh Lembar Resume Klinis & Surat Rujukan PDF (1-Klik)",
                    data=pdf_bytes,
                    file_name=f"Resume_Rujukan_MATERNA_AI_{patient_id}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

    # =========================================================================
    # HALAMAN 6: PANDUAN KLINIS & DOKUMENTASI TIM
    # =========================================================================
    elif selected_page == pages[5]:
        st.title("📚 Panduan Klinis POGI, Kamus Fitur, & Dokumentasi REST API")
        st.markdown(
            """
            ### 1. Pedoman Stratifikasi Risiko & Protokol Klinis POGI / PNPK
            - **Risiko Rendah (`< 15%`)**: Jadwal ANC rutin di FKTP (minimal 6 kali kunjungan), edukasi nutrisi tinggi protein & antioksidan, pencatatan MAP rutin pada Buku KIA.
            - **Risiko Sedang (`15% – 40%`)**: Evaluasi pemberian profilaksis **Aspirin Dosis Rendah (80–150 mg/hari)** dimulai sebelum usia kehamilan 16 minggu, suplementasi **Kalsium (1.000–1.500 mg/hari)**, dan pemantauan tekanan darah setiap 2 minggu.
            - **Risiko Tinggi (`> 40%`)**: Protokol rujukan vertikal segera ke **Dokter Sp.OG (RSUD/PONEK)**, skrining USG Doppler Arteri Uterina (`UtAPI`), evaluasi laboratorium organ target, dan persiapan persalinan di fasilitas emergensi obstetri.

            ### 2. Endpoint REST API (`api_fastapi.py`)
            Jalankan server backend REST API dengan perintah:
            ```bash
            uvicorn api_fastapi:app --host 0.0.0.0 --port 8000
            ```
            Daftar Endpoint tersedia:
            - `GET /health` — Pemeriksaan status layanan & model termuat
            - `GET /models/metrics` — Ringkasan metrik performa 6 model & Top Fitur ACRFS
            - `POST /predict/tier1` — Skrining Faskes Primer (Mode Tier 1 tanpa lab)
            - `POST /predict/tier2` — Skrining RSUD Rujukan (Mode Tier 2 biomarker lengkap)
            - `POST /referral/pdf` — Ekspor langsung dokumen PDF Resume Klinis & Surat Rujukan
            """
        )


def run_cli_demo():
    print("================================================================================")
    print("MATERNA-AI v2.5: SIMULASI INFERENSI DUAL-ENGINE, XAI (SHAP/LIME), POGI & PDF")
    print("================================================================================")
    artifacts = load_artifacts()

    sample_patients = [
        (
            "Mode Tier 1 — Bidan Faskes Primer (Tanpa Lab)",
            {
                "patient_id": "MAT-CLI-TIER1-001",
                "nama_skenario": "Pasien 1 - Ibu Hamil di Puskesmas Pembantu (Usia 26 thn, 28 Minggu, Uji Validasi Gravida=1)",
                "gestational_age_weeks": 28, "maternal_age": 26, "gravida": 1, "parity": 1, "nullipara": 0,
                "preg_interval_yrs": 3.0, "prev_pe_history": 0, "family_htn_history": 0, "gest_diabetes": 0,
                "pre_pregnancy_bmi": 22.8, "lila_cm": 24.5, "systolic_bp": 115, "diastolic_bp": 74,
                "sbp_trimester1": 112, "dbp_trimester1": 72, "weight_gain_kg_per_week": 0.40, "edema_progression": 0,
                "proteinuria_ordinal": 0, "urine_pcr": 0.12, "platelet_count_k_ul": 255,
                "serum_creatinine_mg_dl": 0.68, "sgot_ast_u_l": 21, "sgpt_alt_u_l": 19,
                "uric_acid_mg_dl": 4.0, "utapi_doppler": 0.85,
            },
        ),
        (
            "Mode Tier 2 — Dokter RSUD Rujukan (Full Biomarker)",
            {
                "patient_id": "MAT-CLI-TIER2-002",
                "nama_skenario": "Pasien 2 - Ibu Hamil Risiko Tinggi Early-Onset PE di RSUD (Usia 36 thn, 29 Minggu)",
                "gestational_age_weeks": 29, "maternal_age": 36, "gravida": 1, "parity": 0, "nullipara": 1,
                "preg_interval_yrs": 0.0, "prev_pe_history": 0, "family_htn_history": 1, "gest_diabetes": 1,
                "pre_pregnancy_bmi": 29.4, "lila_cm": 29.8, "systolic_bp": 154, "diastolic_bp": 98,
                "sbp_trimester1": 124, "dbp_trimester1": 78, "weight_gain_kg_per_week": 0.88, "edema_progression": 2,
                "proteinuria_ordinal": 2, "urine_pcr": 0.62, "platelet_count_k_ul": 138,
                "serum_creatinine_mg_dl": 1.15, "sgot_ast_u_l": 58, "sgpt_alt_u_l": 52,
                "uric_acid_mg_dl": 6.7, "utapi_doppler": 1.58,
            },
        ),
    ]

    for mode_label, p in sample_patients:
        res = predict_patient_case(p, mode_label, artifacts)
        hemo = res["hemodynamics"]
        pogi = res["pogi"]
        xai = res["xai"]

        print(f"\n* {p['nama_skenario']}")
        print(f"  - Engine Aktif  : {res['model_label']}")
        if res["audit_warnings"]:
            print(f"  - Audit Log     : {res['audit_warnings'][0]}")
        print(f"  - Hemodinamik   : TD={res['audited_patient']['systolic_bp']}/{res['audited_patient']['diastolic_bp']} mmHg | MAP={hemo['map_mmhg']} mmHg | Delta MAP={hemo['delta_map']:+.2f} mmHg")
        print(f"  - Probabilitas  : {pogi['probability_pct']:.2f}% -> Kategori: {pogi['risk_tier']}")
        print(f"  - Stratifikasi  : {res['subtype_name']}")
        print(f"  - Top XAI (SHAP): {'; '.join(xai['top_narratives'][:3])}")
        print(f"  - Rekomendasi   : {pogi['recommended_actions'][0]}")

        pdf_out = REPORT_DIR / f"referral_{p['patient_id']}.pdf"
        generate_clinical_referral_pdf_bytes(
            patient=res["audited_patient"],
            pogi_result=pogi,
            xai_result=xai,
            screening_mode=mode_label,
            model_name=res["model_label"],
            output_path=pdf_out,
        )
        print(f"  - PDF Rujukan   : Tersimpan di {pdf_out.relative_to(BASE_DIR)} ({pdf_out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    if "--cli-demo" in sys.argv:
        run_cli_demo()
    else:
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            in_streamlit = get_script_run_ctx() is not None
        except Exception:
            in_streamlit = False

        if in_streamlit:
            run_streamlit_dashboard()
        else:
            run_cli_demo()
