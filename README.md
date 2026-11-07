# MATERNA-AI: Multi-Tier Clinical Decision Support System for Early Preeclampsia Prediction & Risk Trajectory

**Ujian Tengah Semester (UTS) Data Mining — Tahun Akademik 2026**  
**Nama Mahasiswa:** Nayyara Rauf Ghani Al Khayri  
**NIM:** A11.2024.16055  

---

## Ringkasan Sistem & Rangkaian 10 Fitur Utama + Aksesoris (v2.5 Full Upgrade)

**MATERNA-AI** adalah Sistem Pendukung Keputusan Klinis Bertingkat (*Multi-Tier Clinical Decision Support System*) yang mengintegrasikan **4 Mesin Prediktif Inti** dan **6 Fitur Aksesoris Klinis**:

### I. Fitur Utama (*Core Predictive Engines*)
1. **Dual-Engine Multi-Tier Screening Architecture**:
   - **Mode Tier 1 (*Low-Resource / Faskes Primer*)**: Penapisan khusus Puskesmas Pembantu / Posyandu / Bidan tanpa laboratorium menggunakan 12 variabel klinis dasar inti (usia ibu, usia kehamilan, gravida, nullipara, jarak kehamilan, IMT pra-kehamilan, LILA, tensi sistolik/diastolik, MAP, riwayat preeklamsia, komorbid hipertensi/DM) serta trajektori kunjungan ANC (`models/tier1_best_model.joblib`).
   - **Mode Tier 2 (*Advanced Biomarkers / RSUD Rujukan*)**: Mengintegrasikan seluruh fitur Tier 1 + Trajectory ditambah 8 biomarker organ target: proteinuria carik celup (`0` s/d `+3`), rasio protein/kreatinin urin (`urine_pcr`), hitung trombosit, enzim hepar `SGOT/SGPT`, kreatinin serum, asam urat, serta mean indeks Doppler Arteri Uterina (`UtAPI`) (`models/best_model.joblib` & `models/tier2_best_model.joblib`).
2. **Automated Hemodynamic & Longitudinal Trajectory Tracking**:
   - Komputasi otomatis $\text{MAP} = \frac{\text{Sistolik} + 2 \times \text{Diastolik}}{3}$ dan *Pulse Pressure*.
   - Pelacakan dinamika kunjungan ANC: $\Delta\text{MAP}$ (kenaikan tensi dibanding trimester sebelumnya) dan laju penambahan berat badan per minggu (*weight gain velocity*).
3. **Subtype Clinical Stratification (Multi-Class)**:
   - Klasifikasi 3 kelas: `(0)` Normal/Non-Preeklamsia, `(1)` *Late-Onset Preeclampsia* ($\ge 34$ minggu), dan `(2)` *Early-Onset Preeclampsia* ($< 34$ minggu dengan risiko komplikasi maternal-fetal kritis) (`models/subtype_multiclass_model.joblib` & `models/tier1_subtype_model.joblib`).
4. **Robust Machine Learning Multi-Model Pipeline (SMOTE-NC & Ensemble)**:
   - 4 Algoritma Komparatif Standar: `KNN`, `Gaussian Naive Bayes`, `Decision Tree`, dan `SVM`, ditambah perluasan Ensemble `XGBoost` & `LightGBM`.
   - Penanganan *class imbalance* via **SMOTE-NC** (khusus pada *training fold* di dalam pipeline CV sehingga 100% bebas *data leakage*) dan `class_weight='balanced'`.
   - Optimasi hyperparameter otomatis via `GridSearchCV` (Stratified 5-Fold CV) melampaui target klinis: **Recall/Sensitivitas $\ge 88\%$**, **Specificity $\ge 80\%$**, dan **ROC-AUC $\ge 0.88$**.

### II. Fitur Aksesoris & Complementary Value-Add
5. **Explainable AI (XAI) Modul (SHAP & LIME)** (`materna_ai/xai_explainer.py`):
   - Menghasilkan **SHAP Waterfall Plot** dan atribusi lokal **LIME** untuk setiap prediksi pasien tunggal disertai narasi kuantitatif faktor pemicu & protektif.
6. **POGI Clinical Decision Rule-Matching Engine** (`materna_ai/pogi_engine.py`):
   - Menerjemahkan probabilitas model & indikator klinis ke protokol PNPK POGI: Risiko Rendah (`<15%`), Risiko Sedang (`15%–40%`: evaluasi profilaksis Aspirin dosis rendah 80–150 mg/hari sebelum 16 minggu & Kalsium), dan Risiko Tinggi (`>40%`: rujukan segera ke Sp.OG, USG Doppler UtAPI, & persiapan emergensi obstetri).
7. **Automated Clinical Data Auditing & Boundary Validator** (`materna_ai/validator.py`):
   - Validasi logika obstetrik otomatis (`gravida == 1` $\rightarrow$ `nullipara = 1`, `parity = 0`, `preg_interval_yrs = 0`), validasi hemodinamik (`Diastolik < Sistolik`), *boundary check* nilai fisiologis ekstrem, serta *Smart Imputation* (median per usia kehamilan untuk numerik, modus untuk kategorikal).
8. **Clinical Summary & Referral PDF Generator** (`materna_ai/pdf_generator.py`):
   - Ekspor lembar resume hasil penapisan & surat rujukan formal berformat PDF siap cetak dalam 1 klik.
9. **Full-Featured 6-Page Streamlit Web Dashboard** (`app.py`):
   - Antarmuka interaktif 6 halaman mencakup Ringkasan Arsitektur, Eksplorasi Dataset, Seleksi Fitur **ACRFS (*Pearson + MI + Chi2*)**, Evaluasi 6 Model, Simulator Skrining Pasien (*Toggle Mode Bidan Faskes Primer vs Mode Dokter RSUD*), dan Dokumentasi Klinis.
10. **REST API Ready Backend** (`api_fastapi.py`):
    - Layanan API modular (`/health`, `/models/metrics`, `/predict/tier1`, `/predict/tier2`, `/referral/pdf`).

---

## Struktur Direktori Proyek

```text
dm-UTS-A11.2024.16055-Nayyara Rauf Ghani Al Khayri/
├── materna_ai/                                 # Paket Modular Core AI & Clinical Engine
│   ├── __init__.py
│   ├── config.py                               # Konfigurasi fitur Tier 1/2, batas fisiologis, & metadata klinis
│   ├── validator.py                            # Data Auditing, Obstetric/Hemodynamic Validator & Smart Imputation
│   ├── feature_engineering.py                  # Kalkulasi MAP, Trajectory ANC, & Seleksi Fitur ACRFS
│   ├── smote_nc.py                             # Algoritma SMOTE-NC khusus training fold (bebas data leakage)
│   ├── xai_explainer.py                        # Modul Explainable AI (SHAP Waterfall & LIME)
│   ├── pogi_engine.py                          # Mesin Aturan Klinis & Protokol Rujukan POGI/PNPK
│   └── pdf_generator.py                        # Generator PDF Resume Klinis & Surat Rujukan 1-Klik
├── data/
│   └── preeclampsia_clinical_dataset.csv       # Dataset klinis & longitudinal maternal (N=1.200 pasien bersih)
├── logs/
│   ├── run_log.txt                             # Log eksekusi audit data, training 6 model, & evaluasi
│   ├── metrics_summary.json                    # Ringkasan metrik terstruktur hasil eksperimen
│   └── acrfs_feature_ranking.csv               # Hasil pemeringkatan 30 fitur dengan metode ACRFS
├── models/
│   ├── decision_tree_model.joblib              # Model 1: Decision Tree Classifier (SMOTE-NC Tuned)
│   ├── knn_model.joblib                        # Model 2: K-Nearest Neighbors (SMOTE-NC Tuned)
│   ├── naive_bayes_model.joblib                # Model 3: Gaussian Naive Bayes (SMOTE-NC Tuned)
│   ├── svm_model.joblib                        # Model 4: Support Vector Machine (SMOTE-NC Tuned)
│   ├── xgboost_model.joblib                    # Model 5: XGBoost Classifier (Ensemble Extension)
│   ├── lightgbm_model.joblib                   # Model 6: LightGBM / HistGradientBoosting (Ensemble Extension)
│   ├── best_model.joblib                       # Model terbaik otomatis hasil Hyperparameter Optimization
│   ├── tier1_best_model.joblib                 # Engine khusus Mode Tier 1 (Faskes Primer Tanpa Lab)
│   ├── tier2_best_model.joblib                 # Engine khusus Mode Tier 2 (RSUD Rujukan Full Biomarker)
│   ├── subtype_multiclass_model.joblib         # Model Subtype Stratification 3-Kelas (Normal/Late/Early-Onset)
│   ├── tier1_subtype_model.joblib              # Model Subtype Stratification khusus Faskes Primer
│   └── xai_background.joblib                   # Distribusi referensi populasi untuk atribusi SHAP & LIME
├── notebooks/
│   └── uts_dm_preeclampsia_optimization.ipynb  # Notebook utama EDA, ACRFS, Modeling, Ablasi, & XAI
├── presentation/
│   └── Link Video Presentasi.txt               # Template & outline dokumentasi pemaparan proyek
├── report/
│   ├── Laporan Data Mining_A11.2024.16055.docx # Dokumen konsep produk & Jawaban UTS lengkap
│   ├── Laporan_Lengkap_UTS_A11.2024.16055.md   # Versi Markdown lengkap laporan UTS
│   ├── paper.pdf                               # Luaran artikel ilmiah format jurnal/konferensi
│   ├── sample_referral_report.pdf              # Contoh cetakan PDF Resume Klinis & Surat Rujukan Pasien
│   └── figures/                                # 6 Grafik beresolusi tinggi (Kohort, ROC, Ablasi, ACRFS, PR/LC, SHAP)
├── app.py                                      # Dashboard Web Interaktif 6 Halaman (Streamlit) & Simulasi CLI
├── api_fastapi.py                              # Backend REST API Modular (FastAPI / Starlette ASGI)
├── log.md                                      # Catatan riwayat eksperimen & revisi fitur
├── generate_notebook.py                        # Skrip otomatisasi end-to-end dataset, modeling, XAI, & notebook
└── generate_reports.py                         # Skrip otomatisasi pembuatan laporan DOCX & paper PDF
```

---

## Cara Menjalankan Pipeline, Dashboard, & REST API

1. **Meregenerasi Dataset, Melatih 6 Model + Dual-Engine, & Memperbarui Notebook:**
   ```bash
   python generate_notebook.py
   ```
2. **Meregenerasi Dokumen Laporan Word (`.docx`) & Artikel Ilmiah (`paper.pdf`):**
   ```bash
   python generate_reports.py
   ```
3. **Menjalankan Full-Featured Streamlit Web Dashboard (6 Halaman):**
   ```bash
   streamlit run app.py
   ```
4. **Menjalankan Simulasi Inferensi CLI & Ekspor PDF Rujukan:**
   ```bash
   python app.py --cli-demo
   ```
5. **Menjalankan Server REST API (`api_fastapi.py`) atau Self-Test Endpoint:**
   ```bash
   python api_fastapi.py --self-test
   uvicorn api_fastapi:app --host 0.0.0.0 --port 8000
   ```
