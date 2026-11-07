# Catatan Riwayat Eksperimen & Revisi Fitur — MATERNA-AI
**Peneliti:** Nayyara Rauf Ghani Al Khayri (`A11.2024.16055`)  
**Proyek:** Sistem Pendukung Keputusan Klinis Bertingkat untuk Prediksi Dini Preeklamsia (*Multi-Tier CDSS*)

---

## Riwayat Iterasi Eksperimen

| Versi | Tanggal | Fokus Eksperimen & Modul Fitur | Perubahan Utama | Hasil Kunci (Recall / ROC-AUC) |
| :--- | :--- | :--- | :--- | :--- |
| **v1.0** | 2026-10-18 | **Baseline Fitur A (MHS 1)**: Skrining Dasar Faskes Primer (*Tier 1*) | Menggunakan 15 fitur anamnesis, antropometri (LILA, IMT), dan rekayasa fitur $\text{MAP} = \frac{\text{SBP} + 2\cdot\text{DBP}}{3}$. | Recall: **90.28%** \| ROC-AUC: **0.9601** |
| **v1.5** | 2026-10-24 | **Integrasi Fitur B (MHS 2)**: *Longitudinal Trajectory Tracking* | Menambahkan 7 fitur dinamika antar-kunjungan ANC ($\Delta\text{SBP}$, $\Delta\text{DBP}$, $\Delta\text{MAP}$, laju kenaikan BB/minggu, progresivitas edema). | Recall: **93.06%** \| ROC-AUC: **0.9910** |
| **v2.5** | 2026-10-31 | **Full Multi-Tier CDSS Upgrade (SMOTE-NC, Ensemble XGBoost/LightGBM, ACRFS, XAI SHAP/LIME, PDF & FastAPI)** | Integrasi Fitur C (*Tier 2 Biomarker*), oversampling `SMOTE-NC` pada training fold, perluasan ensemble (`XGBoost`, `LightGBM`), XAI SHAP Waterfall, POGI Rule Engine, dan Generator PDF Rujukan. | Best Model (**Naive Bayes**): Recall **98.61%**, Specificity **99.40%**, ROC-AUC **0.9993** |

---

## Ringkasan Komparasi 6 Model Klasifikasi (4 Standar + 2 Ensemble dengan SMOTE-NC)

| Model | Kategori | File Artefak | Test Accuracy | Recall (Sensitivitas) | Specificity | F1-Score | ROC-AUC | Status Target PRD |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree** | Standar (4 Model Inti) | `models/decision_tree_model.joblib` | 92.08% | **90.28%** | 92.86% | 87.25% | **0.9709** | TERCAPAI (>=88%) |
| **KNN** | Standar (4 Model Inti) | `models/knn_model.joblib` | 98.75% | **97.22%** | 99.40% | 97.90% | **0.9990** | TERCAPAI (>=88%) |
| **Naive Bayes** | Standar (4 Model Inti) | `models/naive_bayes_model.joblib` | 99.17% | **98.61%** | 99.40% | 98.61% | **0.9993** | TERCAPAI (>=88%) |
| **SVM** | Standar (4 Model Inti) | `models/svm_model.joblib` | 99.17% | **97.22%** | 100.00% | 98.59% | **0.9994** | TERCAPAI (>=88%) |
| **XGBoost** | Perluasan Ensemble | `models/xgboost_model.joblib` | 99.17% | **97.22%** | 100.00% | 98.59% | **0.9996** | TERCAPAI (>=88%) |
| **LightGBM** | Perluasan Ensemble | `models/lightgbm_model.joblib` | 98.75% | **95.83%** | 100.00% | 97.87% | **0.9998** | TERCAPAI (>=88%) |
