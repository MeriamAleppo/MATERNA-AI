# DOKUMEN KONSEP PRODUK & LAPORAN UJIAN TENGAH SEMESTER (UTS) DATA MINING


### MATERNA-AI: Multi-Tier Clinical Decision Support System for Early Preeclampsia Prediction & Risk Trajectory

Nama Mahasiswa : Nayyara Rauf Ghani Al Khayri

NIM            : A11.2024.16055

Mata Kuliah    : Data Mining (UTS - Tahun Akademik 2026)

Program Studi  : Teknik Informatika - Fakultas Ilmu Komputer


---

## BAGIAN I: STRUKTUR DOKUMEN IDE / KONSEP PRODUK & PEMBAGIAN FITUR (KETENTUAN 1)


### 1.1. Judul Produk dan Deskripsi Umum

Judul Produk: MATERNA-AI — Sistem Pendukung Keputusan Klinis Bertingkat (Multi-Tier CDSS) untuk Prediksi Dini Preeklamsia, Pemantauan Trajektori Tanda Vital Maternal, dan Stratifikasi Subtipe Risiko.

Deskripsi Umum: MATERNA-AI adalah produk HealthTech berbasis Data Mining dan Machine Learning yang dirancang untuk mengatasi kesenjangan deteksi dini preeklamsia antara Fasilitas Kesehatan Tingkat Pertama (Puskesmas Pembantu, Posyandu, Bidan Praktik Mandiri) yang minim alat laboratorium dengan Rumah Sakit Rujukan (RSUD/Sp.OG). Berbeda dari model klasifikasi konvensional yang hanya memprediksi label biner statis pada satu titik waktu, MATERNA-AI mengintegrasikan tiga inovasi utama: (1) Tiered Clinical Screening (adaptif terhadap ketersediaan data lab), (2) Longitudinal Trajectory Tracking (memantau kecepatan kenaikan tekanan darah dan berat badan antar-kunjungan Antenatal Care / ANC), serta (3) Subtype Stratification (membedakan Early-Onset Preeclampsia <34 minggu yang kritis terhadap mortalitas ibu-janin dengan Late-Onset Preeclampsia >=34 minggu) disertai rekomendasi klinis berbasis pedoman POGI.


### 1.2. Personil Terlibat dan Pembagian Fitur Hasil Brainstorming

Sesuai tata cara pengerjaan proyek UTS Data Mining, tim melakukan brainstorming untuk membagi arsitektur produk MATERNA-AI menjadi 3 Fitur Inti yang saling melengkapi, namun dieksekusi dan dilaporkan secara mandiri (individu) dengan mengacu pada Kriteria Penilaian Soal UTS (Poin 1 s.d. 3):


| Kode Fitur | Penanggung Jawab (Personil) | Nama Modul / Fitur Produk | Fokus Utama pada Soal UTS (Kriteria 1 - 3) |
| --- | --- | --- | --- |
| Fitur A (Utama/Individu) | MHS 1: Nayyara Rauf Ghani Al Khayri (A11.2024.16055) | Tiered Clinical Screening Engine (Mode Faskes Primer Tier 1 vs Mode Rujukan Tier 2) | Prediksi risiko preeklamsia berbasis fitur klinis dasar (MAP, LILA, IMT, Anamnesis Obstetrik) & integrasi biomarker, disertai optimasi 4 model utama (Decision Tree, KNN, Naive Bayes, SVM). |
| Fitur B (Kolaborasi Brainstorming) | MHS 2: Personil Brainstorming 2 (Fokus Longitudinal) | Longitudinal Maternal Vital Trajectory & Velocity Tracker | Rekayasa fitur perubahan dinamis antar-kunjungan ANC (Delta SBP, Delta DBP, Delta MAP, Laju Kenaikan BB/minggu, Progresivitas Edema) untuk mendeteksi lonjakan hipertensi gestasional laten. |
| Fitur C (Kolaborasi Brainstorming) | MHS 3: Personil Brainstorming 3 (Fokus Subtipe & CDSS) | Subtype Stratification (Early-Onset vs Late-Onset) & Explainable POGI Rule Matching | Klasifikasi multikelas subtipe keparahan (<34 minggu vs >=34 minggu) berdasarkan biomarker ginjal/hepar/Doppler UtAPI serta pemetaan rekomendasi profilaksis aspirin & rujukan POGI. |


### 1.3. Pembahasan Ringkas Tiap Personil / Fitur Mengacu Soal UTS

- Pembahasan Fitur A (MHS 1 - Nayyara Rauf Ghani Al Khayri): Menjawab permasalahan keterlambatan skrining di Puskesmas akibat ketiadaan alat lab. Menggunakan rekayasa fitur Mean Arterial Pressure (MAP = [SBP + 2*DBP]/3), Pulse Pressure, LILA, IMT Pra-Hamil, dan riwayat obstetrik untuk menghasilkan sensitivitas penapisan >=88%.
- Pembahasan Fitur B (MHS 2 - Fokus Trajectory): Menjawab kelemahan pemeriksaan one-shot (satu waktu). Banyak ibu hamil memiliki tekanan darah 125/80 mmHg yang tampak 'normal', namun sebenarnya mengalami lonjakan +25 mmHg dari Trimester 1 (Delta MAP tinggi) disertai kenaikan berat badan >0.8 kg/minggu akibat retensi cairan endotel.
- Pembahasan Fitur C (MHS 3 - Fokus Subtype & Biomarker): Menjawab kebutuhan dokter Sp.OG di RS Rujukan untuk membedakan Early-Onset Preeclampsia (<34 minggu) yang berisiko tinggi memicu sindrom HELLP dan IUGR menggunakan kombinasi rasio protein/kreatinin urin (PCR), trombosit, enzim hati (SGOT/SGPT), asam urat, dan Uterine Artery Pulsatility Index (UtAPI).

---

## BAGIAN II: JAWABAN SOAL UJIAN UTS DATA MINING (KRITERIA PENILAIAN 1 - 3)


### SOAL 1. Identifikasi Masalah dan Tujuan Penelitian (15 Poin)


#### 1.a. Topik dan Judul Penelitian

Topik Penelitian : Data Mining Kesehatan Maternal / Clinical Decision Support System (CDSS) untuk Deteksi Dini Komplikasi Kehamilan.

Judul Penelitian : "Rancang Bangun Sistem Pendukung Keputusan Klinis Bertingkat (MATERNA-AI) Berbasis Optimasi Algoritma Klasifikasi Data Mining dan Trajektori Longitudinal untuk Prediksi Dini serta Stratifikasi Risiko Preeklamsia".


#### 1.b. Latar Belakang Permasalahan

Preeklamsia merupakan sindrom hipertensi multisistem spesifik kehamilan yang muncul setelah usia gestasi 20 minggu, ditandai dengan hipertensi baru (SBP >= 140 mmHg atau DBP >= 90 mmHg) disertai proteinuria atau disfungsi organ target (trombositopenia, gangguan fungsi ginjal, peningkatan transaminase hati, atau gangguan sirkulasi uteroplasenta). Berdasarkan laporan Kementerian Kesehatan RI dan WHO, hipertensi dalam kehamilan dan preeklamsia/eklamsia merupakan salah satu dari dua penyebab utama Angka Kematian Ibu (AKI) tertinggi di Indonesia (menyumbang sekitar 24%–30% kematian maternal) serta pemicu utama kelahiran prematur iatrogenik dan pertumbuhan janin terhambat (IUGR).

Secara praktis di lapangan, terdapat tiga permasalahan fundamental dalam penapisan preeklamsia saat ini: Pertama, kesenjangan fasilitas diagnostik (Resource Gap), di mana model prediksi medis modern sangat bergantung pada biomarker angiogenik mahal (seperti rasio sFlt-1/PlGF) atau USG Doppler Arteri Uterina yang tidak tersedia di Puskesmas Pembantu atau Posyandu. Kedua, pendekatan pemeriksaan yang bersifat statis satu titik waktu (Single-Timepoint Assessment), padahal perubahan fisiologis preeklamsia tercermin dari kemiringan lonjakan (trajectory/velocity) tekanan darah arteri rata-rata (MAP) dan retensi cairan (kenaikan berat badan mingguan) sejak Trimester 1 ke Trimester 2. Ketiga, belum adanya pembedaan otomatis antara Early-Onset Preeclampsia (<34 minggu) dan Late-Onset Preeclampsia (>=34 minggu) yang memiliki patofisiologi dan tingkat kedaruratan rujukan yang sangat berbeda.


#### 1.c. Urgensi Penelitian

Penelitian ini memiliki urgensi klinis yang sangat tinggi karena jendela intervensi pencegahan preeklamsia sangat sempit. Pemberian profilaksis aspirin dosis rendah (75–150 mg/hari) dan suplementasi kalsium sesuai pedoman Perkumpulan Obstetri dan Ginekologi Indonesia (POGI) paling efektif apabila risiko tinggi terdeteksi sejak dini sebelum kerusakan plasenta bersifat ireversibel. Kegagalan mendeteksi kasus positif (False Negative) di tingkat Puskesmas berakibat fatal karena pasien baru dirujuk saat sudah mengalami kejang eklamsia atau sindrom HELLP. Sebaliknya, rujukan membabi buta (False Positive tinggi) membebani kapasitas RSUD. Oleh sebab itu, diperlukan model Data Mining yang mengoptimalkan keseimbangan antara Sensitivitas/Recall (>= 88%) dan Spesifisitas (>= 80%).


#### 1.d. Rumusan Masalah

- Bagaimana merancang skema fitur klinis bertingkat (Tier 1 Faskes Primer dan Tier 2 Biomarker Faskes Rujukan) serta fitur trajektori longitudinal (Delta MAP dan laju kenaikan berat badan ANC) yang efektif merepresentasikan patofisiologi preeklamsia?
- Bagaimana perbandingan kinerja algoritma klasifikasi Data Mining (Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes, dan Support Vector Machine) setelah melalui tahapan Data Preprocessing dan Hyperparameter Optimization dalam memprediksi risiko preeklamsia?
- Seberapa besar peningkatan performa prediksi (Recall, Specificity, dan ROC-AUC) yang dihasilkan dari kolaborasi penggabungan Fitur A (Klinis Dasar), Fitur B (Longitudinal Trajectory), dan Fitur C (Biomarker & Subtype Stratification)?

#### 1.e. Tujuan Penelitian

- Membangun pipeline Data Preprocessing dan Feature Engineering klinis yang bersih, bebas kebocoran data (data leakage), serta mampu menghitung indikator hemodinamik (MAP, Pulse Pressure, Delta MAP) secara otomatis.
- Melatih, mengevaluasi, dan mengoptimasi empat model klasifikasi utama (Decision Tree, KNN, Naive Bayes, dan SVM) menggunakan Stratified 5-Fold Cross-Validation hingga melampaui target klinis PRD (Recall >= 88%, Specificity >= 80%, dan ROC-AUC >= 0,88).
- Mengukur kontribusi masing-masing modul fitur hasil brainstorming (Fitur A, Fitur B, dan Fitur C) melalui studi ablasi serta mengimplementasikan prototipe aplikasi CDSS siap pakai.

#### 1.f. Manfaat Penelitian

- Manfaat bagi Tenaga Medis Faskes Primer (Bidan/Perawat Puskesmas): Menyediakan alat bantu penapisan risiko dini yang akurat hanya bermodalkan tensimeter, pita LILA, timbangan, dan anamnesis terstruktur.
- Manfaat bagi Dokter Umum & Sp.OG di RS Rujukan: Memberikan probabilitas risiko terkalibrasi, deteksi potensi Early-Onset Preeclampsia, dan rekomendasi tindakan klinis berbasis pedoman POGI.
- Manfaat Akademis / Keilmuan Data Mining: Menjadi referensi empiris penerapan multi-tier feature engineering dan optimasi algoritma klasifikasi pada domain data medis tabular yang memiliki ketidakseimbangan kelas (class imbalance).

### SOAL 2. Studi Literatur dan Kebaruan Penelitian (15 Poin)


#### 2.a. Telaah Penelitian Terdahulu yang Relevan (Minimal 3 Penelitian)

Berikut adalah telaah komprehensif terhadap lima penelitian terdahulu bereputasi internasional yang menjadi landasan pengembangan sistem MATERNA-AI:


| No | Peneliti & Tahun | Judul & Publikasi | Metode / Algoritma | Hasil Utama & Keterbatasan |
| --- | --- | --- | --- | --- |
| 1 | Jhee et al. (2019) | Prediction model development of late-onset preeclampsia using machine learning-based methods (Yonsei Medical Journal / PLOS ONE) | Logistic Regression, Decision Tree, Naive Bayes, SVM, Random Forest, Stochastic Gradient Boosting pada data EMR 11.006 ibu hamil. | Stochastic Gradient Boosting & SVM mencapai AUC 0,86–0,92. Keterbatasan: Hanya berfokus pada Late-Onset Preeclampsia dan membutuhkan variabel biokimia lab rutin rumah sakit besar. |
| 2 | Marić et al. (2020) | Early prediction of preeclampsia via machine learning (Science Advances, Vol. 6, No. 45) | Elastic Net & Gradient Boosting pada 66 variabel klinis dan laboratorium kohort prospektif. | Mencapai ROC-AUC 0,79 (semua kasus) dan 0,89 (Early-Onset). Keterbatasan: Menggunakan puluhan variabel rekam medis elektronik kompleks yang tidak praktis diinput di faskes primer negara berkembang. |
| 3 | Poon et al. / FMF Group (2019) | The Fetal Medicine Foundation (FMF) competing-risks model for preeclampsia screening (AJOG) | Competing-Risks Survival Model berbasis MAP, Uterine Artery PI (UtAPI), PAPP-A, dan PlGF. | Deteksi ~90% Early-Onset PE dengan False Positive Rate 10%. Keterbatasan: Sangat bergantung pada reagen PlGF dan operator USG Doppler tersertifikasi yang mahal. |
| 4 | Li et al. (2022) | Machine learning models for predicting preeclampsia: a systematic review and cohort validation (BMC Pregnancy and Childbirth) | Komparasi SVM, KNN, Random Forest, dan XGBoost pada data klinis antenatal. | Model non-linear (SVM RBF & Ensemble) mengungguli regresi konvensional (AUC 0,84–0,91). Keterbatasan: Hanya menggunakan pengukuran statis satu titik waktu tanpa variabel delta longitudinal. |
| 5 | Ahmed et al. / UCI Maternal Health Study (2020) | Review and Analysis of Risk Factor of Maternal Health in Remote Area Using the Internet of Things (IoT) & ML (Lecture Notes in Electrical Engineering) | Decision Tree, KNN, Random Forest pada tanda vital dasar maternal (SBP, DBP, BS, BodyTemp, HeartRate). | Menunjukkan kelayakan skrining komunitas di daerah terpencil. Keterbatasan: Tidak menyertakan atribut spesifik preeklamsia seperti MAP, LILA, paritas, proteinuria, maupun stratifikasi onset. |


#### 2.b. Identifikasi Keterbatasan atau Celah Penelitian (Research Gap)

Berdasarkan matriks telaah literatur di atas, teridentifikasi tiga celah penelitian (research gap) utama yang belum diselesaikan secara simultan oleh studi terdahulu:

- Celah Aksesibilitas Fasilitas (Single-Tier Dependency Gap): Penelitian berakurasi tinggi seperti Poon et al. (2019) dan Marić et al. (2020) mengasumsikan seluruh variabel laboratorium dan Doppler selalu tersedia. Sebaliknya, model faskes dasar (Ahmed et al., 2020) terlalu sederhana sehingga kurang spesifik untuk preeklamsia. Belum ada arsitektur Multi-Tier yang menjembatani keduanya.
- Celah Dinamika Temporal (Static Single-Visit Gap): Sebagian besar studi (Jhee et al., 2019; Li et al., 2022) memperlakukan tekanan darah sebagai angka statis pada satu kunjungan, mengabaikan fakta klinis bahwa lonjakan tekanan darah antar-trimester (Delta MAP) dan laju kenaikan berat badan per minggu merupakan sinyal prodromal disfungsi endotel.
- Celah Stratifikasi & Aksi Klinis (Binary-Only Output Gap): Mayoritas model hanya mengeluarkan label biner (0/1) tanpa membedakan subtipe Early-Onset (<34 minggu) vs Late-Onset (>=34 minggu) serta belum memetakan probabilitas model ke dalam panduan rujukan klinis nasional (POGI).

#### 2.c. Kontribusi dan Kebaruan Penelitian (Novelty)

- Kontribusi 1 — Kerangka Kerja Fitur Multi-Tier & Longitudinal (Fitur A + B + C): Menggabungkan 15 fitur klinis dasar Faskes Primer (termasuk MAP, Pulse Pressure, LILA, IMT), 7 fitur trajektori longitudinal ANC (Delta SBP, Delta DBP, Delta MAP, laju kenaikan BB/minggu), dan 8 fitur biomarker laboratorium Tier 2.
- Kontribusi 2 — Optimasi Komprehensif 4 Algoritma Klasifikasi Berorientasi Sensitivitas Klinis: Melakukan benchmark terstandarisasi dan pencarian hiperparameter optimal (GridSearchCV Stratified 5-Fold) pada Decision Tree, KNN, Naive Bayes, dan SVM dengan prioritas meminimalkan False Negative (Recall >= 88%).
- Kontribusi 3 — Kemampuan Stratifikasi Subtipe & Integrasi Aturan Klinis POGI: Menyediakan pemetaan otomatis dari skor probabilitas dan subtipe onset menuju rekomendasi klinis yang dapat langsung ditindaklanjuti oleh bidan maupun dokter.

### SOAL 3. Perancangan Dataset dan Atribut Penelitian (20 Poin)


#### 3.a. Sumber, Cara Memperoleh Dataset, dan Jumlah Data

Dataset yang digunakan dalam proyek ini disimpan pada file `data/preeclampsia_clinical_dataset.csv` dengan jumlah data bersih sebanyak N = 1.200 sampel rekam medis pasien ibu hamil (dari 1.212 baris data mentah sebelum deduplikasi). Dataset ini dirancang melalui pendekatan Clinically-Calibrated Synthetic & Empirical Cohort Generation yang mengacu langsung pada parameter distribusi statistik dari tiga sumber rujukan:

- 1. Distribusi tanda vital dasar (tekanan darah sistolik, diastolik, usia ibu) dari UCI Machine Learning Repository: Maternal Health Risk Data Set (Ahmed et al., 2020).
- 2. Ambang batas diagnostik dan faktor risiko obstetrik dari Pedoman Nasional Pelayanan Kedokteran (PNPK) Diagnosis dan Tata Laksana Preeklamsia oleh POGI serta kriteria ACOG (American College of Obstetricians and Gynecologists).
- 3. Distribusi biomarker laboratorium (proteinuria, PCR urin, trombosit, kreatinin serum, SGOT/SGPT, asam urat, dan UtAPI Doppler) dari literatur kohort klinis preeklamsia (Jhee et al., 2019; Poon et al., 2019).
Komposisi kelas dirancang mencerminkan proporsi klinis rumah sakit rujukan (imbalanced dataset): 840 pasien (70,0%) Kehamilan Normotensif/Normal, 216 pasien (18,0%) Late-Onset Preeclampsia (>= 34 minggu), dan 144 pasien (12,0%) Early-Onset Preeclampsia (< 34 minggu), sehingga total kasus positif preeklamsia adalah 360 pasien (30,0%).


#### 3.b. Daftar Atribut / Fitur Penelitian beserta Penjelasannya

Dataset terdiri atas 35 kolom secara keseluruhan (1 ID pasien, 1 metadata fasilitas, 30 fitur prediktor yang terbagi ke dalam Fitur A, Fitur B, dan Fitur C, serta 3 kolom representasi variabel target):


| No | Nama Atribut | Kelompok Modul | Tipe & Satuan | Penjelasan Klinis & Rentang Nilai |
| --- | --- | --- | --- | --- |
| 1 | patient_id | Identitas | String (ID) | Kode unik rekam medis pasien (MAT-20260001 s.d. MAT-20261200); tidak dimasukkan ke matriks fitur model (de-identifikasi UU PDP). |
| 2 | faskes_origin | Metadata Faskes | Kategorikal | Asal fasilitas kesehatan pemeriksaan (Puskesmas_Pembantu, Puskesmas_Induk, RSUD_Rujukan). |
| 3 | gestational_age_weeks | Fitur A (Tier 1) | Numerik (Minggu) | Usia kehamilan saat skrining berlangsung (16 – 40 minggu). |
| 4 | maternal_age | Fitur A (Tier 1) | Numerik (Tahun) | Usia ibu hamil (16 – 46 tahun). Usia <20 atau >=35 tahun meningkatkan risiko vaskular. |
| 5 | gravida | Fitur A (Tier 1) | Numerik (Frekuensi) | Total jumlah kehamilan yang pernah dialami ibu termasuk kehamilan saat ini (1 – 7). |
| 6 | parity | Fitur A (Tier 1) | Numerik (Frekuensi) | Jumlah persalinan janin hidup/viabel sebelumnya (0 – 6). |
| 7 | nullipara | Fitur A (Tier 1) | Biner (0/1) | Status belum pernah melahirkan sebelumnya (1 = Ya/Nullipara, 0 = Multipara). Faktor risiko imunologis plasenta. |
| 8 | preg_interval_yrs | Fitur A (Tier 1) | Numerik (Tahun) | Jarak waktu dari persalinan terakhir (0,0 untuk nullipara; >5 tahun meningkatkan risiko preeklamsia). |
| 9 | prev_pe_history | Fitur A (Tier 1) | Biner (0/1) | Riwayat mengalami preeklamsia pada kehamilan sebelumnya (0 = Tidak, 1 = Ya). |
| 10 | family_htn_history | Fitur A (Tier 1) | Biner (0/1) | Riwayat hipertensi atau preeklamsia pada ibu/saudara kandung (0 = Tidak, 1 = Ya). |
| 11 | gest_diabetes | Fitur A (Tier 1) | Biner (0/1) | Status komorbid diabetes melitus gestasional (0 = Tidak, 1 = Ya). |
| 12 | pre_pregnancy_bmi | Fitur A (Tier 1) | Numerik (kg/m2) | Indeks Massa Tubuh (IMT) pra-kehamilan (16,5 – 42,0 kg/m2). Obesitas (>=25–30) memicu resistensi insulin & inflamasi endotel. |
| 13 | lila_cm | Fitur A (Tier 1) | Numerik (cm) | Lingkar Lengan Atas (19,5 – 39,0 cm), indikator antropometri status gizi & metabolik di Posyandu/Puskesmas. |
| 14 | systolic_bp | Fitur A (Tier 1) | Numerik (mmHg) | Tekanan darah sistolik kunjungan saat ini (90 – 200 mmHg). Cut-off hipertensi >= 140 mmHg. |
| 15 | diastolic_bp | Fitur A (Tier 1) | Numerik (mmHg) | Tekanan darah diastolik kunjungan saat ini (58 – 130 mmHg). Cut-off hipertensi >= 90 mmHg. |
| 16 | map_mmhg | Fitur A (Tier 1) | Numerik (mmHg) | Mean Arterial Pressure = (SBP + 2*DBP)/3. Indikator perfusi organ & resistensi vaskular (kritis bila >95–106 mmHg). |
| 17 | pulse_pressure | Fitur A (Tier 1) | Numerik (mmHg) | Selisih tekanan nadi (SBP - DBP), merefleksikan kekakuan arteri maternal. |
| 18 | sbp_trimester1 | Fitur B (Trajectory) | Numerik (mmHg) | Tekanan darah sistolik dasar (baseline) pada kunjungan ANC Trimester 1. |
| 19 | dbp_trimester1 | Fitur B (Trajectory) | Numerik (mmHg) | Tekanan darah diastolik dasar (baseline) pada kunjungan ANC Trimester 1. |
| 20 | delta_sbp | Fitur B (Trajectory) | Numerik (mmHg) | Lonjakan tekanan sistolik dari Trimester 1 ke kunjungan saat ini (SBP_kini - SBP_T1). |
| 21 | delta_dbp | Fitur B (Trajectory) | Numerik (mmHg) | Lonjakan tekanan diastolik dari Trimester 1 ke kunjungan saat ini (DBP_kini - DBP_T1). |
| 22 | delta_map | Fitur B (Trajectory) | Numerik (mmHg) | Perubahan Mean Arterial Pressure antar-kunjungan ANC (kenaikan >=15 mmHg sangat prediktif terhadap preeklamsia). |
| 23 | weight_gain_kg_per_week | Fitur B (Trajectory) | Numerik (kg/mgg) | Rata-rata kenaikan berat badan ibu per minggu pada trimester berjalan (>0,75 kg/minggu mengindikasikan retensi cairan ekstravaskular). |
| 24 | edema_progression | Fitur B (Trajectory) | Ordinal (0, 1, 2) | Skor progresivitas pembengkakan/edema klinis (0 = Tidak ada, 1 = Edema tungkai ringan, 2 = Edema anasarka/wajah/tangan). |
| 25 | proteinuria_dipstick | Fitur C (Tier 2 Lab) | Kategorikal Ordinal | Hasil pemeriksaan carik celup urin kualitatif ('Negatif', '+1', '+2', '+3') yang dikonversi ke `proteinuria_ordinal` (0–3). |
| 26 | urine_pcr | Fitur C (Tier 2 Lab) | Numerik (mg/mg) | Rasio Protein/Kreatinin Urin kuantitatif (abnormal jika >= 0,30 mg/mg). |
| 27 | platelet_count_k_ul | Fitur C (Tier 2 Lab) | Numerik (10^3/uL) | Hitung trombosit darah (65 – 390 x 10^3/uL). Trombositopenia (<150 atau <100) menandakan koagulopati mikroangiopati/HELLP. |
| 28 | serum_creatinine_mg_dl | Fitur C (Tier 2 Lab) | Numerik (mg/dL) | Kadar kreatinin serum untuk memantau filtrasi glomerulus ginjal (>1,1 mg/dL menunjukkan insufisiensi ginjal). |
| 29 | sgot_ast_u_l | Fitur C (Tier 2 Lab) | Numerik (U/L) | Enzim hati Aspartate Aminotransferase (peningkatan >2x lipat normal menandakan keterlibatan kapsul hepar). |
| 30 | sgpt_alt_u_l | Fitur C (Tier 2 Lab) | Numerik (U/L) | Enzim hati Alanine Aminotransferase (10 – 150 U/L). |
| 31 | uric_acid_mg_dl | Fitur C (Tier 2 Lab) | Numerik (mg/dL) | Kadar asam urat darah (2,4 – 10,2 mg/dL). Hiperurisemia (>5,5 mg/dL) mendahului proteinuria pada banyak kasus preeklamsia. |
| 32 | utapi_doppler | Fitur C (Tier 2 USG) | Numerik (Indeks) | Uterine Artery Pulsatility Index dari USG Doppler (0,48 – 2,50). Nilai tinggi (>1,4) menandakan kegagalan invasi trofoblas arteri spiralis. |
| 33 | preeclampsia_diagnosis | Variabel Target Utama | Biner (0 / 1) | Label klasifikasi supervised learning utama: 0 = Normal/Non-Preeklamsia (70%), 1 = Positif Preeklamsia (30%). |
| 34 | preeclampsia_subtype | Variabel Target Subtipe | Multikelas (3 Kelas) | Label stratifikasi subtipe: 'Normal' (0), 'Late_Onset_PE' >=34 minggu (1), 'Early_Onset_PE' <34 minggu (2). |


#### 3.c. Variabel Target / Keluaran (Pendekatan Supervised Learning)

Penelitian ini menggunakan pendekatan Supervised Learning (Classification) dengan dua skema keluaran (dual-target formulation):

- 1. Target Klasifikasi Biner Utama (`preeclampsia_diagnosis`): Bernilai `0` untuk kehamilan normotensif/normal dan `1` untuk kasus preeklamsia. Variabel ini digunakan untuk melatih dan mengevaluasi keempat model wajib (`decision_tree_model.joblib`, `knn_model.joblib`, `naive_bayes_model.joblib`, dan `svm_model.joblib`). Selain menghasilkan prediksi kelas diskrit, model mengeluarkan Calibrated Probability $P(Y=1|X) \in [0, 1]$ yang dikelompokkan menjadi Risiko Rendah (<15%), Risiko Sedang (15%–40%), dan Risiko Tinggi (>40%).
- 2. Target Klasifikasi Multikelas Subtipe (`preeclampsia_subtype_code`): Bernilai `0` (Normal), `1` (Late-Onset Preeclampsia >=34 minggu), dan `2` (Early-Onset Preeclampsia <34 minggu) untuk mendukung keputusan rujukan vertikal di RSUD.

#### 3.d. Alasan Pemilihan Atribut dan Relevansinya terhadap Permasalahan Penelitian

Setiap atribut dipilih berdasarkan mekanisme patofisiologi preeklamsia dua-tahap (Two-Stage Placental & Maternal Endothelial Model):

- Relevansi Fitur Hemodinamik & Antropometri (Fitur A): Mean Arterial Pressure (MAP) terbukti dalam berbagai uji klinis memiliki daya diskriminasi lebih kuat daripada tekanan sistolik atau diastolik tunggal karena mengintegrasikan resistensi vaskular sistemik. Nullipara, riwayat preeklamsia terdahulu, IMT pra-hamil tinggi, dan LILA besar merepresentasikan kerentanan imunologis maladaptasi plasenta serta sindrom metabolik.
- Relevansi Fitur Longitudinal (Fitur B): Kenaikan tajam tekanan darah dibandingkan baseline Trimester 1 (`delta_map`) dan lonjakan berat badan mingguan (`weight_gain_kg_per_week`) akibat kebocoran kapiler (edema) memungkinkan sistem menangkap pasien yang tekanan darah absolutnya masih di ambang perbatasan (borderline 135/85 mmHg) namun memiliki laju perburukan cepat.
- Relevansi Fitur Biomarker & Doppler (Fitur C): Kerusakan endotel glomerulus menyebabkan kebocoran protein ke urin (`proteinuria_dipstick`, `urine_pcr`), penurunan klirens asam urat (`uric_acid_mg_dl`) dan kreatinin (`serum_creatinine_mg_dl`). Aktivasi koagulasi intravaskular mengonsumsi keping darah (`platelet_count_k_ul`), sementara nekrosis hepatoseluler meningkatkan `sgot_ast_u_l` dan `sgpt_alt_u_l`. Adapun `utapi_doppler` mengukur langsung impedansi aliran darah arteri uterina.

#### 3.e. Strategi Penanganan Data Kosong, Data Duplikat, Data Tidak Konsisten, dan Atribut Kategorikal

Pada pipeline eksperimen (`generate_notebook.py`), audit kualitas data terhadap 1212 baris data mentah menemukan 12 baris duplikat dan nilai kosong pada 4 fitur klinik/lab. Berikut adalah strategi sistematis yang diterapkan:

- 1. Penanganan Data Duplikat: Dilakukan pemeriksaan kunci primer `patient_id` dan kombinasi fitur klinis. Sebanyak 12 baris duplikat dihapus menggunakan `df.drop_duplicates(subset=['patient_id'], keep='first')`, menyisakan tepat 1200 rekam medis unik.
- 2. Penanganan Data Kosong (Missing Values): Untuk menghindari bias penghapusan baris (listwise deletion) yang dapat membuang kasus preeklamsia langka, digunakan Median Imputation (`SimpleImputer(strategy='median')` di dalam Pipeline pelatihan) untuk fitur numerik yang rentan tidak terisi di lapangan (`lila_cm`, `urine_pcr`, `uric_acid_mg_dl`, `utapi_doppler`). Penggunaan median dipilih karena tahan (robust) terhadap kemencengan distribusi (skewness) nilai laboratorium.
- 3. Penanganan Data Tidak Konsisten & Outlier: Diterapkan validasi aturan fisiologis otomatis (memastikan `systolic_bp > diastolic_bp` serta menghitung ulang `map_mmhg = (systolic_bp + 2*diastolic_bp)/3` dan `pulse_pressure = systolic_bp - diastolic_bp`). Untuk nilai ekstrem pada biomarker hati/trombosit, tidak dilakukan penghapusan outlier secara membabi buta karena nilai ekstrem justru merupakan ciri klinis preeklamsia berat, melainkan dibatasi pada batas fisiologis (physiological capping / Winsorizing persentil 1%–99%).
- 4. Penanganan Atribut Kategorikal & Normalisasi Skala: Atribut `proteinuria_dipstick` memiliki urutan tingkat keparahan alami sehingga ditransformasi menggunakan Ordinal Encoding (`Negatif` -> 0, `+1` -> 1, `+2` -> 2, `+3` -> 3). Seluruh fitur numerik kemudian dinormalisasi menggunakan `StandardScaler` (Z-score standardization) di dalam `sklearn.pipeline.Pipeline` agar algoritma berbasis jarak dan margin seperti KNN dan SVM tidak didominasi oleh fitur berskala ratusan (seperti jumlah trombosit) dibandingkan fitur berskala satuan (seperti kreatinin serum).

#### 3.f. Penjelasan Khusus Dataset Simulasi / Sintetis Terkalibrasi Klinis

- Cara Pembentukan Data: Dibentuk menggunakan generator probabilistik multivariat terstruktur (`generate_calibrated_preeclampsia_dataset`) dengan random seed tetap (`seed=42`) untuk menjamin reprodusibilitas 100%. Setiap kelas subtipe (Normal, Late-Onset, Early-Onset) dimodelkan menggunakan distribusi Gaussian terpotong (Truncated Normal) dan distribusi Bernoulli/Multinomial bersyarat yang menjaga korelasi antar-fitur (misalnya korelasi positif antara SBP, DBP, MAP, proteinuria, dan asam urat, serta korelasi negatif terhadap jumlah trombosit).
- Dasar Penentuan Nilai: Seluruh parameter mean, standar deviasi, dan batas bawah-atas diselaraskan dengan ambang diagnostik PNPK Preeklamsia POGI dan ACOG (SBP >=140 mmHg, DBP >=90 mmHg, PCR urin >=0,3 mg/mg, trombositopenia <150.000/uL, dan UtAPI >1,4 pada Early-Onset).
- Keterbatasan Penggunaan: Meskipun telah dikalibrasi secara klinis dan statistik, dataset sintetis belum sepenuhnya menangkap artefak pencatatan manual atau variasi komorbiditas langka (seperti lupus nefritis atau hipertensi kronis esensial pra-hamil) pada populasi nyata. Oleh karena itu, sebelum diimplementasikan untuk keputusan medis langsung di rumah sakit, model wajib melewati tahap validasi eksternal prospektif menggunakan rekam medis lokal Puskesmas/RSUD.

---

## BAGIAN III: HASIL IMPLEMENTASI, OPTIMASI 4 MODEL, & STUDI ABLASI FITUR


### 3.1. Perbandingan Kinerja 4 Model Utama Hasil Hyperparameter Optimization

Keempat model klasifikasi dilatih pada data latih (80% = 960 sampel) menggunakan Stratified 5-Fold Cross-Validation (`GridSearchCV`) dan diuji pada data uji independen (20% = 240 sampel). Berikut adalah rangkuman performa lengkapnya:


| Model Klasifikasi | File Model (.joblib) | CV ROC-AUC | Test Accuracy | Recall (Sensitivitas) | Specificity | F1-Score | Test ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Decision Tree | models/decision_tree_model.joblib | 0.9536 | 92.08% | 90.28% | 92.86% | 87.25% | 0.9709 |
| KNN | models/knn_model.joblib | 0.9996 | 98.75% | 97.22% | 99.40% | 97.90% | 0.9990 |
| Naive Bayes | models/naive_bayes_model.joblib | 0.9994 | 99.17% | 98.61% | 99.40% | 98.61% | 0.9993 |
| SVM | models/svm_model.joblib | 0.9998 | 99.17% | 97.22% | 100.00% | 98.59% | 0.9994 |
| XGBoost | models/xgboost_model.joblib | 0.9991 | 99.17% | 97.22% | 100.00% | 98.59% | 0.9996 |
| LightGBM | models/lightgbm_model.joblib | 0.9994 | 98.75% | 95.83% | 100.00% | 97.87% | 0.9998 |

Berdasarkan kriteria seleksi klinis terkombinasi (mengutamakan Recall >= 88%, Specificity >= 80%, dan ROC-AUC >= 0,88 sesuai PRD), model terbaik yang terpilih dan disimpan ke `models/best_model.joblib` adalah Naive Bayes.


### 3.2. Hasil Studi Ablasi Kolaborasi Fitur (MHS 1 vs MHS 2 vs MHS 3)

Untuk membuktikan nilai tambah dari pembagian fitur hasil brainstorming, dilakukan pengujian bertahap dari Fitur A (Tier 1 Faskes Primer), dilanjutkan penambahan Fitur B (Longitudinal Trajectory), hingga penggabungan penuh dengan Fitur C (Tier 2 Biomarker):


| Skenario Kombinasi Fitur | Jumlah Fitur | Test Accuracy | Recall (Sensitivitas) | Specificity | F1-Score | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Fitur A (MHS 1 - Tier 1 Faskes Primer) | 15 | 90.42% | 90.28% | 90.48% | 84.97% | 0.9601 |
| Fitur A + B (MHS 2 - +Longitudinal Trajectory) | 22 | 94.58% | 93.06% | 95.24% | 91.16% | 0.9910 |
| Fitur A + B + C (MHS 3 - Full Multi-Tier & Biomarker) | 30 | 99.17% | 97.22% | 100.00% | 98.59% | 0.9996 |

Selain klasifikasi biner, pengujian stratifikasi subtipe 3-kelas (Normal vs Late-Onset PE vs Early-Onset PE) berhasil mencapai Multiclass Accuracy sebesar 97.92% dan Macro F1-Score sebesar 96.64%.
