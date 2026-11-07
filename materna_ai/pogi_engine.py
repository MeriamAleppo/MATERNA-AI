# -*- coding: utf-8 -*-
"""
Modul 6: POGI Clinical Decision Rule-Matching Engine
Menerjemahkan probabilitas model Machine Learning & indikator hemodinamik/biomarker
ke rekomendasi klinis terstruktur berbasis Pedoman Nasional Pelayanan Kedokteran (PNPK)
Perkumpulan Obstetri dan Ginekologi Indonesia (POGI):
- Risiko Rendah (<15%): Jadwal ANC rutin FKTP & edukasi nutrisi.
- Risiko Sedang (15% - 40%): Evaluasi pemberian profilaksis Aspirin dosis rendah (80-150 mg/hari)
  sebelum usia 16 minggu, suplementasi kalsium (1.000-1.500 mg/hari), & monitoring tensi per 2 minggu.
- Risiko Tinggi (>40%): Protokol rujukan segera ke Sp.OG, skrining USG Doppler arteri uterina,
  dan rujukan persiapan persalinan di fasilitas emergensi obstetri (PONEK/RSUD).
"""

from typing import Dict, List, Any


def evaluate_pogi_decision_rules(
    prob_pe: float,
    patient: Dict[str, Any],
    subtype_pred: int = 0,
    screening_mode: str = "Tier 2",
) -> Dict[str, Any]:
    """
    Mencocokkan probabilitas prediksi model & profil klinis pasien dengan protokol POGI/PNPK.
    Mengembalikan struktur lengkap untuk Dashboard, API, dan Cetakan Surat Rujukan PDF.
    """
    gest_age = int(patient.get("gestational_age_weeks", 28))
    sbp = float(patient.get("systolic_bp", 120))
    dbp = float(patient.get("diastolic_bp", 78))
    map_val = float(patient.get("map_mmhg", round((sbp + 2 * dbp) / 3.0, 2)))
    delta_map = float(patient.get("delta_map", 0.0))
    wg_vel = float(patient.get("weight_gain_kg_per_week", 0.45))
    proteinuria_ord = int(patient.get("proteinuria_ordinal", 0))
    platelet = float(patient.get("platelet_count_k_ul", 240))
    creatinine = float(patient.get("serum_creatinine_mg_dl", 0.72))
    sgot = float(patient.get("sgot_ast_u_l", 22))
    utapi = float(patient.get("utapi_doppler", 0.90))

    # Deteksi tanda bahaya organ target (Red Flags PNPK POGI)
    red_flags: List[str] = []
    if sbp >= 160 or dbp >= 110:
        red_flags.append(f"Krisis Hipertensi Berat (TD {int(sbp)}/{int(dbp)} mmHg >= 160/110 mmHg)")
    elif sbp >= 140 or dbp >= 90:
        red_flags.append(f"Hipertensi Gestasional / Preeklamsia (TD {int(sbp)}/{int(dbp)} mmHg >= 140/90 mmHg)")
    if map_val >= 106:
        red_flags.append(f"Mean Arterial Pressure Kritis (MAP {map_val:.1f} mmHg >= 106 mmHg)")
    elif map_val >= 95:
        red_flags.append(f"Peningkatan Resistensi Vaskular (MAP {map_val:.1f} mmHg >= 95 mmHg)")
    if delta_map >= 15:
        red_flags.append(f"Lonjakan Hemodinamik Longitudinal Tajam (ΔMAP +{delta_map:.1f} mmHg dari Trimester 1)")
    if wg_vel >= 0.75:
        red_flags.append(f"Laju Penambahan BB Cepat ({wg_vel:.2f} kg/minggu >= 0.75 kg/minggu; curiga retensi cairan endotel)")

    if "Tier 2" in screening_mode or "RSUD" in screening_mode:
        if proteinuria_ord >= 2:
            red_flags.append(f"Proteinuria Signifikan (+{proteinuria_ord} pada carik celup)")
        if platelet < 150:
            red_flags.append(f"Trombositopenia ({int(platelet)} x10³/µL < 150 x10³/µL; risiko koagulopati/HELLP)")
        if creatinine >= 1.1:
            red_flags.append(f"Insufisiensi Ginjal Maternal (Kreatinin {creatinine:.2f} mg/dL >= 1.1 mg/dL)")
        if sgot >= 70:
            red_flags.append(f"Peningkatan Transaminase Hepar (SGOT {int(sgot)} U/L >= 70 U/L)")
        if utapi >= 1.40:
            red_flags.append(f"Impedansi Arteri Uterina Tinggi (UtAPI {utapi:.2f} >= 1.40; hipoperfusi plasenta)")

    # Penentuan Tier Risiko & Rekomendasi POGI
    if prob_pe > 0.40 or sbp >= 140 or dbp >= 90 or map_val >= 106 or (proteinuria_ord >= 2 and "Tier 2" in screening_mode):
        risk_tier = "RISIKO TINGGI (> 40%)"
        risk_level_code = "HIGH"
        badge_color = "#c92a2a"
        urgency = "SEGERA / EMERGENSI OBSTETRI (Rujukan Vertikal dalam < 24 Jam)"
        target_facility = "RSUD Rujukan PONEK / Poliklinik Spesialis Obstetri & Ginekologi (Sp.OG)"

        if subtype_pred == 2 or gest_age < 34:
            subtype_label = "Early-Onset Preeclampsia (< 34 Minggu) — Risiko Kritis Maternal-Fetal"
        else:
            subtype_label = "Late-Onset Preeclampsia (>= 34 Minggu)"

        actions = [
            "Protokol rujukan segera ke Dokter Spesialis Obstetri & Ginekologi (Sp.OG) di RSUD / Fasilitas PONEK.",
            "Lakukan skrining USG Doppler Arteri Uterina (UtAPI), biometri janin, dan pemantauan kesejahteraan janin (NST/CTG).",
            "Evaluasi panel laboratorium organ target lengkap (rasio protein/kreatinin urin, trombosit, SGOT/SGPT, kreatinin serum, LDH).",
            "Pertimbangkan pemberian antihipertensi lini pertama (Nifedipin lepas lambat / Metildopa) jika TD >= 140/90 mmHg dan profilaksis MgSO4 bila terdapat gejala preeklamsia berat.",
            "Persiapkan rencana terminasi kehamilan terarah atau pematangan paru janin (Kortikosteroid Deksametason) apabila usia kehamilan < 34 minggu (Early-Onset).",
        ]
    elif prob_pe >= 0.15 or map_val >= 95 or delta_map >= 12 or wg_vel >= 0.75:
        risk_tier = "RISIKO SEDANG (15% - 40%)"
        risk_level_code = "MODERATE"
        badge_color = "#e67700"
        urgency = "PENGAWASAN KETAT (Evaluasi Dokter FKTP / Konsultasi Sp.OG Terjadwal)"
        target_facility = "Puskesmas Induk (Dokter Umum) / Poliklinik Kebidanan RS"
        subtype_label = (
            "Waspada Pre-Preeklamsia / Hipertensi Gestasional Dini (< 34 Minggu)"
            if gest_age < 34
            else "Waspada Late-Onset Pre-Preeklamsia (>= 34 Minggu)"
        )
        actions = [
            "Evaluasi pemberian profilaksis Aspirin Dosis Rendah (80–150 mg/hari malam hari, optimal dimulai sebelum usia kehamilan 16 minggu hingga 36 minggu) sesuai PNPK POGI.",
            "Berikan suplementasi Kalsium oral (1.000–1.500 mg/hari) terutama pada populasi dengan asupan kalsium harian rendah.",
            "Tingkatkan frekuensi monitoring tekanan darah, MAP, dan berat badan setiap 2 minggu sekali pada kunjungan ANC.",
            "Lakukan pemeriksaan proteinuria dipstick ulang dan edukasi tanda bahaya preeklamsia (nyeri kepala frontal hebat, gangguan visus/pandangan kabur, nyeri ulu hati/epigastrium).",
        ]
    else:
        risk_tier = "RISIKO RENDAH (< 15%)"
        risk_level_code = "LOW"
        badge_color = "#2b8a3e"
        urgency = "ANC RUTIN (Pemantauan Berkala Standar Kemenkes)"
        target_facility = "Fasilitas Kesehatan Tingkat Pertama (Puskesmas / Pustu / Posyandu / Bidan)"
        subtype_label = "Normal / Non-Preeklamsia (Kehamilan Normotensif)"
        actions = [
            "Lanjutkan jadwal Antenatal Care (ANC) rutin terpadu di FKTP (minimal 6x kunjungan selama kehamilan sesuai pedoman Kemenkes RI).",
            "Berikan edukasi nutrisi seimbang tinggi protein, pembatasan natrium berlebih, dan suplementasi zat besi-asam folat serta kalsium standar.",
            "Pantau kurva penambahan berat badan mingguan dan catat Mean Arterial Pressure (MAP) pada Buku KIA setiap kunjungan.",
        ]

    return {
        "probability": round(float(prob_pe), 4),
        "probability_pct": round(float(prob_pe) * 100.0, 2),
        "risk_tier": risk_tier,
        "risk_level_code": risk_level_code,
        "badge_color": badge_color,
        "subtype_label": subtype_label,
        "urgency": urgency,
        "target_facility": target_facility,
        "red_flags": red_flags,
        "recommended_actions": actions,
    }
