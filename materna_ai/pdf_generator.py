# -*- coding: utf-8 -*-
"""
Modul 8: Clinical Summary & Referral PDF Generator
Menghasilkan dokumen Lembar Resume Skrining Klinis & Surat Rujukan Formal POGI/FKTP
berformat PDF siap cetak dalam 1 klik (mendukung output bytes untuk Streamlit/FastAPI
maupun penyimpanan ke file lokal).
"""

import io
import datetime
import textwrap
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from .config import INV_DIPSTICK_MAP


def generate_clinical_referral_pdf_bytes(
    patient: Dict[str, Any],
    pogi_result: Dict[str, Any],
    xai_result: Dict[str, Any],
    screening_mode: str = "Mode Tier 2 (RSUD Rujukan)",
    model_name: str = "Best Model Ensemble",
    output_path: Optional[Path] = None,
) -> bytes:
    """
    Menyusun PDF 1 halaman A4 berstandar klinis yang memuat:
    1. Kop Resmi Sistem Pendukung Keputusan Klinis MATERNA-AI & Identitas Teranonimkan (UU PDP)
    2. Kategori Risiko Terkalibrasi, Probabilitas ML, & Stratifikasi Subtipe Onset
    3. Tabel Riwayat Tanda Vital, Hemodinamik Longitudinal (MAP & Delta MAP), dan Biomarker
    4. Diagram Kontribusi Risiko (SHAP Waterfall Plot) & Faktor Pemicu Kuantitatif
    5. Protokol Rekomendasi Klinis POGI & Lembar Rujukan Formal Fasilitas Kesehatan
    """
    buf = io.BytesIO()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M WIB")
    patient_id = str(patient.get("patient_id", "ANON-MAT-2026-001"))
    faskes_origin = str(patient.get("faskes_origin", "FKTP / Puskesmas")).replace("_", " ")

    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 Portrait
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        # 1. Header / Kop Surat
        ax.add_patch(plt.Rectangle((0.05, 0.905), 0.90, 0.068, facecolor="#1B365D", edgecolor="none"))
        ax.text(
            0.50, 0.952,
            "MATERNA-AI — LEMBAR RESUME SKRINING PREEKLAMSIA & SURAT RUJUKAN KLINIS",
            ha="center", va="center", fontsize=11.2, fontweight="bold", color="white"
        )
        ax.text(
            0.50, 0.923,
            f"Multi-Tier Clinical Decision Support System (POGI / PNPK Compliant)  |  Mode: {screening_mode}  |  Model: {model_name}",
            ha="center", va="center", fontsize=8.3, color="#E0E8F5"
        )

        # 2. Identitas Pasien Teranonimkan & Ringkasan Risiko
        gest_age = int(patient.get("gestational_age_weeks", 28))
        mat_age = int(patient.get("maternal_age", 28))
        gravida = int(patient.get("gravida", 1))
        parity = int(patient.get("parity", 0))
        nullipara_str = "Ya (Primigravida/Nullipara)" if int(patient.get("nullipara", 0)) == 1 else "Tidak (Multipara)"

        ax.add_patch(plt.Rectangle((0.05, 0.805), 0.52, 0.088, facecolor="#F4F7FA", edgecolor="#B0C4DE", lw=1.0))
        id_text = (
            f"ID Rekam Medis (Anonim) : {patient_id}\n"
            f"Asal Faskes Pengirim    : {faskes_origin}\n"
            f"Usia Ibu / Usia Gestasi : {mat_age} Tahun  /  {gest_age} Minggu\n"
            f"Status Obstetrik        : G{gravida}P{parity}  |  Nullipara: {nullipara_str}\n"
            f"Waktu Cetak Penapisan   : {now_str}"
        )
        ax.text(0.065, 0.883, id_text, ha="left", va="top", fontsize=8.2, linespacing=1.38, family="monospace")

        # Badge Risiko Terkalibrasi
        badge_color = pogi_result.get("badge_color", "#1B365D")
        ax.add_patch(plt.Rectangle((0.59, 0.805), 0.36, 0.088, facecolor=badge_color, edgecolor="black", lw=1.0, alpha=0.92))
        ax.text(
            0.77, 0.873,
            f"PROBABILITAS RISIKO: {pogi_result.get('probability_pct', 0):.1f}%",
            ha="center", va="center", fontsize=10.5, fontweight="bold", color="white"
        )
        ax.text(
            0.77, 0.846,
            pogi_result.get("risk_tier", "RISIKO RENDAH"),
            ha="center", va="center", fontsize=9.5, fontweight="bold", color="#FFF3BF"
        )
        sub_wrapped = "\n".join(textwrap.wrap(pogi_result.get("subtype_label", ""), width=38))
        ax.text(
            0.77, 0.818,
            sub_wrapped,
            ha="center", va="center", fontsize=7.6, color="white"
        )

        # 3. Tabel Tanda Vital, Hemodinamik Longitudinal, & Biomarker
        ax.text(
            0.05, 0.788,
            "I. PROFIL HEMODINAMIK LONGITUDINAL ANC, ANTROPOMETRI, & BIOMARKER ORGAN TARGET",
            ha="left", va="top", fontsize=9.2, fontweight="bold", color="#1B365D"
        )

        sbp = int(patient.get("systolic_bp", 120))
        dbp = int(patient.get("diastolic_bp", 78))
        map_val = float(patient.get("map_mmhg", 92.0))
        sbp_t1 = int(patient.get("sbp_trimester1", 110))
        dbp_t1 = int(patient.get("dbp_trimester1", 72))
        delta_map = float(patient.get("delta_map", 0.0))
        wg_vel = float(patient.get("weight_gain_kg_per_week", 0.45))
        bmi = float(patient.get("pre_pregnancy_bmi", 23.5))
        lila = float(patient.get("lila_cm", 25.0))
        dip_str = INV_DIPSTICK_MAP.get(int(patient.get("proteinuria_ordinal", 0)), "Negatif")
        pcr = float(patient.get("urine_pcr", 0.15))
        plt_cnt = int(patient.get("platelet_count_k_ul", 240))
        creat = float(patient.get("serum_creatinine_mg_dl", 0.72))
        sgot = int(patient.get("sgot_ast_u_l", 22))
        sgpt = int(patient.get("sgpt_alt_u_l", 20))
        uric = float(patient.get("uric_acid_mg_dl", 4.4))
        utapi = float(patient.get("utapi_doppler", 0.92))

        vital_rows = [
            ["Tekanan Darah Saat Ini", f"{sbp}/{dbp} mmHg", "Cutoff < 140/90 mmHg", "Proteinuria / PCR Urin", f"{dip_str} ({pcr:.2f} mg/mg)", "Normal: Neg / <0.30"],
            ["Mean Arterial Pressure (MAP)", f"{map_val:.2f} mmHg", "Waspada >= 95 mmHg", "Hitung Trombosit", f"{plt_cnt} x10³/µL", "Normal >= 150 x10³/µL"],
            ["TD Trimester 1 & Delta MAP", f"{sbp_t1}/{dbp_t1} | ΔMAP {delta_map:+.1f}", "Waspada ΔMAP >= +12", "Kreatinin & Asam Urat", f"{creat:.2f} & {uric:.1f} mg/dL", "Kreat <1.1 | AU <5.5"],
            ["Laju Kenaikan BB Mingguan", f"{wg_vel:.2f} kg/minggu", "Waspada >= 0.75 kg/mgg", "Enzim Hepar SGOT/SGPT", f"{sgot} / {sgpt} U/L", "Normal < 40 U/L"],
            ["IMT Pra-Hamil & LILA", f"{bmi:.1f} kg/m² | {lila:.1f} cm", "IMT 18.5-24.9 kg/m²", "Doppler Arteri Uterina", f"UtAPI = {utapi:.2f}", "Normal < 1.25 - 1.40"],
        ]

        ax_tbl = fig.add_axes([0.05, 0.645, 0.90, 0.13])
        ax_tbl.axis("off")
        tbl = ax_tbl.table(
            cellText=vital_rows,
            colLabels=["Parameter Klinis & Trajectory", "Nilai Pasien", "Nilai Rujukan", "Biomarker Tier 2", "Nilai Lab", "Rujukan Lab"],
            loc="center",
            cellLoc="left",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.3)
        tbl.scale(1.0, 1.26)
        for (r_i, c_i), cell in tbl.get_celld().items():
            if r_i == 0:
                cell.set_facecolor("#1B365D")
                cell.set_text_props(color="white", fontweight="bold")
            elif r_i % 2 == 0:
                cell.set_facecolor("#F4F7FA")

        # 4. Diagram Kontribusi Risiko (SHAP Waterfall Mini) & Narasi XAI
        ax.text(
            0.05, 0.625,
            "II. EXPLAINABLE AI (XAI) — KONTRIBUSI FAKTOR PEMICU RISIKO KUANTITATIF (SHAP & LIME)",
            ha="left", va="top", fontsize=9.2, fontweight="bold", color="#1B365D"
        )

        attrs = xai_result.get("attributions", [])[:7]
        base_val = float(xai_result.get("base_value", 0.30))
        pred_val = float(xai_result.get("predicted_prob", pogi_result.get("probability", 0.30)))

        ax_shap = fig.add_axes([0.19, 0.425, 0.34, 0.175])
        items_rev = list(reversed(attrs))
        running = base_val
        y_pos = np.arange(len(items_rev))
        y_labels = []
        for idx, it in enumerate(items_rev):
            delta = float(it["shap_value"])
            start = running
            end = running + delta
            running = end
            col_bar = "#c92a2a" if delta >= 0 else "#1d70b8"
            ax_shap.barh(idx, max(abs(delta), 0.005), left=min(start, end), height=0.58, color=col_bar, alpha=0.88)
            ax_shap.text(
                np.clip(end + (0.015 if delta >= 0 else -0.015), 0.02, 0.98),
                idx,
                f"{delta*100:+.1f}%",
                va="center",
                ha="left" if delta >= 0 else "right",
                fontsize=7.0,
                fontweight="bold",
                color=col_bar,
            )
            y_labels.append(it["clinical_label"][:22])

        ax_shap.axvline(base_val, color="gray", linestyle="--", lw=0.9)
        ax_shap.axvline(pred_val, color="#1B365D", linestyle="-", lw=1.3)
        ax_shap.set_yticks(y_pos)
        ax_shap.set_yticklabels(y_labels, fontsize=7.0)
        ax_shap.set_xlim(0.0, 1.05)
        ax_shap.set_xlabel("Probabilitas Risiko Preeklamsia", fontsize=7.2)
        ax_shap.set_title(f"SHAP Waterfall (Base={base_val*100:.1f}% -> Akhir={pred_val*100:.1f}%)", fontsize=7.8, fontweight="bold")
        ax_shap.tick_params(axis="x", labelsize=7.0)

        # Panel Narasi Kuantitatif di sebelah kanan grafik SHAP
        top_narratives = xai_result.get("top_narratives", [])[:5]
        narr_lines = ["Faktor Pemicu & Protektif Utama (XAI):"]
        for n_item in top_narratives:
            wrapped_n = textwrap.wrap(f"• {n_item}", width=48)
            narr_lines.extend(wrapped_n)

        ax.text(
            0.555, 0.598,
            "\n".join(narr_lines[:11]),
            ha="left", va="top", fontsize=7.7, linespacing=1.32,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#FFF9DB", edgecolor="#E67700", lw=0.8)
        )

        # 5. Protokol Keputusan Klinis POGI & Surat Rujukan Formal
        ax.text(
            0.05, 0.388,
            "III. REKOMENDASI PROTOKOL KLINIS POGI / PNPK & LEMBAR RUJUKAN FASILITAS KESEHATAN",
            ha="left", va="top", fontsize=9.2, fontweight="bold", color="#1B365D"
        )

        red_flags = pogi_result.get("red_flags", [])
        rf_str = "; ".join(red_flags[:4]) if red_flags else "Tidak ditemukan tanda bahaya akut organ target pada saat pemeriksaan."
        actions = pogi_result.get("recommended_actions", [])

        ref_lines = [
            f"Urgensi Penanganan : {pogi_result.get('urgency', '-')}",
            f"Tujuan Rujukan     : {pogi_result.get('target_facility', '-')}",
            f"Temuan Red Flags   : {rf_str}",
            "",
            "Rekomendasi Tata Laksana Medis & Obstetrik (Sesuai PNPK POGI):",
        ]
        for idx_a, act in enumerate(actions, start=1):
            wrapped_act = textwrap.wrap(f"{idx_a}. {act}", width=102)
            ref_lines.extend(wrapped_act)

        ax.add_patch(plt.Rectangle((0.05, 0.135), 0.90, 0.238, facecolor="#F8F9FA", edgecolor="#1B365D", lw=1.0))
        ax.text(0.065, 0.362, "\n".join(ref_lines[:13]), ha="left", va="top", fontsize=8.0, linespacing=1.35)

        # Blok Tanda Tangan & Validasi Rujukan
        ax.text(
            0.06, 0.112,
            "Catatan: Dokumen ini dihasilkan secara otomatis oleh Sistem Pendukung Keputusan Klinis MATERNA-AI.\n"
            "Keputusan medis definitif tetap berada pada otoritas klinis Dokter / Bidan penanggung jawab pasien.",
            ha="left", va="top", fontsize=7.2, style="italic", color="#555555"
        )
        ax.text(
            0.78, 0.115,
            "Petugas Pemeriksa / Perujuk,\n\n\n____________________________\n(Bidan / Dokter Pemeriksa FKTP/RS)",
            ha="center", va="top", fontsize=7.8, color="#222222"
        )
        ax.text(
            0.50, 0.02,
            "MATERNA-AI v2.5 — Multi-Tier Clinical Decision Support System for Early Preeclampsia Prediction & Risk Trajectory",
            ha="center", va="center", fontsize=7.5, color="#666666"
        )

        pdf.savefig(fig)
        plt.close(fig)

    pdf_bytes = buf.getvalue()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(pdf_bytes)

    return pdf_bytes
