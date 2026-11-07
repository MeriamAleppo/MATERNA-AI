#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
MATERNA-AI: Modular REST API Backend (FastAPI / Starlette ASGI Structure)
UTS Data Mining - A11.2024.16055 - Nayyara Rauf Ghani Al Khayri
================================================================================
Menyediakan layanan REST API untuk integrasi Rekam Medis Elektronik (RME / SIMRS /
e-Puskesmas) dengan endpoint:
- GET  /health           : Status kesehatan server & daftar model termuat
- GET  /models/metrics   : Ringkasan evaluasi 6 model & Top Fitur ACRFS
- POST /predict/tier1    : Penapisan Mode Tier 1 (Faskes Primer — Tanpa Lab)
- POST /predict/tier2    : Penapisan Mode Tier 2 (RSUD Rujukan — Full Biomarker)
- POST /referral/pdf     : Generator Lembar Resume Klinis & Surat Rujukan PDF
- GET  /openapi.json     : Spesifikasi OpenAPI 3.0

Cara menjalankan server produksi:
    uvicorn api_fastapi:app --host 0.0.0.0 --port 8000
Cara menjalankan pengujian mandiri seluruh endpoint:
    python api_fastapi.py --self-test
================================================================================
"""

import sys
import json
import warnings
from typing import Dict, Any

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app import load_artifacts, predict_patient_case
from materna_ai.pdf_generator import generate_clinical_referral_pdf_bytes

_ARTIFACTS_CACHE = None


def get_artifacts() -> Dict[str, Any]:
    global _ARTIFACTS_CACHE
    if _ARTIFACTS_CACHE is None:
        _ARTIFACTS_CACHE = load_artifacts()
    return _ARTIFACTS_CACHE


def _handle_health() -> Dict[str, Any]:
    art = get_artifacts()
    metrics = art["metrics"]
    return {
        "status": "healthy",
        "service": "MATERNA-AI Clinical Decision Support System API",
        "version": "2.5.0",
        "active_best_model": metrics.get("best_model_name", "SVM"),
        "available_engines": [
            "Tier 1 (Low-Resource / Faskes Primer)",
            "Tier 2 (Advanced Biomarkers / RSUD Rujukan)",
            "Subtype 3-Class Stratification (Normal / Late-Onset / Early-Onset)",
            "Explainable AI (SHAP & LIME)",
            "POGI Clinical Decision Rule Engine",
        ],
    }


def _handle_metrics() -> Dict[str, Any]:
    art = get_artifacts()
    return art["metrics"]


def _handle_predict(payload: Dict[str, Any], mode_label: str) -> Dict[str, Any]:
    art = get_artifacts()
    res = predict_patient_case(payload, mode_label, art)
    return {
        "status": "success",
        "mode": res["mode"],
        "model_used": res["model_label"],
        "audit_warnings": res["audit_warnings"],
        "audited_patient": res["audited_patient"],
        "hemodynamics": res["hemodynamics"],
        "prediction": {
            "preeclampsia_probability": round(res["probability"], 4),
            "preeclampsia_probability_pct": round(res["probability"] * 100.0, 2),
            "risk_tier": res["pogi"]["risk_tier"],
            "subtype_code": res["subtype_code"],
            "subtype_name": res["subtype_name"],
            "subtype_probabilities": [round(float(x), 4) for x in res["subtype_probs"]],
        },
        "xai_explanation": {
            "base_value": res["xai"]["base_value"],
            "predicted_prob": res["xai"]["predicted_prob"],
            "top_narratives": res["xai"]["top_narratives"],
            "top_attributions": res["xai"]["attributions"][:10],
        },
        "pogi_recommendation": res["pogi"],
    }


# Bangun aplikasi ASGI: Gunakan FastAPI jika terinstal, atau Starlette ASGI kompatibel
try:
    from fastapi import FastAPI, Body
    from fastapi.responses import Response

    app = FastAPI(
        title="MATERNA-AI Clinical Decision Support System API",
        description="Multi-Tier Early Preeclampsia Prediction, Longitudinal Trajectory & XAI Referral API",
        version="2.5.0",
    )

    @app.get("/health")
    async def health_endpoint():
        return _handle_health()

    @app.get("/models/metrics")
    async def metrics_endpoint():
        return _handle_metrics()

    @app.post("/predict/tier1")
    async def predict_tier1_endpoint(payload: Dict[str, Any] = Body(...)):
        return _handle_predict(payload, "Mode Tier 1 (Faskes Primer)")

    @app.post("/predict/tier2")
    async def predict_tier2_endpoint(payload: Dict[str, Any] = Body(...)):
        return _handle_predict(payload, "Mode Tier 2 (RSUD Rujukan)")

    @app.post("/referral/pdf")
    async def referral_pdf_endpoint(payload: Dict[str, Any] = Body(...)):
        art = get_artifacts()
        mode_str = payload.get("screening_mode", "Mode Tier 2 (RSUD Rujukan)")
        res = predict_patient_case(payload, mode_str, art)
        pdf_bytes = generate_clinical_referral_pdf_bytes(
            patient=res["audited_patient"],
            pogi_result=res["pogi"],
            xai_result=res["xai"],
            screening_mode=mode_str,
            model_name=res["model_label"],
        )
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=MATERNA_AI_Referral_Report.pdf"},
        )

except ImportError:
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse, Response
    from starlette.routing import Route

    async def health_endpoint(request):
        return JSONResponse(_handle_health())

    async def metrics_endpoint(request):
        return JSONResponse(_handle_metrics())

    async def predict_tier1_endpoint(request):
        payload = await request.json()
        return JSONResponse(_handle_predict(payload, "Mode Tier 1 (Faskes Primer)"))

    async def predict_tier2_endpoint(request):
        payload = await request.json()
        return JSONResponse(_handle_predict(payload, "Mode Tier 2 (RSUD Rujukan)"))

    async def referral_pdf_endpoint(request):
        payload = await request.json()
        art = get_artifacts()
        mode_str = payload.get("screening_mode", "Mode Tier 2 (RSUD Rujukan)")
        res = predict_patient_case(payload, mode_str, art)
        pdf_bytes = generate_clinical_referral_pdf_bytes(
            patient=res["audited_patient"],
            pogi_result=res["pogi"],
            xai_result=res["xai"],
            screening_mode=mode_str,
            model_name=res["model_label"],
        )
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=MATERNA_AI_Referral_Report.pdf"},
        )

    async def openapi_endpoint(request):
        return JSONResponse({
            "openapi": "3.0.2",
            "info": {
                "title": "MATERNA-AI Clinical Decision Support System API",
                "version": "2.5.0",
            },
            "paths": {
                "/health": {"get": {"summary": "Health Check"}},
                "/models/metrics": {"get": {"summary": "Model Metrics & ACRFS Ranking"}},
                "/predict/tier1": {"post": {"summary": "Tier 1 Screening (Primary Care)"}},
                "/predict/tier2": {"post": {"summary": "Tier 2 Screening (Referral Hospital)"}},
                "/referral/pdf": {"post": {"summary": "Generate Printable Clinical Referral PDF"}},
            },
        })

    app = Starlette(
        debug=False,
        routes=[
            Route("/health", health_endpoint, methods=["GET"]),
            Route("/models/metrics", metrics_endpoint, methods=["GET"]),
            Route("/predict/tier1", predict_tier1_endpoint, methods=["POST"]),
            Route("/predict/tier2", predict_tier2_endpoint, methods=["POST"]),
            Route("/referral/pdf", referral_pdf_endpoint, methods=["POST"]),
            Route("/openapi.json", openapi_endpoint, methods=["GET"]),
        ],
    )


def run_api_self_test():
    """Menguji seluruh endpoint REST API secara otomatis menggunakan Starlette TestClient."""
    from starlette.testclient import TestClient

    print("================================================================================")
    print("MATERNA-AI REST API SELF-TEST (FastAPI / Starlette ASGI Verification)")
    print("================================================================================")
    client = TestClient(app)

    r_health = client.get("/health")
    assert r_health.status_code == 200
    print(f"[GET  /health]         -> {r_health.status_code} OK | Active Model: {r_health.json()['active_best_model']}")

    r_metrics = client.get("/models/metrics")
    assert r_metrics.status_code == 200
    print(f"[GET  /models/metrics] -> {r_metrics.status_code} OK | Models evaluated: {list(r_metrics.json()['models'].keys())}")

    sample_t1 = {
        "patient_id": "API-T1-001",
        "maternal_age": 25,
        "gestational_age_weeks": 27,
        "gravida": 1,
        "parity": 1,  # Intentional inconsistency to test Obstetric Validator
        "nullipara": 0,
        "preg_interval_yrs": 2.5,
        "prev_pe_history": 0,
        "family_htn_history": 0,
        "gest_diabetes": 0,
        "pre_pregnancy_bmi": 22.4,
        "lila_cm": 24.5,
        "systolic_bp": 116,
        "diastolic_bp": 74,
        "sbp_trimester1": 112,
        "dbp_trimester1": 72,
        "weight_gain_kg_per_week": 0.42,
        "edema_progression": 0,
    }
    r_t1 = client.post("/predict/tier1", json=sample_t1)
    assert r_t1.status_code == 200
    d_t1 = r_t1.json()
    print(
        f"[POST /predict/tier1]  -> {r_t1.status_code} OK | Prob: {d_t1['prediction']['preeclampsia_probability_pct']}% "
        f"| Tier: {d_t1['prediction']['risk_tier']} | Warnings: {len(d_t1['audit_warnings'])}"
    )

    sample_t2 = {
        "patient_id": "API-T2-002",
        "maternal_age": 36,
        "gestational_age_weeks": 29,
        "gravida": 1,
        "parity": 0,
        "nullipara": 1,
        "preg_interval_yrs": 0.0,
        "prev_pe_history": 0,
        "family_htn_history": 1,
        "gest_diabetes": 1,
        "pre_pregnancy_bmi": 29.4,
        "lila_cm": 29.8,
        "systolic_bp": 154,
        "diastolic_bp": 98,
        "sbp_trimester1": 124,
        "dbp_trimester1": 78,
        "weight_gain_kg_per_week": 0.88,
        "edema_progression": 2,
        "proteinuria_ordinal": 2,
        "urine_pcr": 0.62,
        "platelet_count_k_ul": 138,
        "serum_creatinine_mg_dl": 1.15,
        "sgot_ast_u_l": 58,
        "sgpt_alt_u_l": 52,
        "uric_acid_mg_dl": 6.7,
        "utapi_doppler": 1.58,
    }
    r_t2 = client.post("/predict/tier2", json=sample_t2)
    assert r_t2.status_code == 200
    d_t2 = r_t2.json()
    print(
        f"[POST /predict/tier2]  -> {r_t2.status_code} OK | Prob: {d_t2['prediction']['preeclampsia_probability_pct']}% "
        f"| Subtype: {d_t2['prediction']['subtype_name']}"
    )

    r_pdf = client.post("/referral/pdf", json=sample_t2)
    assert r_pdf.status_code == 200
    assert r_pdf.headers["content-type"] == "application/pdf"
    assert r_pdf.content[:4] == b"%PDF"
    print(f"[POST /referral/pdf]   -> {r_pdf.status_code} OK | PDF Size: {len(r_pdf.content):,} bytes")
    print("================================================================================")
    print("ALL REST API ENDPOINTS VERIFIED SUCCESSFULLY!")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        run_api_self_test()
    else:
        import uvicorn
        uvicorn.run("api_fastapi:app", host="0.0.0.0", port=8000, reload=False)
