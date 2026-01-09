# physics-guided-ml-elastic-velocities
Physics-guided machine learning workflows for estimating elastic velocities from conventional well logs
[CHONAI_full_workflow_RF_Vs.py](https://github.com/user-attachments/files/24521001/CHONAI_full_workflow_RF_Vs.py)
"""
CHONAI-01 full workflow: Convert DTLF→Vp and DTLN→Vs, train Random Forest to predict Vs,
and export paper-ready outputs (metrics, feature importance, depth-wise Vs comparison, PDFs).

Assumptions:
- DTLF and DTLN are sonic slowness in microseconds per foot (µs/ft).
- Velocity conversion: V (km/s) = 304.8 / DT(µs/ft)

Outputs:
1) Excel workbook with RF performance and logs:
   - RF_metrics
   - Feature_importance
   - Vs_measured_vs_RF
   - Predictor_list
2) PDF: Vs Measured vs RF Predicted vs Depth
3) Saved model bundle (joblib)

Author: Generated via ChatGPT for Ali Hassan
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    # -------------------- Paths (edit if needed) --------------------
    data_path = Path(r"/mnt/data/CHONAI-01_DATTA_Formation_with_values.xlsx")
    out_xlsx = Path("CHONAI_RF_Performance_for_Paper.xlsx")
    out_pdf_vs = Path("CHONAI_Vs_Measured_vs_RF_only.pdf")
    out_model = Path("CHONAI_RF_Vs_model_for_paper.joblib")

    # -------------------- Load --------------------
    df = pd.read_excel(data_path).copy()
    for col in ["DEPT", "DTLF", "DTLN"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # -------------------- Convert DT -> velocity --------------------
    # V (km/s) = 304.8 / DT(µs/ft)
    df["Vp_kms"] = np.where(df["DTLF"] > 0, 304.8 / df["DTLF"], np.nan)
    df["Vs_kms"] = np.where(df["DTLN"] > 0, 304.8 / df["DTLN"], np.nan)

    # -------------------- Prepare modeling data --------------------
    # Target is Vs_kms (from DTLN). Exclude DTLN from inputs to avoid leakage.
    feature_cols = [c for c in df.columns if c not in ["Vs_kms", "DTLN"]]
    if "DTLF" in feature_cols:
        feature_cols.remove("DTLF")  # avoid duplicating Vp signal; use Vp_kms instead

    model_df = df[feature_cols + ["Vs_kms"]].dropna()
    X = model_df[feature_cols]
    y = model_df["Vs_kms"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, shuffle=True
    )

    # -------------------- Train RF --------------------
    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
        oob_score=True,
        bootstrap=True
    )
    rf.fit(X_train, y_train)

    # -------------------- Metrics --------------------
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)

    metrics = {
        "Assumption_DT_units": "µs/ft",
        "Conversion_formula": "V (km/s) = 304.8 / DT(µs/ft)",
        "Target": "Vs_kms (from DTLN)",
        "Leakage_prevention": "DTLN excluded from predictors; Vp_kms used instead of DTLF",
        "Rows_used_total": int(len(model_df)),
        "Train_rows": int(len(X_train)),
        "Test_rows": int(len(X_test)),
        "Num_features": int(X.shape[1]),
        "RF_n_estimators": 300,
        "RF_min_samples_leaf": 2,
        "RF_bootstrap": True,
        "RF_random_state": 42,
        "OOB_R2": float(rf.oob_score_),
        "Train_R2": float(r2_score(y_train, y_train_pred)),
        "Test_R2": float(r2_score(y_test, y_test_pred)),
        "Train_RMSE_km_s": rmse(y_train, y_train_pred),
        "Test_RMSE_km_s": rmse(y_test, y_test_pred),
        "Train_MAE_km_s": float(mean_absolute_error(y_train, y_train_pred)),
        "Test_MAE_km_s": float(mean_absolute_error(y_test, y_test_pred)),
    }
    metrics_df = pd.DataFrame([metrics])

    # Feature importance
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)

    # Depth-wise predictions
    df = df.sort_values("DEPT")
    df["Vs_RF_kms"] = np.nan
    mask = df[feature_cols].notna().all(axis=1)
    df.loc[mask, "Vs_RF_kms"] = rf.predict(df.loc[mask, feature_cols])

    vs_track_df = df[["DEPT", "Vs_kms", "Vs_RF_kms"]].rename(columns={
        "Vs_kms": "Vs_measured_km_s",
        "Vs_RF_kms": "Vs_RF_pred_km_s"
    })

    # -------------------- Export Excel --------------------
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        metrics_df.to_excel(writer, index=False, sheet_name="RF_metrics")
        imp_df.to_excel(writer, index=False, sheet_name="Feature_importance")
        vs_track_df.to_excel(writer, index=False, sheet_name="Vs_measured_vs_RF")
        pd.DataFrame({"Predictor_features_used": feature_cols}).to_excel(
            writer, index=False, sheet_name="Predictor_list"
        )

    # -------------------- Export PDF (Vs only) --------------------
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_subplot(111)

    ax.plot(vs_track_df["Vs_measured_km_s"], vs_track_df["DEPT"],
            color="black", linewidth=1.2, label="Vs (measured from DTLN)")
    ax.plot(vs_track_df["Vs_RF_pred_km_s"], vs_track_df["DEPT"],
            color="tab:red", linewidth=1.2, linestyle="--", label="Vs (RF predicted)")

    ax.set_xlabel("Vs (km/s)")
    ax.set_ylabel("Depth")
    ax.set_title("CHONAI-01: Vs Measured vs RF Predicted")
    ax.grid(True)
    ax.invert_yaxis()
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_pdf_vs, format="pdf")
    plt.close(fig)

    # -------------------- Save model --------------------
    joblib.dump({"model": rf, "features": feature_cols, "assumption": metrics["Conversion_formula"]}, out_model)

    print("Saved:", out_xlsx)
    print("Saved:", out_pdf_vs)
    print("Saved:", out_model)


if __name__ == "__main__":
    main()
