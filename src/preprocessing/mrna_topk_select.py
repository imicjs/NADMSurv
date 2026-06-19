# src/preprocessing/mrna_topk_select.py
# Top-1000 mRNA gene pre-selection with variance filtering and univariate Cox.

import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.feature_selection import VarianceThreshold
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.io_utils import load_config, align_patient_ids


def main():
    config = load_config()
    hparams = config["hyperparams"]

    clinical_path = Path(config["files"]["clinical"])
    mrna_path = Path(config["files"]["mrna"])
    wsi_delete_path = Path(config["files"]["wsi_delete"])
    output_path = Path(config["data"]["processed_genomics"]) / "Cancer_mRNA_top1000.h5"

    print("[INFO] Loading clinical, mRNA and WSI delete lists")

    clinical_df = pd.read_csv(clinical_path)
    mrna_df = pd.read_csv(mrna_path)
    wsi_df = pd.read_csv(wsi_delete_path)

    common_patients = align_patient_ids(clinical_df, mrna_df, wsi_df)
    print(f"[INFO] Aligned patients: {len(common_patients)}")

    clinical_df = clinical_df[clinical_df["patient_id"].isin(common_patients)].sort_values("patient_id").reset_index(drop=True)
    mrna_df = mrna_df[mrna_df["patient_id"].isin(common_patients)].sort_values("patient_id").reset_index(drop=True)

    assert list(clinical_df["patient_id"]) == list(mrna_df["patient_id"])
    patient_ids = clinical_df["patient_id"].tolist()

    expr_matrix = mrna_df.drop(columns=["patient_id"]).values
    gene_names = mrna_df.drop(columns=["patient_id"]).columns.tolist()
    print(f"[INFO] Expression matrix: {expr_matrix.shape}")

    if clinical_df["os_status"].dtype == object:
        mapping = {"Dead": 1, "Alive": 0, "dead": 1, "alive": 0}
        clinical_df["os_status"] = clinical_df["os_status"].map(mapping)

    clinical_df["os_time"] = pd.to_numeric(clinical_df["os_time"], errors="coerce")
    clinical_df["os_status"] = pd.to_numeric(clinical_df["os_status"], errors="coerce")

    if clinical_df["os_time"].isna().any() or clinical_df["os_status"].isna().any():
        raise ValueError("Invalid survival data")

    survival_data = clinical_df[["os_time", "os_status"]].copy()
    survival_data.columns = ["duration", "event"]
    n_events = int(survival_data["event"].sum())
    print(f"[INFO] Survival events: {n_events} ({n_events/len(survival_data):.2%})")

    nan_ratio = np.isnan(expr_matrix).mean()
    if nan_ratio > 0:
        gene_medians = np.nanmedian(expr_matrix, axis=0)
        expr_matrix = np.where(np.isnan(expr_matrix), gene_medians, expr_matrix)
        print(f"[INFO] NaN ratio: {nan_ratio:.2%}")
    else:
        print("[INFO] NaN ratio: 0.00%")

    selector = VarianceThreshold(threshold=hparams["variance_threshold"])
    expr_filtered = selector.fit_transform(expr_matrix)
    kept_genes = [gene_names[i] for i in selector.get_support(indices=True)]
    print(f"[INFO] Variance filtering (threshold={hparams['variance_threshold']}): {len(gene_names)} -> {len(kept_genes)} genes")

    cox_results = []
    for i, gene in enumerate(tqdm(kept_genes, desc="Cox regression")):
        df_temp = survival_data.copy()
        df_temp["gene_expr"] = expr_filtered[:, i]
        try:
            cph = CoxPHFitter()
            cph.fit(df_temp, duration_col="duration", event_col="event")
            if len(cph.summary) > 0:
                cox_results.append({
                    "gene_name": gene,
                    "p_value": float(cph.summary["p"].iloc[0]),
                    "hr": float(cph.summary["exp(coef)"].iloc[0]),
                    "gene_index": i,
                })
        except Exception:
            continue

    cox_df = pd.DataFrame(cox_results)
    significant = cox_df[cox_df["p_value"] < hparams["cox_p_threshold"]].copy()
    if len(significant) == 0:
        significant = cox_df[cox_df["p_value"] < 0.05].copy()
    if len(significant) == 0:
        raise ValueError("No significant genes found")

    print(f"[INFO] Cox regression (P<{hparams['cox_p_threshold']}): {len(kept_genes)} -> {len(significant)} significant genes")

    significant = significant.sort_values("hr", key=lambda x: x.abs(), ascending=False)
    top_n = significant.head(hparams["n_top_genes"]).copy()

    expr_top_n = expr_filtered[:, top_n["gene_index"].values]
    selected_gene_names = [kept_genes[i] for i in top_n["gene_index"].values]

    print(f"[INFO] Selected top 1000 by absolute HR")
    print(f"[INFO] HR range: [{top_n['hr'].min():.4f}, {top_n['hr'].max():.4f}]")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.attrs["creation_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.attrs["description"] = "mRNA Top-1000 prognostic genes (unstandardized)"
        f.attrs["n_genes"] = len(selected_gene_names)
        f.attrs["variance_threshold"] = hparams["variance_threshold"]

        f.create_dataset("expression", data=expr_top_n, compression="gzip", compression_opts=4)
        f.create_dataset("gene_names", data=np.array(selected_gene_names, dtype="S32"))
        f.create_dataset("patient_ids", data=np.array(patient_ids, dtype="S32"))

        meta_group = f.create_group("metadata")
        for _, row in top_n.iterrows():
            g = meta_group.create_group(row["gene_name"])
            g.attrs["hr"] = row["hr"]
            g.attrs["p_value"] = row["p_value"]
            g.attrs["gene_index"] = row["gene_index"]

    print(f"[INFO] Saved Cancer_mRNA_top1000.h5, shape {expr_top_n.shape}")
    print("[INFO] Done")


if __name__ == "__main__":
    main()