# src/preprocessing/mrna_uncertainty_quantify.py
# Uncertainty quantification via Cox SE and dual-track selection.

import sys
import json
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.io_utils import load_config


def main():
    config = load_config()
    hparams = config["hyperparams"]

    input_h5 = Path(config["data"]["processed_genomics"]) / "Cancer_mRNA_top1000.h5"
    output_csv = Path(config["data"]["processed_genomics"]) / "Cancer_mRNA_IU_selected.csv"
    output_meta = Path(config["data"]["processed_genomics"]) / "Cancer_mRNA_IU_selection_metadata.csv"
    output_json = Path(config["data"]["processed_genomics"]) / "Cancer_mRNA_IU_index.json"

    print("[INFO] mRNA uncertainty quantification (SE method) + dual-track selection")
    print(f"[INFO] Input: {input_h5}")

    # Load top-1000
    with h5py.File(input_h5, "r") as f:
        expr_top_n = f["expression"][:]
        gene_names = [n.decode() for n in f["gene_names"][:]]
        patient_ids = [p.decode() for p in f["patient_ids"][:]]

        metadata = []
        meta_group = f["metadata"]
        for gene in gene_names:
            grp = meta_group[gene]
            metadata.append({
                "gene_name": gene,
                "hr": grp.attrs["hr"],
                "p_value": grp.attrs["p_value"],
            })
        top_n_genes = pd.DataFrame(metadata)

    print(f"[INFO] Loaded: {expr_top_n.shape}")

    # Survival data
    clinical_path = Path(config["files"]["clinical"])
    clinical_df = pd.read_csv(clinical_path)
    clinical_df = clinical_df[clinical_df["patient_id"].isin(patient_ids)].sort_values("patient_id").reset_index(drop=True)

    if clinical_df["os_status"].dtype == object:
        mapping = {"Dead": 1, "Alive": 0, "dead": 1, "alive": 0}
        clinical_df["os_status"] = clinical_df["os_status"].map(mapping)
    clinical_df["os_time"] = pd.to_numeric(clinical_df["os_time"], errors="coerce")

    survival_data = clinical_df[["os_time", "os_status"]].copy()
    survival_data.columns = ["duration", "event"]
    print(f"[INFO] Patients: {len(survival_data)}, Events: {int(survival_data['event'].sum())}")

    # Compute SE uncertainty
    def compute_se(gene_idx: int) -> float:
        gene_expr = expr_top_n[:, gene_idx]
        df_temp = survival_data.copy()
        df_temp["gene_expr"] = gene_expr
        try:
            cph = CoxPHFitter()
            cph.fit(df_temp, duration_col="duration", event_col="event")
            if len(cph.summary) > 0:
                return float(cph.summary["se(coef)"].iloc[0])
        except Exception:
            pass
        return np.nan

    uncertainty_scores = []
    for idx in tqdm(range(len(gene_names)), desc="SE computation"):
        uncertainty_scores.append(compute_se(idx))

    top_n_genes["uncertainty_score"] = uncertainty_scores
    top_n_genes = top_n_genes.dropna(subset=["uncertainty_score"]).reset_index(drop=True)
    print(f"[INFO] Valid genes: {len(top_n_genes)}")
    print(f"[INFO] SE range: [{top_n_genes['uncertainty_score'].min():.4f}, {top_n_genes['uncertainty_score'].max():.4f}]")

    # Dual-track selection
    unc_scores = top_n_genes["uncertainty_score"].values
    unc_low = np.mean(unc_scores) - 0.5 * np.std(unc_scores)
    unc_high = np.mean(unc_scores) + 0.5 * np.std(unc_scores)

    n_consensus = hparams["n_consensus"]
    n_exploration = hparams["n_exploration"]

    consensus = top_n_genes[top_n_genes["uncertainty_score"] <= unc_low].head(n_consensus)
    if len(consensus) < n_consensus:
        consensus = top_n_genes.nsmallest(n_consensus, "uncertainty_score")

    exploration = top_n_genes[top_n_genes["uncertainty_score"] >= unc_high].head(n_exploration)
    if len(exploration) < n_exploration:
        exploration = top_n_genes.nlargest(n_exploration, "uncertainty_score")

    selected = pd.concat([consensus, exploration]).drop_duplicates(subset="gene_name").reset_index(drop=True)
    selected["selection_type"] = selected.apply(
        lambda row: "consensus" if row.name in consensus.index else "exploration", axis=1
    )

    print(f"[INFO] Consensus: {len(consensus)}, Exploration: {len(exploration)}, Total: {len(selected)}")

    # Extract and standardize
    selected_indices = selected.index.values
    final_expr = expr_top_n[:, selected_indices]
    scaler = StandardScaler()
    final_expr_scaled = scaler.fit_transform(final_expr)

    selected_gene_names = top_n_genes.iloc[selected_indices]["gene_name"].tolist()

    # Save CSV
    output_df = pd.DataFrame(final_expr_scaled, columns=selected_gene_names)
    output_df.insert(0, "patient_id", patient_ids)
    output_df.to_csv(output_csv, index=False)
    print(f"[INFO] Saved CSV: {output_csv}, shape {output_df.shape}")

    # Save metadata
    selected.to_csv(output_meta, index=False)
    print(f"[INFO] Saved metadata: {output_meta}")

    # Save JSON index
    index_data = {
        "file_path": str(output_csv),
        "n_patients": final_expr_scaled.shape[0],
        "n_genes": final_expr_scaled.shape[1],
        "gene_names": selected_gene_names,
        "patient_ids": patient_ids,
        "attributes": {
            "n_consensus": len(consensus),
            "n_exploration": len(exploration),
            "selection_method": "cox + se_uncertainty",
            "normalization": "zscore",
        },
    }
    with open(output_json, "w") as f:
        json.dump(index_data, f, indent=2)
    print(f"[INFO] Saved index: {output_json}")
    print("[INFO] Done")


if __name__ == "__main__":
    main()