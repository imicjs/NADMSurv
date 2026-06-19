# src/preprocessing/mrna_noise_quantify.py
# Patient-level technical noise quantification via Poisson-binomial hierarchical model.

import sys
import json
import warnings
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.special import digamma, polygamma
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.io_utils import load_config, align_patient_ids

warnings.filterwarnings("ignore")


def main():
    config = load_config()
    hparams = config["hyperparams"]

    clinical_path = Path(config["files"]["clinical"])
    mrna_path = Path(config["files"]["mrna"])
    wsi_delete_path = Path(config["files"]["wsi_delete"])

    output_scores = Path(config["data"]["processed_genomics"]) / "Cancer_mRNA_noise_IU_scores.h5"
    output_labels = Path(config["data"]["processed_genomics"]) / "Cancer_mRNA_noise_IU_labels.json"
    output_report = Path(config["data"]["processed_genomics"]) / "Cancer_mRNA_noise_IU_report.csv"

    print("[INFO] Loading aligned data")

    clinical_df = pd.read_csv(clinical_path)
    mrna_df = pd.read_csv(mrna_path)
    wsi_df = pd.read_csv(wsi_delete_path)

    common_patients = align_patient_ids(clinical_df, mrna_df, wsi_df)
    print(f"[INFO] Aligned patients: {len(common_patients)}")

    clinical_df = clinical_df[clinical_df["patient_id"].isin(common_patients)].sort_values("patient_id").reset_index(drop=True)
    mrna_df = mrna_df[mrna_df["patient_id"].isin(common_patients)].sort_values("patient_id").reset_index(drop=True)
    patient_ids = clinical_df["patient_id"].tolist()

    assert list(clinical_df["patient_id"]) == list(mrna_df["patient_id"])

    expr_counts = mrna_df.drop(columns=["patient_id"]).values
    gene_names = mrna_df.drop(columns=["patient_id"]).columns.tolist()
    print(f"[INFO] Count matrix: {expr_counts.shape}")

    mean_counts = np.mean(expr_counts, axis=0)
    valid_genes = mean_counts >= 1
    expr_filtered = expr_counts[:, valid_genes]
    gene_names_filtered = [gene_names[i] for i in range(len(gene_names)) if valid_genes[i]]
    print(f"[INFO] Gene filtering: {len(gene_names_filtered)} genes retained")

    if len(gene_names_filtered) > 50000:
        top_idx = np.argsort(-mean_counts[valid_genes])[:50000]
        expr_filtered = expr_filtered[:, top_idx]
        gene_names_filtered = [gene_names_filtered[i] for i in top_idx]
        print("[INFO] Trimmed to top 50000 genes")

    N_patients, N_genes = expr_filtered.shape

    pi_params = np.ones(N_genes) * 0.9
    alpha_params = np.ones(N_genes) * 5.0
    beta_params = np.ones(N_genes) * 1.0

    converged = False
    for em_iter in tqdm(range(50), desc="EM iteration"):
        lambda_hat = expr_filtered

        numerator = alpha_params + lambda_hat
        denominator = alpha_params + beta_params + lambda_hat.sum(axis=0)
        E_pi = numerator / denominator

        pi_new = E_pi.mean(axis=0)

        mean_pi = pi_new
        var_pi = E_pi.var(axis=0)
        var_pi = np.clip(var_pi, 1e-8, None)

        alpha_new = mean_pi * (mean_pi * (1 - mean_pi) / var_pi - 1)
        beta_new = (1 - mean_pi) * (mean_pi * (1 - mean_pi) / var_pi - 1)

        alpha_new = np.clip(alpha_new, 0.1, 50.0)
        beta_new = np.clip(beta_new, 0.1, 50.0)

        delta_pi = np.mean(np.abs(pi_new - pi_params))
        if delta_pi < 1e-6 and em_iter > 10:
            converged = True
            break

        pi_params = pi_new
        alpha_params = alpha_new
        beta_params = beta_new

    print(f"[INFO] EM converged at iteration {em_iter + 1}, delta_pi={delta_pi:.2e}")

    gene_importance = 1.0 - np.abs(pi_params - 0.5)
    gene_uncertainty = (alpha_params * beta_params) / (
        (alpha_params + beta_params) ** 2 * (alpha_params + beta_params + 1)
    )

    patient_scores = []
    for j in tqdm(range(N_patients), desc="Patient aggregation"):
        patient_expr = expr_filtered[j, :]
        expr_weights = np.sqrt(patient_expr / (patient_expr.max() + 1e-8))

        noise_intensity = float(np.mean(expr_weights * (1.0 - gene_importance)))
        importance_score = float(1.0 - noise_intensity)
        uncertainty_score = float(np.mean(expr_weights * gene_uncertainty))

        patient_scores.append({
            "patient_id": patient_ids[j],
            "noise_intensity": noise_intensity,
            "importance_score": importance_score,
            "uncertainty_score": uncertainty_score,
            "status": "SUCCESS",
        })

    print("[INFO] Patient-level noise scores computed")

    importance_list = [p["importance_score"] for p in patient_scores]
    uncertainty_list = [p["uncertainty_score"] for p in patient_scores]

    imp_high = np.percentile(importance_list, 70)
    imp_low = np.percentile(importance_list, 30)

    high_imp_idx = [i for i, s in enumerate(importance_list) if s >= imp_high]
    low_imp_idx = [i for i, s in enumerate(importance_list) if s <= imp_low]

    label3 = sorted(high_imp_idx, key=lambda i: uncertainty_list[i])[:100]
    remaining_high = [i for i in high_imp_idx if i not in label3]
    label2 = sorted(remaining_high, key=lambda i: -uncertainty_list[i])[:200]

    label1 = sorted(low_imp_idx, key=lambda i: uncertainty_list[i])[:200]
    remaining_low = [i for i in low_imp_idx if i not in label1]
    label0 = sorted(remaining_low, key=lambda i: -uncertainty_list[i])[:200]

    assigned = set(label0 + label1 + label2 + label3)
    for i in range(len(patient_scores)):
        if i not in assigned:
            p = patient_scores[i]
            if p["importance_score"] >= imp_high:
                if p["uncertainty_score"] <= np.median([uncertainty_list[j] for j in high_imp_idx]):
                    label2.append(i)
                else:
                    label2.append(i)
            elif p["importance_score"] <= imp_low:
                if p["uncertainty_score"] <= np.median([uncertainty_list[j] for j in low_imp_idx]):
                    label1.append(i)
                else:
                    label0.append(i)
            else:
                if p["uncertainty_score"] > np.median(uncertainty_list):
                    label0.append(i)
                else:
                    label1.append(i)

    label0 = list(set(label0))
    label1 = list(set(label1))
    label2 = list(set(label2))
    label3 = list(set(label3))

    label_names = ["low-quality unstable", "low-quality stable", "high-quality unstable", "high-quality stable"]
    for label, indices in enumerate([label0, label1, label2, label3]):
        for i in indices:
            patient_scores[i]["pseudo_label"] = label
            patient_scores[i]["label_name"] = label_names[label]

    label_counts = {}
    for p in patient_scores:
        label_counts[p["pseudo_label"]] = label_counts.get(p["pseudo_label"], 0) + 1

    print(f"[INFO] Label distribution: {dict(label_counts)}")

    output_scores.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_scores, "w") as h5f:
        h5f.attrs["creation_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        h5f.attrs["description"] = "mRNA technical noise (Poisson-binomial, adaptive threshold)"
        h5f.attrs["n_patients"] = len(patient_scores)
        h5f.attrs["n_genes"] = N_genes
        h5f.attrs["em_convergence"] = float(delta_pi)
        h5f.attrs["label3_target_size"] = 100
        h5f.attrs["adaptive_threshold"] = True

        for p in patient_scores:
            grp = h5f.create_group(p["patient_id"])
            grp.attrs["noise_intensity"] = float(p["noise_intensity"])
            grp.attrs["importance_score"] = float(p["importance_score"])
            grp.attrs["uncertainty_score"] = float(p["uncertainty_score"])
            grp.attrs["pseudo_label"] = int(p["pseudo_label"])
            grp.attrs["label_name"] = str(p["label_name"])

            params = np.column_stack([gene_importance, gene_uncertainty, pi_params])
            dset = grp.create_dataset("gene_parameters", data=params, compression="gzip")
            dset.attrs["columns"] = ["importance", "uncertainty", "pi"]

    print(f"[INFO] Saved Cancer_mRNA_noise_IU_scores.h5")

    labels_json = {
        "thresholds": {
            "importance_high": float(imp_high),
            "importance_low": float(imp_low),
            "uncertainty_high": float(np.percentile(uncertainty_list, 70)),
            "uncertainty_low": float(np.percentile(uncertainty_list, 30)),
        },
        "label_distribution": {int(k): int(v) for k, v in label_counts.items()},
        "patients": patient_scores,
        "model_parameters": {
            "n_genes": int(N_genes),
            "n_patients": len(patient_scores),
            "label3_target_size": 100,
        },
    }
    with open(output_labels, "w") as f:
        json.dump(labels_json, f, indent=2)
    print(f"[INFO] Saved Cancer_mRNA_noise_IU_labels.json")

    report_df = pd.DataFrame(patient_scores)
    report_df.to_csv(output_report, index=False)
    print(f"[INFO] Saved Cancer_mRNA_noise_IU_report.csv")
    print("[INFO] Done")


if __name__ == "__main__":
    main()