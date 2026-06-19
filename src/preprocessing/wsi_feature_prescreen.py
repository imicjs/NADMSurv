# src/preprocessing/wsi_feature_prescreen.py
# WSI feature pre-screening: 24 key patches via MIL attention and k-means.

import sys
import json
import warnings
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.io_utils import load_config

warnings.filterwarnings("ignore")


class MILAttentionScorer(nn.Module):
    def __init__(self, input_dim: int = 2048, hidden_dim: int = 256):
        super().__init__()
        self.V = nn.Linear(input_dim, hidden_dim)
        self.U = nn.Linear(input_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        V_h = torch.tanh(self.V(x))
        U_h = torch.sigmoid(self.U(x))
        attention = self.w(V_h * U_h).squeeze()
        return attention


def select_24_patches(patient_id: str, h5f_in, mil_scorer, m_groups: int, patches_per_group: int, feature_dim: int):
    try:
        features = h5f_in[f"{patient_id}/features"][:]
        n_original = features.shape[0]

        if n_original >= 3000:
            features_use = features[:3000]
        else:
            features_use = np.zeros((3000, feature_dim), dtype=np.float32)
            features_use[:n_original] = features

        kmeans = MiniBatchKMeans(
            n_clusters=m_groups,
            random_state=42,
            batch_size=256,
            max_iter=100,
            n_init=10,
        )
        cluster_labels = kmeans.fit_predict(features_use)

        distances = cosine_distances(features_use, kmeans.cluster_centers_)
        sigma = np.median(np.min(distances, axis=1)) + 1e-8
        similarities = np.exp(-distances / sigma)
        probs = similarities / np.sum(similarities, axis=1, keepdims=True)

        epsilon = 1e-8
        entropy = -np.sum(probs * np.log(probs + epsilon), axis=1)

        features_tensor = torch.from_numpy(features_use).float()
        with torch.no_grad():
            importance_scores = mil_scorer(features_tensor).cpu().numpy()

        selected_indices = []
        uncertainty_labels = []

        for group_id in range(m_groups):
            group_mask = cluster_labels == group_id
            group_indices = np.where(group_mask)[0]

            while len(group_indices) < patches_per_group:
                deficit = patches_per_group - len(group_indices)
                other = np.where(cluster_labels != group_id)[0][:deficit]
                if len(other) == 0:
                    break
                group_indices = np.concatenate([group_indices, other])

            group_importance = importance_scores[group_indices]
            sorted_idx = np.argsort(group_importance)[::-1]
            group_indices = group_indices[sorted_idx]
            group_importance = group_importance[sorted_idx]

            candidate_threshold = max(int(len(group_indices) * 0.2), patches_per_group)
            candidate_indices = group_indices[:candidate_threshold]
            candidate_entropy = entropy[candidate_indices]
            candidate_importance = group_importance[:candidate_threshold]

            median_entropy = np.median(candidate_entropy)

            high_uncertain_mask = candidate_entropy >= median_entropy
            low_uncertain_mask = candidate_entropy < median_entropy

            high_uncertain_indices = candidate_indices[high_uncertain_mask]
            low_uncertain_indices = candidate_indices[low_uncertain_mask]

            high_uncertain_scores = candidate_importance[high_uncertain_mask]
            low_uncertain_scores = candidate_importance[low_uncertain_mask]

            n_high = min(4, len(high_uncertain_indices))
            if n_high < 4:
                if n_high > 0:
                    top_idx = np.argsort(high_uncertain_scores)[-n_high:]
                    selected_high = high_uncertain_indices[top_idx]
                else:
                    selected_high = np.array([], dtype=np.int64)

                remaining = 4 - n_high
                if len(low_uncertain_scores) >= remaining:
                    sup_idx = np.argsort(low_uncertain_scores)[-remaining:]
                    selected_high = np.concatenate([selected_high, low_uncertain_indices[sup_idx]])
                else:
                    sup_needed = 4 - len(selected_high)
                    all_candidates = candidate_indices
                    all_scores = candidate_importance
                    if len(all_scores) >= sup_needed:
                        sup_idx = np.argsort(all_scores)[-sup_needed:]
                        selected_high = np.concatenate([selected_high, all_candidates[sup_idx]])
                    else:
                        selected_high = np.concatenate([selected_high, [-1] * sup_needed])
            else:
                top_idx = np.argsort(high_uncertain_scores)[-4:]
                selected_high = high_uncertain_indices[top_idx]

            n_low = min(4, len(low_uncertain_indices))
            if n_low < 4:
                if n_low > 0:
                    top_idx = np.argsort(low_uncertain_scores)[-n_low:]
                    selected_low = low_uncertain_indices[top_idx]
                else:
                    selected_low = np.array([], dtype=np.int64)

                remaining = 4 - n_low
                if len(high_uncertain_indices) > 4:
                    available_scores = [s for idx, s in zip(high_uncertain_indices, high_uncertain_scores) if idx not in selected_high]
                    available_indices = [idx for idx in high_uncertain_indices if idx not in selected_high]
                    if len(available_scores) >= remaining:
                        sup_idx = np.argsort(available_scores)[-remaining:]
                        selected_low = np.concatenate([selected_low, np.array(available_indices)[sup_idx]])
                    else:
                        sup_needed = 4 - len(selected_low)
                        mask = ~np.isin(candidate_indices, np.concatenate([selected_high, selected_low]))
                        remaining_candidates = candidate_indices[mask]
                        remaining_scores = candidate_importance[mask]
                        if len(remaining_scores) >= sup_needed:
                            sup_idx = np.argsort(remaining_scores)[-sup_needed:]
                            selected_low = np.concatenate([selected_low, remaining_candidates[sup_idx]])
                        else:
                            selected_low = np.concatenate([selected_low, [-1] * sup_needed])
                else:
                    sup_needed = 4 - len(selected_low)
                    mask = ~np.isin(candidate_indices, np.concatenate([selected_high, selected_low]))
                    remaining_candidates = candidate_indices[mask]
                    remaining_scores = candidate_importance[mask]
                    if len(remaining_scores) >= sup_needed:
                        sup_idx = np.argsort(remaining_scores)[-sup_needed:]
                        selected_low = np.concatenate([selected_low, remaining_candidates[sup_idx]])
                    else:
                        selected_low = np.concatenate([selected_low, [-1] * sup_needed])
            else:
                top_idx = np.argsort(low_uncertain_scores)[-4:]
                selected_low = low_uncertain_indices[top_idx]

            selected_high = selected_high[:4]
            selected_low = selected_low[:4]

            selected_indices.extend(selected_high.tolist())
            uncertainty_labels.extend([0] * len(selected_high))
            selected_indices.extend(selected_low.tolist())
            uncertainty_labels.extend([1] * len(selected_low))

        assert len(selected_indices) == m_groups * patches_per_group

        valid_mask = np.array(selected_indices) != -1
        valid_indices = np.array(selected_indices)[valid_mask]

        key_features = np.zeros((m_groups * patches_per_group, feature_dim), dtype=np.float32)
        if len(valid_indices) > 0:
            key_features[valid_mask] = features_use[valid_indices]

        return key_features, selected_indices, uncertainty_labels, entropy

    except Exception as e:
        print(f"[WARN] Patient {patient_id} failed: {e}")
        return None


def main():
    config = load_config()
    hparams = config["hyperparams"]

    features_h5 = Path(config["files"]["wsi_features"])
    clinical_path = Path(config["files"]["clinical"])
    wsi_delete_path = Path(config["files"]["wsi_delete"])

    output_h5 = Path(config["data"]["processed_pathology"]) / "Cancer_WSI_IU_selected.h5"
    output_meta = Path(config["data"]["processed_pathology"]) / "Cancer_WSI_IU_selection_metadata.csv"
    output_json = Path(config["data"]["processed_pathology"]) / "Cancer_WSI_selected_index.json"

    m_groups = hparams["wsi_m_groups"]
    patches_per_group = hparams["wsi_patches_per_group"]
    feature_dim = 2048

    clinical_df = pd.read_csv(clinical_path)
    wsi_df = pd.read_csv(wsi_delete_path)
    patient_ids = sorted(list(set(wsi_df["patient_id"]) & set(clinical_df["patient_id"])))
    print(f"[INFO] Valid patients: {len(patient_ids)}")

    mil_scorer = MILAttentionScorer(input_dim=feature_dim)
    mil_scorer.eval()
    print(f"[INFO] MIL attention scorer initialized, params: {sum(p.numel() for p in mil_scorer.parameters())}")

    if output_h5.exists():
        output_h5.unlink()

    results = []
    failed_patients = []

    with h5py.File(features_h5, "r") as h5f_in:
        all_patient_ids = [pid.decode() if isinstance(pid, bytes) else pid for pid in h5f_in["patient_ids"][:]]
        patient_ids_filtered = [pid for pid in patient_ids if pid in all_patient_ids]

        print(f"[INFO] Processing patients: {len(patient_ids_filtered)}")

        with h5py.File(output_h5, "w") as h5f_out:
            h5f_out.attrs["creation_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            h5f_out.attrs["description"] = "WSI 24 key patches (alternating high/low uncertain)"
            h5f_out.attrs["m_groups"] = m_groups
            h5f_out.attrs["patches_per_group"] = patches_per_group
            h5f_out.attrs["total_patches_per_patient"] = m_groups * patches_per_group
            h5f_out.attrs["feature_dim"] = feature_dim
            h5f_out.create_dataset("patient_ids", data=np.array(patient_ids_filtered, dtype="S32"))

            for patient_id in tqdm(patient_ids_filtered, desc="Processing patients"):
                result = select_24_patches(patient_id, h5f_in, mil_scorer, m_groups, patches_per_group, feature_dim)

                if result is not None:
                    key_features, selected_indices, uncertainty_labels, entropy = result

                    patient_group = h5f_out.create_group(patient_id)
                    dset = patient_group.create_dataset("features", data=key_features, compression="gzip", compression_opts=4)
                    dset.attrs["n_patches"] = key_features.shape[0]
                    dset.attrs["feature_dim"] = key_features.shape[1]

                    patient_group.create_dataset("original_indices", data=np.array(selected_indices))
                    patient_group.create_dataset("uncertainty_labels", data=np.array(uncertainty_labels))
                    patient_group.create_dataset("entropy_values", data=entropy)

                    n_valid = sum(1 for idx in selected_indices if idx != -1)
                    results.append({
                        "patient_id": patient_id,
                        "status": "SUCCESS",
                        "n_selected": len(selected_indices),
                        "n_valid": n_valid,
                        "n_high_uncertain": sum(1 for x in uncertainty_labels if x == 0),
                        "n_low_uncertain": sum(1 for x in uncertainty_labels if x == 1),
                    })
                else:
                    results.append({
                        "patient_id": patient_id,
                        "status": "FAILED",
                        "n_selected": 0,
                        "n_valid": 0,
                        "n_high_uncertain": 0,
                        "n_low_uncertain": 0,
                    })
                    failed_patients.append(patient_id)

            h5f_out.attrs["total_patients"] = len(patient_ids_filtered)
            h5f_out.attrs["successful_patients"] = len([r for r in results if r["status"] == "SUCCESS"])

    success_df = pd.DataFrame([r for r in results if r["status"] == "SUCCESS"])
    failed_count = len([r for r in results if r["status"] == "FAILED"])
    print(f"[INFO] Success: {len(success_df)}, Failed: {failed_count}")

    metadata_df = pd.DataFrame(results)
    metadata_df.to_csv(output_meta, index=False)
    print(f"[INFO] Saved Cancer_WSI_IU_selection_metadata.csv")

    index_data = {
        "file_path": str(output_h5),
        "feature_dim": feature_dim,
        "patches_per_patient": m_groups * patches_per_group,
        "patient_ids": patient_ids_filtered,
        "metadata_csv": str(output_meta),
        "attributes": {
            "m_groups": m_groups,
            "patches_per_group": patches_per_group,
            "order": "alternating: groupX_high_uncertain (4), groupX_low_uncertain (4)",
            "padding_value": -1,
        },
    }
    with open(output_json, "w") as f:
        json.dump(index_data, f, indent=2)
    print(f"[INFO] Saved Cancer_WSI_selected_index.json")
    print("[INFO] Done")


if __name__ == "__main__":
    main()