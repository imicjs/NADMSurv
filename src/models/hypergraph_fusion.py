# src/models/hypergraph_fusion.py
# Evidential hypergraph fusion for survival prediction with 5-fold cross-validation.

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.io_utils import load_config, get_device, set_seed
from utils.losses import CoxLoss
from utils.hypergraph_utils import build_hypergraph_knn, HypergraphConv

warnings.filterwarnings("ignore")


class HypergraphNetworkWithUncertainty(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(HypergraphConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 2):
            self.convs.append(HypergraphConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.convs.append(HypergraphConv(hidden_dim, 1))
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, X, H, edge_weights):
        for i, conv in enumerate(self.convs[:-1]):
            X = conv(X, H, edge_weights)
            X = self.bns[i](X)
            X = self.relu(X)
            X = self.dropout(X)
        risk_scores = self.convs[-1](X, H, edge_weights)
        return risk_scores.squeeze(1)

    def compute_uncertainty(self, risk_scores: torch.Tensor, base_uncertainty: float = 0.2) -> float:
        variance = torch.var(risk_scores).item()
        uncertainty = base_uncertainty + min(variance * 0.1, 0.3)
        uncertainty = max(0.1, min(0.5, uncertainty))
        return uncertainty


class CrossAttentionFusion(nn.Module):
    def __init__(self, input_dim: int = 256, hidden_dim: int = 64):
        super().__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key_wsi = nn.Linear(input_dim, hidden_dim)
        self.key_mrna = nn.Linear(input_dim, hidden_dim)
        self.value_wsi = nn.Linear(input_dim, input_dim)
        self.value_mrna = nn.Linear(input_dim, input_dim)
        self.scale = hidden_dim ** -0.5

    def forward(self, z_common, z_wsi, z_mrna):
        q = self.query(z_common)
        k_wsi = self.key_wsi(z_wsi)
        v_wsi = self.value_wsi(z_wsi)
        k_mrna = self.key_mrna(z_mrna)
        v_mrna = self.value_mrna(z_mrna)

        attn_wsi = torch.sum(q * k_wsi, dim=1, keepdim=True) * self.scale
        attn_mrna = torch.sum(q * k_mrna, dim=1, keepdim=True) * self.scale

        attn_scores = torch.cat([attn_wsi, attn_mrna], dim=1)
        attn_scores = torch.softmax(attn_scores, dim=1)

        attn_wsi = attn_scores[:, 0:1]
        attn_mrna = attn_scores[:, 1:2]

        fused = attn_wsi * v_wsi + attn_mrna * v_mrna
        return fused, attn_wsi, attn_mrna


class SubjectiveLogicFusion(nn.Module):
    def __init__(self, event_rate: float = 0.5):
        super().__init__()
        self.event_rate = event_rate

    def generate_opinion(self, risk_scores, uncertainty, modality):
        e_risk = torch.sigmoid(risk_scores)
        e_safe = 1.0 - e_risk
        e_total = e_risk + e_safe + 0.1

        b_risk = e_risk / e_total
        b_safe = e_safe / e_total
        u = torch.full_like(b_risk, uncertainty)

        return {
            "b_risk": b_risk,
            "b_safe": b_safe,
            "u": u,
            "a": torch.full_like(b_risk, self.event_rate),
        }

    def expectation_value(self, opinion):
        return opinion["b_risk"] + opinion["u"] * opinion["a"]


class HybridFusionSurvivalModel(nn.Module):
    def __init__(self, input_dim: int = 256, hg_hidden_dim: int = 128,
                 hg_layers: int = 3, fusion_hidden_dim: int = 64,
                 event_rate: float = 0.5, dropout: float = 0.2):
        super().__init__()

        self.hg_common = HypergraphNetworkWithUncertainty(input_dim, hg_hidden_dim, hg_layers, dropout)
        self.hg_wsi = HypergraphNetworkWithUncertainty(input_dim, hg_hidden_dim, hg_layers, dropout)
        self.hg_mrna = HypergraphNetworkWithUncertainty(input_dim, hg_hidden_dim, hg_layers, dropout)

        self.attention_fusion = CrossAttentionFusion(input_dim, fusion_hidden_dim)
        self.sl_fusion = SubjectiveLogicFusion(event_rate)

        self.risk_predictor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self.last_attn_weights = {}
        self.last_opinions = {}

    def forward(self, z_common, z_wsi, z_mrna, H_common, H_wsi, H_mrna,
                ew_common, ew_wsi, ew_mrna, times=None, events=None):
        risk_common = self.hg_common(z_common, H_common, ew_common)
        risk_wsi = self.hg_wsi(z_wsi, H_wsi, ew_wsi)
        risk_mrna = self.hg_mrna(z_mrna, H_mrna, ew_mrna)

        u_common = self.hg_common.compute_uncertainty(risk_common, base_uncertainty=0.15)
        u_wsi = self.hg_wsi.compute_uncertainty(risk_wsi, base_uncertainty=0.20)
        u_mrna = self.hg_mrna.compute_uncertainty(risk_mrna, base_uncertainty=0.20)

        op_common = self.sl_fusion.generate_opinion(risk_common, u_common, "common")
        op_wsi = self.sl_fusion.generate_opinion(risk_wsi, u_wsi, "wsi")
        op_mrna = self.sl_fusion.generate_opinion(risk_mrna, u_mrna, "mrna")

        fused_features, attn_wsi, attn_mrna = self.attention_fusion(z_common, z_wsi, z_mrna)
        risk_fused = self.risk_predictor(fused_features).squeeze(1)

        weighted_risk = (
            risk_fused * 0.5
            + op_common["b_risk"] * (1 - u_common) * 0.3
            + op_wsi["b_risk"] * attn_wsi.squeeze() * (1 - u_wsi) * 0.1
            + op_mrna["b_risk"] * attn_mrna.squeeze() * (1 - u_mrna) * 0.1
        )

        self.last_attn_weights = {
            "wsi": attn_wsi.mean().item(),
            "mrna": attn_mrna.mean().item(),
        }
        self.last_opinions = {
            "common": op_common,
            "wsi": op_wsi,
            "mrna": op_mrna,
        }

        return weighted_risk


class SurvivalDataset(Dataset):
    def __init__(self, rep_h5_path: Path, clinical_path: Path):
        self.rep_h5_path = rep_h5_path

        clinical_df = pd.read_csv(clinical_path)
        self.survival_data = {}
        for _, row in clinical_df.iterrows():
            pid = str(row["patient_id"])
            self.survival_data[pid] = {
                "time": float(row["os_time"]),
                "event": int(row["os_status"] == "Dead" or row["os_status"] == 1),
            }

        with h5py.File(rep_h5_path, "r") as f:
            patient_ids_raw = f["patient_ids"][:]
            self.patient_ids = []
            for pid in patient_ids_raw:
                if isinstance(pid, bytes):
                    self.patient_ids.append(pid.decode("utf-8"))
                else:
                    self.patient_ids.append(str(pid))

            self.z_common = {}
            self.z_wsi_private = {}
            self.z_mrna_private = {}

            for pid in self.patient_ids:
                if pid in f and pid in self.survival_data:
                    try:
                        self.z_common[pid] = f[pid]["z_common"][:].astype(np.float32)
                        self.z_wsi_private[pid] = f[pid]["z_wsi_private"][:].astype(np.float32)
                        self.z_mrna_private[pid] = f[pid]["z_mrna_private"][:].astype(np.float32)
                    except KeyError:
                        continue

        self.valid_pids = [pid for pid in self.patient_ids if pid in self.survival_data and pid in self.z_common]

    def __len__(self):
        return len(self.valid_pids)

    def __getitem__(self, idx: int):
        pid = self.valid_pids[idx]
        return {
            "z_common": self.z_common[pid],
            "z_wsi_private": self.z_wsi_private[pid],
            "z_mrna_private": self.z_mrna_private[pid],
            "survival_time": self.survival_data[pid]["time"],
            "event": self.survival_data[pid]["event"],
            "patient_id": pid,
        }

    def get_features_array(self):
        z_common = [self.z_common[pid] for pid in self.valid_pids]
        z_wsi = [self.z_wsi_private[pid] for pid in self.valid_pids]
        z_mrna = [self.z_mrna_private[pid] for pid in self.valid_pids]
        return np.stack(z_common), np.stack(z_wsi), np.stack(z_mrna)

    def get_survival_labels(self):
        times = [self.survival_data[pid]["time"] for pid in self.valid_pids]
        events = [self.survival_data[pid]["event"] for pid in self.valid_pids]
        return np.array(times, dtype=np.float32), np.array(events, dtype=np.int32)


class HybridSurvivalCVTrainer:
    def __init__(self, dataset, batch_size=32, lr=0.001, k=10,
                 epochs=100, patience=10, lambda_uncertainty=0.01):
        self.dataset = dataset
        self.batch_size = batch_size
        self.lr = lr
        self.k = k
        self.epochs = epochs
        self.patience = patience
        self.lambda_uncertainty = lambda_uncertainty

        self.valid_pids = dataset.valid_pids
        self.z_common_all, self.z_wsi_all, self.z_mrna_all = dataset.get_features_array()
        self.times_all, self.events_all = dataset.get_survival_labels()

        if len(self.z_common_all) == 0:
            raise ValueError("No valid patient data")

        self.kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        self.criterion = CoxLoss()
        self.results = []

    def compute_cindex(self, risk_scores, times, events):
        risk_np = risk_scores.detach().cpu().numpy()
        times_np = times.detach().cpu().numpy()
        events_np = events.detach().cpu().numpy()

        n = len(risk_np)
        concordant = 0
        permissible = 0

        for i in range(n):
            if events_np[i] == 1:
                for j in range(n):
                    if times_np[j] > times_np[i] or (times_np[j] == times_np[i] and events_np[j] == 0):
                        permissible += 1
                        if risk_np[i] > risk_np[j]:
                            concordant += 1

        return concordant / (permissible + 1e-8) if permissible > 0 else 0.5

    def train_fold(self, train_idx, val_idx, fold):
        z_common_train = self.z_common_all[train_idx]
        z_wsi_train = self.z_wsi_all[train_idx]
        z_mrna_train = self.z_mrna_all[train_idx]
        z_common_val = self.z_common_all[val_idx]
        z_wsi_val = self.z_wsi_all[val_idx]
        z_mrna_val = self.z_mrna_all[val_idx]

        times_train = self.times_all[train_idx]
        events_train = self.events_all[train_idx]
        times_val = self.times_all[val_idx]
        events_val = self.events_all[val_idx]

        H_train_c, ew_train_c = build_hypergraph_knn(z_common_train, self.k)
        H_train_w, ew_train_w = build_hypergraph_knn(z_wsi_train, self.k)
        H_train_m, ew_train_m = build_hypergraph_knn(z_mrna_train, self.k)

        H_val_c, ew_val_c = build_hypergraph_knn(z_common_val, self.k)
        H_val_w, ew_val_w = build_hypergraph_knn(z_wsi_val, self.k)
        H_val_m, ew_val_m = build_hypergraph_knn(z_mrna_val, self.k)

        device = get_device()

        z_common_train_t = torch.from_numpy(z_common_train).float().to(device)
        z_wsi_train_t = torch.from_numpy(z_wsi_train).float().to(device)
        z_mrna_train_t = torch.from_numpy(z_mrna_train).float().to(device)
        times_train_t = torch.from_numpy(times_train).float().to(device)
        events_train_t = torch.from_numpy(events_train).float().to(device)

        z_common_val_t = torch.from_numpy(z_common_val).float().to(device)
        z_wsi_val_t = torch.from_numpy(z_wsi_val).float().to(device)
        z_mrna_val_t = torch.from_numpy(z_mrna_val).float().to(device)
        times_val_t = torch.from_numpy(times_val).float().to(device)
        events_val_t = torch.from_numpy(events_val).float().to(device)

        model = HybridFusionSurvivalModel(
            input_dim=256, hg_hidden_dim=128, hg_layers=3,
            fusion_hidden_dim=64, event_rate=self.dataset.event_rate,
            dropout=0.2,
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, verbose=False
        )

        best_cindex = 0.0
        no_improve = 0
        history = []

        for epoch in range(self.epochs):
            model.train()
            risk_train = model(
                z_common_train_t, z_wsi_train_t, z_mrna_train_t,
                H_train_c, H_train_w, H_train_m,
                ew_train_c, ew_train_w, ew_train_m,
                times_train_t, events_train_t,
            )
            loss = self.criterion(risk_train, times_train_t, events_train_t)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            model.eval()
            with torch.no_grad():
                risk_val = model(
                    z_common_val_t, z_wsi_val_t, z_mrna_val_t,
                    H_val_c, H_val_w, H_val_m,
                    ew_val_c, ew_val_w, ew_val_m,
                    times_val_t, events_val_t,
                )
                cindex_val = self.compute_cindex(risk_val, times_val_t, events_val_t)

            history.append({"epoch": epoch, "train_loss": loss.item(), "val_cindex": cindex_val})

            scheduler.step(cindex_val)

            if cindex_val > best_cindex + 0.001:
                best_cindex = cindex_val
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.patience:
                break

        final_attn = model.last_attn_weights
        final_opinions = {
            name: {
                "b_risk": op["b_risk"].mean().item(),
                "uncertainty": op["u"].mean().item(),
            } for name, op in model.last_opinions.items()
        }

        return {
            "fold": fold,
            "best_cindex": best_cindex,
            "history": history,
            "final_attn_weights": final_attn,
            "final_opinions": final_opinions,
        }

    def run_cv(self):
        print("[INFO] Starting 5-fold cross-validation")
        
        for fold, (train_idx, val_idx) in enumerate(self.kfold.split(self.valid_pids)):
            try:
                result = self.train_fold(train_idx, val_idx, fold)
                self.results.append(result)
                print(f"[INFO] Fold {fold}/5, best C-index: {result['best_cindex']:.4f}")
            except Exception as e:
                print(f"[ERROR] Fold {fold} failed: {e}")
                import traceback
                traceback.print_exc()
                continue

        if not self.results:
            raise RuntimeError("All folds failed")

        cindices = [r['best_cindex'] for r in self.results]
        attn_weights = [r['final_attn_weights'] for r in self.results]

        avg_opinions = {}
        for key in ["common", "wsi", "mrna"]:
            avg_opinions[key] = {
                "b_risk": np.mean([r["final_opinions"][key]["b_risk"] for r in self.results]),
                "uncertainty": np.mean([r["final_opinions"][key]["uncertainty"] for r in self.results]),
            }

        avg_attn = {
            "wsi": np.mean([w["wsi"] for w in attn_weights]),
            "mrna": np.mean([w["mrna"] for w in attn_weights]),
        }

        summary = {
            "fold_results": self.results,
            "mean_cindex": float(np.mean(cindices)),
            "std_cindex": float(np.std(cindices)),
            "all_cindices": cindices,
            "avg_opinions": avg_opinions,
            "avg_attn_weights": avg_attn,
        }

        print("[INFO] Cross-validation complete")
        print(f"[RESULT] Mean C-index: {summary['mean_cindex']:.4f} +/- {summary['std_cindex']:.4f}")
        print(f"[RESULT] All folds: {cindices}")
        print(f"[RESULT] Average attention WSI: {avg_attn['wsi']:.2f}, mRNA: {avg_attn['mrna']:.2f}")
        
        common_b = avg_opinions['common']['b_risk']
        common_u = avg_opinions['common']['uncertainty']
        wsi_b = avg_opinions['wsi']['b_risk']
        wsi_u = avg_opinions['wsi']['uncertainty']
        mrna_b = avg_opinions['mrna']['b_risk']
        mrna_u = avg_opinions['mrna']['uncertainty']
        print(f"[RESULT] Average belief common: {common_b:.2f} (u={common_u:.2f}), WSI: {wsi_b:.2f} (u={wsi_u:.2f}), mRNA: {mrna_b:.2f} (u={mrna_u:.2f})")

        return summary


def main():
    print("[INFO] Evidential hypergraph fusion for survival prediction")
    set_seed(42)
    config = load_config()

    rep_h5 = Path(config["data"]["decoupled"]) / "Cancer_decoupled_representations.h5"
    clinical_path = Path(config["files"]["clinical"])

    for p in [rep_h5, clinical_path]:
        if not p.exists():
            print(f"[ERROR] Missing: {p}")
            return

    print("[INFO] Loading representations")
    dataset = SurvivalDataset(rep_h5, clinical_path)
    print(f"[INFO] Loading survival labels")
    print(f"[INFO] Valid patients for survival analysis: {len(dataset)}")
    
    if len(dataset) == 0:
        print("[ERROR] No valid patients")
        return

    trainer = HybridSurvivalCVTrainer(
        dataset=dataset,
        batch_size=32,
        lr=0.001,
        k=10,
        epochs=100,
        patience=10,
        lambda_uncertainty=0.01,
    )

    try:
        results = trainer.run_cv()
        print("[INFO] Done")
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()