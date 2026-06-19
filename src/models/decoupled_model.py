# src/models/decoupled_model.py
# Noise-aware disentangled representation learning.

import sys
import json
import time
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.io_utils import load_config, get_device, set_seed
from utils.losses import DecouplingLoss

warnings.filterwarnings("ignore")


class FiLM(nn.Module):
    """Feature-wise Linear Modulation conditioned on noise embedding."""

    def __init__(self, feature_dim: int, noise_emb_dim: int = 32):
        super().__init__()
        self.gamma_net = nn.Sequential(
            nn.Linear(noise_emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
        )
        self.beta_net = nn.Sequential(
            nn.Linear(noise_emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
        )

    def forward(self, x: torch.Tensor, noise_emb: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_net(noise_emb).unsqueeze(1)
        beta = self.beta_net(noise_emb).unsqueeze(1)
        return gamma * x + beta


class WSIEncoder(nn.Module):
    """WSI encoder: [24, 2048] -> representation."""

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 512,
                 num_layers: int = 4, num_heads: int = 8, noise_emb_dim: int = 32):
        super().__init__()
        self.patch_projection = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.film = FiLM(hidden_dim, noise_emb_dim)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
        )

    def forward(self, x: torch.Tensor, noise_emb: torch.Tensor) -> torch.Tensor:
        x = self.patch_projection(x)
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device).float().unsqueeze(0).unsqueeze(2) * 0.01
        x = x + pos

        for i, layer in enumerate(self.transformer.layers):
            x = layer(x)
            if i == 1:
                x = self.film(x, noise_emb)

        attn_weights = F.softmax(torch.matmul(x, x.transpose(-2, -1)), dim=-1)
        x = torch.matmul(attn_weights.mean(dim=1, keepdim=True), x).squeeze(1)
        return self.projector(x)


class mRNAEncoder(nn.Module):
    """mRNA encoder: [400] -> representation."""

    def __init__(self, input_dim: int = 400, hidden_dim: int = 512, noise_emb_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.film = FiLM(hidden_dim, noise_emb_dim)
        self.projector = nn.Linear(hidden_dim, 1024)

    def forward(self, x: torch.Tensor, noise_emb: torch.Tensor) -> torch.Tensor:
        x = self.mlp[0](x)
        x = self.mlp[1](x)
        x = self.mlp[2](x)
        x = self.mlp[3](x)
        x = self.mlp[4](x)
        x = self.mlp[5](x)
        x = self.film(x.unsqueeze(1), noise_emb).squeeze(1)
        x = self.mlp[6](x)
        x = self.mlp[7](x)
        x = self.mlp[8](x)

        attn_weights = F.softmax(self.attention(x), dim=1)
        x = x * attn_weights
        return self.projector(x)


class WSIDecoder(nn.Module):
    def __init__(self, private_dim: int = 256, output_dim: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(private_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, z_private: torch.Tensor) -> torch.Tensor:
        return self.net(z_private)


class mRNADecoder(nn.Module):
    def __init__(self, private_dim: int = 256, output_dim: int = 400):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(private_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, z_private: torch.Tensor) -> torch.Tensor:
        return self.net(z_private)


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GRL(nn.Module):
    def __init__(self, lambda_: float = 0.1):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class DecoupledModel(nn.Module):
    def __init__(self, wsi_input_dim: int = 2048, mrna_input_dim: int = 400):
        super().__init__()
        self.common_dim = 256
        self.private_dim = 256

        self.wsi_encoder = WSIEncoder(wsi_input_dim, noise_emb_dim=32)
        self.mrna_encoder = mRNAEncoder(mrna_input_dim, noise_emb_dim=32)

        self.wsi_common_head = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, self.common_dim))
        self.wsi_private_head = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, self.private_dim))
        self.mrna_common_head = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, self.common_dim))
        self.mrna_private_head = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, self.private_dim))

        self.wsi_decoder = WSIDecoder(self.private_dim, wsi_input_dim)
        self.mrna_decoder = mRNADecoder(self.private_dim, mrna_input_dim)

        self.grl = GRL(lambda_=0.1)
        self.grl_classifier = nn.Sequential(nn.Linear(self.common_dim, 128), nn.ReLU(), nn.Linear(128, 4))
        self.noise_embedding = nn.Embedding(4, 32)

    def forward(self, wsi_features, mrna_expr, wsi_label, mrna_label):
        wsi_noise_emb = self.noise_embedding(wsi_label)
        mrna_noise_emb = self.noise_embedding(mrna_label)

        z_wsi_enc = self.wsi_encoder(wsi_features, wsi_noise_emb)
        z_mrna_enc = self.mrna_encoder(mrna_expr, mrna_noise_emb)

        z_wsi_common = self.wsi_common_head(z_wsi_enc)
        z_wsi_private = self.wsi_private_head(z_wsi_enc)
        z_mrna_common = self.mrna_common_head(z_mrna_enc)
        z_mrna_private = self.mrna_private_head(z_mrna_enc)

        z_common = (z_wsi_common + z_mrna_common) / 2

        z_common_grl = self.grl(z_common)
        pseudo_pred = self.grl_classifier(z_common_grl)

        wsi_recon = self.wsi_decoder(z_wsi_private)
        mrna_recon = self.mrna_decoder(z_mrna_private)

        return {
            "z_common": z_common,
            "z_wsi_private": z_wsi_private,
            "z_mrna_private": z_mrna_private,
            "pseudo_pred": pseudo_pred,
            "wsi_recon": wsi_recon,
            "mrna_recon": mrna_recon,
            "z_wsi_common": z_wsi_common,
            "z_mrna_common": z_mrna_common,
        }


class HighFrequencyExtractor:
    """Extract high-frequency component from 24 patches (CPU)."""

    def __call__(self, patches: torch.Tensor) -> torch.Tensor:
        mean_feat = patches.mean(dim=1, keepdim=True)
        distances = torch.norm(patches - mean_feat, p=2, dim=2)
        weights = F.softmax(distances, dim=1)
        high_freq = torch.sum(patches * weights.unsqueeze(2), dim=1)
        return high_freq


class DecouplingDataset(Dataset):
    def __init__(self, wsi_h5_path, mrna_csv_path, wsi_noise_h5, mrna_noise_h5):
        self.wsi_h5_path = wsi_h5_path
        self.mrna_csv_path = mrna_csv_path
        self.wsi_noise_h5 = wsi_noise_h5
        self.mrna_noise_h5 = mrna_noise_h5
        self.wsi_h5 = None

        with h5py.File(wsi_h5_path, "r") as f:
            self.patient_ids = [pid.decode() for pid in f["patient_ids"][:]]

        mrna_df = pd.read_csv(mrna_csv_path)
        self.mrna_data = {
            row["patient_id"]: row.drop("patient_id").values.astype(np.float32)
            for _, row in mrna_df.iterrows()
        }

        self.wsi_noise = {}
        with h5py.File(wsi_noise_h5, "r") as f:
            for pid in self.patient_ids:
                if pid in f:
                    self.wsi_noise[pid] = {
                        "label": int(f[pid].attrs.get("noise_label", 2)),
                        "score": float(f[pid].attrs.get("importance_score", 0.5)),
                    }
                else:
                    self.wsi_noise[pid] = {"label": 2, "score": 0.5}

        self.mrna_noise = {}
        with h5py.File(mrna_noise_h5, "r") as f:
            for pid in self.patient_ids:
                if pid in f:
                    self.mrna_noise[pid] = {
                        "label": int(f[pid].attrs.get("pseudo_label", 3)),
                        "score": float(f[pid].attrs.get("importance_score", 0.5)),
                    }
                else:
                    self.mrna_noise[pid] = {"label": 3, "score": 0.5}

        self.high_freq_extractor = HighFrequencyExtractor()
        print(f"[INFO] Dataset initialized: {len(self.patient_ids)} patients")

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx: int):
        if self.wsi_h5 is None:
            self.wsi_h5 = h5py.File(self.wsi_h5_path, "r")

        pid = self.patient_ids[idx]
        wsi_features = self.wsi_h5[pid]["features"][:]
        wsi_features = torch.from_numpy(wsi_features).float()

        wsi_high_freq = self.high_freq_extractor(wsi_features.unsqueeze(0)).squeeze(0)
        mrna_expr = torch.from_numpy(self.mrna_data[pid]).float()

        wsi_label = torch.tensor(self.wsi_noise[pid]["label"], dtype=torch.long)
        mrna_label = torch.tensor(self.mrna_noise[pid]["label"], dtype=torch.long)
        noise_scores = torch.tensor([
            self.wsi_noise[pid]["score"],
            self.mrna_noise[pid]["score"],
        ], dtype=torch.float32)

        return {
            "wsi_features": wsi_features,
            "wsi_high_freq": wsi_high_freq,
            "mrna_expr": mrna_expr,
            "wsi_label": wsi_label,
            "mrna_label": mrna_label,
            "noise_scores": noise_scores,
            "patient_id": pid,
        }

    def close(self):
        if self.wsi_h5 is not None:
            self.wsi_h5.close()
            self.wsi_h5 = None


class DecouplingTrainer:
    def __init__(self, dataset, model, criterion, batch_size: int = 32):
        self.dataset = dataset
        self.model = model
        self.criterion = criterion
        self.batch_size = batch_size

        self.dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )
        self.history = []

    def train(self, epochs: int = 80, patience: int = 15):
        print("[INFO] Starting training...")
        best_loss = float("inf")
        no_improve = 0

        for epoch in range(epochs):
            epoch_losses = defaultdict(float)
            self.model.train()

            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                batch = {
                    k: v.to(get_device()) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                outputs = self.model(
                    batch["wsi_features"], batch["mrna_expr"],
                    batch["wsi_label"], batch["mrna_label"]
                )
                loss, loss_dict = self.criterion(outputs, batch)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                for k, v in loss_dict.items():
                    epoch_losses[k] += v
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_losses = {k: v / len(self.dataloader) for k, v in epoch_losses.items()}
            self.history.append(avg_losses)

            print(f"[INFO] Epoch {epoch+1} | " + " | ".join([f"{k}: {v:.4f}" for k, v in avg_losses.items()]))
            self.scheduler.step(avg_losses["total"])

            if avg_losses["total"] < best_loss:
                best_loss = avg_losses["total"]
                no_improve = 0
                torch.save(self.model.state_dict(), self._checkpoint_path())
            else:
                no_improve += 1

            if no_improve >= patience:
                print("[INFO] Early stopping triggered")
                break

        return self.history

    def _checkpoint_path(self):
        config = load_config()
        return Path(config["data"]["decoupled"]) / "Cancer_best_model.pth"

    def extract_representations(self):
        print("[INFO] Extracting decoupled representations...")
        self.dataset.close()
        time.sleep(1)

        extract_loader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=0, pin_memory=True,
        )

        self.model.load_state_dict(torch.load(self._checkpoint_path()))
        self.model.eval()

        config = load_config()
        output_h5 = Path(config["data"]["decoupled"]) / "Cancer_decoupled_representations.h5"
        if output_h5.exists():
            output_h5.unlink()
            time.sleep(0.5)

        with h5py.File(output_h5, "w") as h5f:
            h5f.attrs["creation_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            h5f.attrs["description"] = "Noise-aware disentangled representations"
            h5f.attrs["common_dim"] = self.model.common_dim
            h5f.attrs["private_dim"] = self.model.private_dim

            with torch.no_grad():
                for batch in tqdm(extract_loader, desc="Extracting"):
                    wsi = batch["wsi_features"].to(get_device())
                    mrna = batch["mrna_expr"].to(get_device())
                    wsi_label = batch["wsi_label"].to(get_device())
                    mrna_label = batch["mrna_label"].to(get_device())
                    pids = batch["patient_id"]

                    outputs = self.model(wsi, mrna, wsi_label, mrna_label)

                    for i, pid in enumerate(pids):
                        grp = h5f.create_group(pid)
                        grp.create_dataset("z_common", data=outputs["z_common"][i].cpu().numpy())
                        grp.create_dataset("z_wsi_private", data=outputs["z_wsi_private"][i].cpu().numpy())
                        grp.create_dataset("z_mrna_private", data=outputs["z_mrna_private"][i].cpu().numpy())
                        grp.attrs["wsi_label"] = int(wsi_label[i].cpu())
                        grp.attrs["mrna_label"] = int(mrna_label[i].cpu())

            h5f.create_dataset("patient_ids", data=np.array(self.dataset.patient_ids, dtype="S32"))

        # Metadata JSON
        metadata = {
            "model_config": {
                "common_dim": self.model.common_dim,
                "private_dim": self.model.private_dim,
                "hidden_dim": 512,
            },
            "training": {
                "epochs": len(self.history),
                "best_loss": best_loss if "best_loss" in dir() else None,
                "batch_size": self.batch_size,
                "patients": len(self.dataset),
            },
            "outputs": {
                "representations": str(output_h5),
                "checkpoint": str(self._checkpoint_path()),
            },
            "loss_history": self.history,
        }
        meta_json = Path(config["data"]["decoupled"]) / "Cancer_decoupled_metadata.json"
        with open(meta_json, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"[INFO] Saved representations: {output_h5} ({output_h5.stat().st_size / 1e6:.1f} MB)")
        print(f"[INFO] Saved checkpoint: {self._checkpoint_path()}")
        return output_h5


def main():
    print("[INFO] Noise-aware disentangled representation learning")
    set_seed(42)
    config = load_config()

    wsi_h5 = Path(config["data"]["processed_pathology"]) / "Cancer_WSI_IU_selected.h5"
    mrna_csv = Path(config["data"]["processed_genomics"]) / "Cancer_mRNA_IU_selected.csv"
    wsi_noise = Path(config["data"]["processed_pathology"]) / "Cancer_patches_IU.h5"
    mrna_noise = Path(config["data"]["processed_genomics"]) / "Cancer_mRNA_noise_IU_scores.h5"

    for p in [wsi_h5, mrna_csv, wsi_noise, mrna_noise]:
        if not p.exists():
            print(f"[ERROR] Missing input: {p}")
            return

    device = get_device()
    if torch.cuda.is_available():
        print(f"[INFO] GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    dataset = DecouplingDataset(str(wsi_h5), str(mrna_csv), str(wsi_noise), str(mrna_noise))
    model = DecoupledModel().to(device)
    criterion = DecouplingLoss(lambda_common=1.0, lambda_private=1.0, lambda_grl=0.1, lambda_recon=1.0)
    trainer = DecouplingTrainer(dataset, model, criterion, batch_size=32)

    try:
        trainer.train(epochs=80, patience=15)
        trainer.extract_representations()
        print("[INFO] Training complete")
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        dataset.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()