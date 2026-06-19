# src/utils/losses.py
# Loss functions for decoupled representation learning and survival analysis.

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecouplingLoss(nn.Module):
    """
    Combined loss for noise-aware disentanglement:
    alignment, adversarial (GRL), separation, and reconstruction.
    """

    def __init__(
        self,
        lambda_common: float = 1.0,
        lambda_private: float = 1.0,
        lambda_grl: float = 0.1,
        lambda_recon: float = 1.0,
    ):
        super().__init__()
        self.lambda_common = lambda_common
        self.lambda_private = lambda_private
        self.lambda_grl = lambda_grl
        self.lambda_recon = lambda_recon

    def forward(self, model_out: dict, batch: dict) -> tuple:
        z_common = model_out["z_common"]
        z_wsi_p = model_out["z_wsi_private"]
        z_mrna_p = model_out["z_mrna_private"]
        pseudo_pred = model_out["pseudo_pred"]
        wsi_recon = model_out["wsi_recon"]
        mrna_recon = model_out["mrna_recon"]

        wsi_features = batch["wsi_features"]
        mrna_expr = batch["mrna_expr"]
        wsi_label = batch["wsi_label"]
        mrna_label = batch["mrna_label"]

        loss_dict = {}
        total_loss = torch.tensor(0.0, device=wsi_features.device)

        # Reconstruction loss
        wsi_target = batch["wsi_high_freq"]
        loss_recon_wsi = F.mse_loss(wsi_recon, wsi_target)
        loss_recon_mrna = F.mse_loss(mrna_recon, mrna_expr)
        loss_dict["recon_wsi"] = loss_recon_wsi.item()
        loss_dict["recon_mrna"] = loss_recon_mrna.item()
        total_loss += self.lambda_recon * (loss_recon_wsi + loss_recon_mrna)

        # Clean confidence scores
        wsi_clean_score, wsi_noisy_score = self._get_label_confidence(wsi_label, "wsi")
        mrna_clean_score, mrna_noisy_score = self._get_label_confidence(mrna_label, "mrna")

        # Alignment loss (weighted)
        align_weights = wsi_clean_score * mrna_clean_score
        z_wsi_common = model_out["z_wsi_common"]
        z_mrna_common = model_out["z_mrna_common"]
        align_loss_per_sample = F.mse_loss(z_wsi_common, z_mrna_common, reduction="none").mean(dim=1)
        loss_align = (align_weights * align_loss_per_sample).mean()
        loss_dict["align"] = loss_align.item()
        total_loss += self.lambda_common * loss_align

        # Separation loss (weighted)
        separate_weights = wsi_noisy_score * mrna_noisy_score
        cos_sim = F.cosine_similarity(z_wsi_p, z_mrna_p, dim=1)
        loss_separate = (separate_weights * cos_sim).mean()
        loss_dict["separate"] = loss_separate.item()
        total_loss += self.lambda_private * loss_separate

        # GRL adversarial loss (weighted)
        grl_weights = (wsi_clean_score + mrna_clean_score) / 2.0
        loss_grl_per_sample = F.cross_entropy(pseudo_pred, wsi_label, reduction="none")
        loss_grl = (grl_weights * loss_grl_per_sample).mean()
        loss_dict["grl"] = loss_grl.item()
        total_loss += self.lambda_grl * loss_grl

        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict

    @staticmethod
    def _get_label_confidence(labels: torch.Tensor, modality: str) -> tuple:
        if modality == "wsi":
            clean = torch.tensor([0.0, 1.0, 0.5, 0.0], device=labels.device)
            noisy = torch.tensor([1.0, 0.0, 0.5, 1.0], device=labels.device)
        else:
            clean = torch.tensor([0.0, 0.0, 1.0, 1.0], device=labels.device)
            noisy = torch.tensor([1.0, 1.0, 0.0, 0.0], device=labels.device)
        return clean[labels], noisy[labels]


class CoxLoss(nn.Module):
    """Negative partial log-likelihood for Cox proportional hazards."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        risk_scores: torch.Tensor,
        survival_times: torch.Tensor,
        events: torch.Tensor,
    ) -> torch.Tensor:
        sort_idx = torch.argsort(survival_times, descending=True)
        risk_sorted = risk_scores[sort_idx]
        events_sorted = events[sort_idx]

        loss = torch.tensor(0.0, device=risk_scores.device, dtype=torch.float32)
        n_events = 0

        for i in range(len(risk_sorted)):
            if events_sorted[i] == 1:
                risk_set = torch.exp(risk_sorted[i:])
                loss = loss + risk_sorted[i] - torch.log(torch.sum(risk_set) + 1e-8)
                n_events += 1

        return -loss / (n_events + 1e-8) if n_events > 0 else torch.tensor(0.0, device=risk_scores.device)