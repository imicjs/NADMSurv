import torch
from torch import nn
from torch.nn import functional as F

class DecouplingLoss(nn.Module):

    def __init__(self, lambda_common: float=1.0, lambda_private: float=1.0, lambda_grl: float=0.1, lambda_recon: float=1.0):
        super().__init__()
        self.lambda_common = lambda_common
        self.lambda_private = lambda_private
        self.lambda_grl = lambda_grl
        self.lambda_recon = lambda_recon

    def forward(self, model_out: dict, batch: dict) -> tuple:
        z_common = model_out['z_common']
        z_wsi_p = model_out['z_wsi_private']
        z_mrna_p = model_out['z_mrna_private']
        pseudo_pred = model_out['pseudo_pred']
        wsi_recon = model_out['wsi_recon']
        mrna_recon = model_out['mrna_recon']
        wsi_features = batch['wsi_features']
        mrna_expr = batch['mrna_expr']
        wsi_label = batch['wsi_label']
        mrna_label = batch['mrna_label']
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=wsi_features.device)
        wsi_target = batch['wsi_high_freq']
        loss_recon_wsi = F.mse_loss(wsi_recon, wsi_target)
        loss_recon_mrna = F.mse_loss(mrna_recon, mrna_expr)
        loss_dict['recon_wsi'] = loss_recon_wsi.item()
        loss_dict['recon_mrna'] = loss_recon_mrna.item()
        total_loss += self.lambda_recon * (loss_recon_wsi + loss_recon_mrna)
        (wsi_clean_score, wsi_noisy_score) = self._get_label_confidence(wsi_label, 'wsi')
        (mrna_clean_score, mrna_noisy_score) = self._get_label_confidence(mrna_label, 'mrna')
        align_weights = wsi_clean_score * mrna_clean_score
        z_wsi_common = model_out['z_wsi_common']
        z_mrna_common = model_out['z_mrna_common']
        align_loss_per_sample = F.mse_loss(z_wsi_common, z_mrna_common, reduction='none').mean(dim=1)
        loss_align = (align_weights * align_loss_per_sample).mean()
        loss_dict['align'] = loss_align.item()
        total_loss += self.lambda_common * loss_align
        separate_weights = wsi_noisy_score * mrna_noisy_score
        cos_sim = F.cosine_similarity(z_wsi_p, z_mrna_p, dim=1)
        loss_separate = (separate_weights * cos_sim).mean()
        loss_dict['separate'] = loss_separate.item()
        total_loss += self.lambda_private * loss_separate
        grl_weights = (wsi_clean_score + mrna_clean_score) / 2.0
        loss_grl_per_sample = F.cross_entropy(pseudo_pred, wsi_label, reduction='none')
        loss_grl = (grl_weights * loss_grl_per_sample).mean()
        loss_dict['grl'] = loss_grl.item()
        total_loss += self.lambda_grl * loss_grl
        loss_dict['total'] = total_loss.item()
        return (total_loss, loss_dict)

    @staticmethod
    def _get_label_confidence(labels: torch.Tensor, modality: str) -> tuple:
        if modality == 'wsi':
            clean = torch.tensor([0.0, 1.0, 0.5, 0.0], device=labels.device)
            noisy = torch.tensor([1.0, 0.0, 0.5, 1.0], device=labels.device)
        else:
            clean = torch.tensor([0.0, 0.0, 1.0, 1.0], device=labels.device)
            noisy = torch.tensor([1.0, 1.0, 0.0, 0.0], device=labels.device)
        return (clean[labels], noisy[labels])

def cox_breslow(log_risk, time, event):
    (r, t, e) = (log_risk.reshape(-1), time.reshape(-1), event.reshape(-1))
    if not len(r) == len(t) == len(e) or not len(r):
        raise ValueError('Invalid survival array lengths')
    if not (torch.isfinite(r).all() and torch.isfinite(t).all() and torch.isfinite(e).all() and (t > 0).all() and ((e == 0) | (e == 1)).all()):
        raise ValueError('Invalid survival values')
    if e.sum() == 0:
        return r.sum() * 0
    terms = []
    for ti in torch.unique(t[e == 1]):
        death = (t == ti) & (e == 1)
        terms.append(r[death].sum() - death.sum() * torch.logsumexp(r[t >= ti], dim=0))
    return -torch.stack(terms).sum() / e.sum()

class CoxLoss(nn.Module):

    def forward(self, risk_scores, survival_times, events):
        return cox_breslow(risk_scores, survival_times, events)
