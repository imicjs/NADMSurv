import os
import warnings
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import torch
import yaml

def load_survival_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df['os_status'].dtype == object:
        mapping = {'Dead': 1, 'Alive': 0, 'dead': 1, 'alive': 0}
        df['os_status'] = df['os_status'].map(mapping)
    df['os_time'] = pd.to_numeric(df['os_time'], errors='coerce')
    df['os_status'] = pd.to_numeric(df['os_status'], errors='coerce')
    if df['os_time'].isna().any() or df['os_status'].isna().any():
        raise ValueError('Invalid survival data containing NaN')
    return df

def safe_h5_write(h5_path: Path, overwrite: bool=False):
    h5_path = Path(h5_path)
    if h5_path.exists():
        if overwrite:
            h5_path.unlink()
        else:
            raise FileExistsError(f'File exists: {h5_path}. Set overwrite=True to replace.')
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    return h5py.File(h5_path, 'w')

def h5_read_dataset(h5_path: Path, dataset_key: str) -> np.ndarray:
    with h5py.File(h5_path, 'r') as f:
        return f[dataset_key][:]

def h5_read_attrs(h5_path: Path, group_key: str=None) -> dict:
    with h5py.File(h5_path, 'r') as f:
        target = f[group_key] if group_key else f
        return dict(target.attrs)

def align_patient_ids(*dataframes: pd.DataFrame) -> list:
    common = set(dataframes[0]['patient_id'])
    for df in dataframes[1:]:
        common &= set(df['patient_id'])
    return sorted(list(common))

def set_seed(seed: int=42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path=None):
    config_path = Path(config_path) if config_path is not None else Path(__file__).resolve().parents[2] / 'configs/config.yaml'
    with config_path.open(encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    project = Path(__file__).resolve().parents[2]
    root = Path(cfg.get('paths', {}).get('root', '.'))
    root = root if root.is_absolute() else (project / root).resolve()
    cfg.setdefault('paths', {})['root'] = str(root)
    data = dict(cfg.get('paths', {}).get('data', {}))
    data.update(cfg.get('data', {}))
    cfg['data'] = {k: str(Path(v) if Path(v).is_absolute() else root / v) for (k, v) in data.items()}
    groups = {'clinical': 'raw_genomics', 'mrna': 'raw_genomics', 'wsi_delete': 'raw_genomics', 'wsi_features': 'raw_pathology', 'wsi_patches': 'raw_pathology', 'decoupled_rep': 'decoupled', 'decoupled_meta': 'decoupled', 'checkpoint': 'decoupled'}
    for (k, v) in cfg.get('files', {}).items():
        p = Path(v)
        if not p.is_absolute():
            p = Path(cfg['data'][groups[k]]) / p if k in groups else root / p
        cfg['files'][k] = str(p)
    return cfg

def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
