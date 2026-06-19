# src/utils/io_utils.py
# I/O utilities, configuration loader, and device detection.

import os
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import yaml


def load_config(config_path: Path = None) -> dict:
    """
    Load project configuration from YAML.
    All relative paths under 'paths' are resolved to absolute paths
    based on the project root.
    """
    if config_path is None:
        # Locate config.yaml relative to this file: ../../configs/config.yaml
        config_path = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    root = Path(cfg["paths"]["root"]).resolve()
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[2]
    
    # Resolve all data paths to absolute paths
    for section in ["data", "files"]:
        if section not in cfg:
            continue
        for key, val in cfg[section].items():
            if isinstance(val, str) and not Path(val).is_absolute():
                if section == "data":
                    cfg[section][key] = str(root / val)
                elif section == "files":
                    # Files are relative to their data subdirectories
                    if key in ("clinical", "mrna", "wsi_delete"):
                        cfg[section][key] = str(Path(cfg["data"]["raw_genomics"]) / val)
                    elif key in ("wsi_features", "wsi_patches"):
                        cfg[section][key] = str(Path(cfg["data"]["raw_pathology"]) / val)
                    elif key in ("decoupled_rep", "decoupled_meta", "checkpoint"):
                        cfg[section][key] = str(Path(cfg["data"]["decoupled"]) / val)
                    else:
                        cfg[section][key] = str(root / val)
    
    cfg["paths"]["root"] = str(root)
    return cfg


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"[INFO] Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[INFO] CUDA not available, using CPU")
    return device


def load_survival_csv(path: Path) -> pd.DataFrame:
    """Load clinical CSV and standardize survival columns."""
    df = pd.read_csv(path)
    
    # Standardize os_status
    if df["os_status"].dtype == object:
        mapping = {"Dead": 1, "Alive": 0, "dead": 1, "alive": 0}
        df["os_status"] = df["os_status"].map(mapping)
    
    df["os_time"] = pd.to_numeric(df["os_time"], errors="coerce")
    df["os_status"] = pd.to_numeric(df["os_status"], errors="coerce")
    
    if df["os_time"].isna().any() or df["os_status"].isna().any():
        raise ValueError("Invalid survival data containing NaN")
    
    return df


def safe_h5_write(h5_path: Path, overwrite: bool = False):
    """Context manager for safe HDF5 writing with optional overwrite."""
    h5_path = Path(h5_path)
    if h5_path.exists():
        if overwrite:
            h5_path.unlink()
        else:
            raise FileExistsError(f"File exists: {h5_path}. Set overwrite=True to replace.")
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    return h5py.File(h5_path, "w")


def h5_read_dataset(h5_path: Path, dataset_key: str) -> np.ndarray:
    """Read a numpy array from an HDF5 dataset."""
    with h5py.File(h5_path, "r") as f:
        return f[dataset_key][:]


def h5_read_attrs(h5_path: Path, group_key: str = None) -> dict:
    """Read all attributes from an HDF5 file or group as a dict."""
    with h5py.File(h5_path, "r") as f:
        target = f[group_key] if group_key else f
        return dict(target.attrs)


def align_patient_ids(*dataframes: pd.DataFrame) -> list:
    """
    Return sorted intersection of patient_ids across multiple DataFrames.
    Assumes each DataFrame has a 'patient_id' column.
    """
    common = set(dataframes[0]["patient_id"])
    for df in dataframes[1:]:
        common &= set(df["patient_id"])
    return sorted(list(common))


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False