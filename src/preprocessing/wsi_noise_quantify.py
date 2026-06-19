# src/preprocessing/wsi_noise_quantify.py
# WSI technical noise quantification via GPU wavelet decomposition.

import os
import sys
import threading
import queue
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import stats
import torch
import ptwt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.io_utils import load_config, get_device


WAVELET = "db2"
DECOMP_LEVEL = 2


class AsyncDataLoader:
    def __init__(self, h5_file: Path, patient_ids: list, max_prefetch: int = 2):
        self.h5_file = h5_file
        self.patient_ids = patient_ids
        self.queue = queue.Queue(maxsize=max_prefetch)
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._preload_worker)
        self.thread.daemon = True
        self.thread.start()

    def _preload_worker(self):
        for pid in self.patient_ids:
            if self.stop_event.is_set():
                break
            with h5py.File(self.h5_file, "r") as h5f:
                patches = h5f[pid]["patches"][:]
            self.queue.put((pid, patches), block=True)
        self.queue.put(None)

    def get_next(self):
        item = self.queue.get(block=True)
        return item

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join()


class GPUBatchProcessor:
    def __init__(self, wavelet: str = "db2", level: int = 2, device: torch.device = None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wavelet = wavelet
        self.level = level
        self.pinned_buffer = None

    def compute_batch_noise_intensity(self, patches_batch: np.ndarray) -> np.ndarray:
        batch_size = patches_batch.shape[0]

        if self.pinned_buffer is None or self.pinned_buffer.shape[0] < batch_size:
            self.pinned_buffer = torch.empty(
                (batch_size, 256, 256, 3),
                dtype=torch.float32,
                pin_memory=True,
            )

        self.pinned_buffer[:batch_size] = torch.from_numpy(patches_batch)
        gpu_tensor = self.pinned_buffer[:batch_size].to(self.device, non_blocking=True)

        gpu_gray = torch.mean(gpu_tensor, dim=3) / 255.0
        gpu_gray = gpu_gray.unsqueeze(1)

        coeffs = ptwt.wavedec2(gpu_gray, self.wavelet, level=self.level)

        cA2 = coeffs[0]
        cH2, cV2, cD2 = coeffs[1]

        energy_low = torch.sum(cA2 ** 2, dim=(1, 2, 3))
        energy_high = (
            torch.sum(cH2 ** 2, dim=(1, 2, 3))
            + torch.sum(cV2 ** 2, dim=(1, 2, 3))
            + torch.sum(cD2 ** 2, dim=(1, 2, 3))
        )
        energy_total = energy_low + energy_high

        noise_intensity = (energy_high / (energy_total + 1e-10)).cpu().numpy()

        del coeffs, gpu_gray, gpu_tensor
        return noise_intensity


def compute_noise_metrics(noise_intensities: np.ndarray) -> tuple:
    noise_intensity = np.mean(noise_intensities)
    importance_score = 1.0 - noise_intensity
    uncertainty_score = np.std(noise_intensities) / (np.mean(noise_intensities) + 1e-10)
    median_ratio = np.median(noise_intensities)
    mad_ratio = stats.median_abs_deviation(noise_intensities, scale="normal")
    return importance_score, uncertainty_score, median_ratio, mad_ratio


def generate_noise_label(importance_score: float, uncertainty_score: float,
                         importance_median: float, uncertainty_median: float) -> int:
    high_importance = importance_score >= importance_median
    high_uncertainty = uncertainty_score >= uncertainty_median

    if high_importance and high_uncertainty:
        return 3
    elif high_importance and not high_uncertainty:
        return 2
    elif not high_importance and high_uncertainty:
        return 1
    else:
        return 0


def main():
    config = load_config()
    hparams = config["hyperparams"]

    input_h5 = Path(config["files"]["wsi_patches"])
    output_h5 = Path(config["data"]["processed_pathology"]) / "Cancer_patches_IU.h5"

    device = get_device()
    gpu_mem_gb = 16
    max_batch = int(gpu_mem_gb * 150 * 0.8)

    print(f"[INFO] Detected {gpu_mem_gb}GB GPU, safe batch size: {max_batch}")

    if not input_h5.exists():
        print(f"[ERROR] Input not found: {input_h5}")
        return

    with h5py.File(input_h5, "r") as h5f:
        patient_ids = sorted(list(h5f.keys()))
    print(f"[INFO] Patients: {len(patient_ids)}")

    processor = GPUBatchProcessor(wavelet=WAVELET, level=DECOMP_LEVEL, device=device)
    dummy = np.random.randint(0, 255, (max_batch, 256, 256, 3), dtype=np.uint8)
    _ = processor.compute_batch_noise_intensity(dummy)
    print("[INFO] GPU warm-up complete")

    all_importance = []
    all_uncertainty = []
    all_patient_data = {}

    with tqdm(total=len(patient_ids), desc="Processing progress") as pbar:
        loader = AsyncDataLoader(input_h5, patient_ids, max_prefetch=2)
        loader.start()

        for _ in range(len(patient_ids)):
            pid, patches_all = loader.get_next()

            noise_intensities = []
            n_patches = patches_all.shape[0]

            for start in range(0, n_patches, max_batch):
                end = min(start + max_batch, n_patches)
                batch = patches_all[start:end]
                batch_noise = processor.compute_batch_noise_intensity(batch)
                noise_intensities.extend(batch_noise)

            noise_intensities = np.array(noise_intensities)
            imp, unc, med, mad = compute_noise_metrics(noise_intensities)

            all_importance.append(imp)
            all_uncertainty.append(unc)
            all_patient_data[pid] = {
                "noise_intensities": noise_intensities,
                "importance_score": imp,
                "uncertainty_score": unc,
            }

            pbar.update(1)

        loader.stop()

    importance_median = np.median(all_importance)
    uncertainty_median = np.median(all_uncertainty)

    print(f"[INFO] Global importance median: {importance_median:.4f}")
    print(f"[INFO] Global uncertainty median: {uncertainty_median:.4f}")

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_h5, "w") as h5f_out:
        for pid in tqdm(patient_ids, desc="Writing H5"):
            data = all_patient_data[pid]
            label = generate_noise_label(
                data["importance_score"],
                data["uncertainty_score"],
                importance_median,
                uncertainty_median,
            )

            grp = h5f_out.create_group(pid)
            grp.attrs["importance_score"] = data["importance_score"]
            grp.attrs["uncertainty_score"] = data["uncertainty_score"]
            grp.attrs["noise_label"] = label

            grp.create_dataset(
                "noise_intensities",
                data=data["noise_intensities"],
                compression="gzip",
                compression_opts=4,
            )

        h5f_out.create_dataset("global_importance_scores", data=np.array(all_importance))
        h5f_out.create_dataset("global_uncertainty_scores", data=np.array(all_uncertainty))
        h5f_out.attrs["description"] = "GPU wavelet noise quantification"
        h5f_out.attrs["wavelet"] = WAVELET
        h5f_out.attrs["decomposition_level"] = DECOMP_LEVEL

    report_data = []
    for pid in patient_ids:
        data = all_patient_data[pid]
        label = generate_noise_label(
            data["importance_score"],
            data["uncertainty_score"],
            importance_median,
            uncertainty_median,
        )
        report_data.append({
            "patient_id": pid,
            "importance_score": data["importance_score"],
            "uncertainty_score": data["uncertainty_score"],
            "noise_label": label,
        })

    report_df = pd.DataFrame(report_data)
    report_csv = output_h5.with_suffix(".csv")
    report_df.to_csv(report_csv, index=False)
    print(f"[INFO] Saved Cancer_patches_IU.h5")
    print(f"[INFO] Saved Cancer_patches_IU_report.csv")
    print("[INFO] Done")


if __name__ == "__main__":
    main()