#!/usr/bin/env python3
"""
MIROC6 ocean-atmosphere profiles dataloader (HEALPix, structured zarr).

Expected zarr layout:
    - experiment: ['historical', 'piControl', ...]
    - member: ['r1i1p1f1', ...]
    - time: (experiment, member, time)
    - 2D vars: (experiment, member, time, space)
    - 3D vars: (experiment, member, time, depth, space)

Returns batches as:
    target: (C, T, H), condition: (0, T, H), labels, second_of_day, day_of_year, mask
"""

from __future__ import annotations

import datetime
import sys
from dataclasses import dataclass, field
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import zarr

# ---------------------------------------------------------------------------
CURRENT_DIR = Path(__file__).parent.absolute()
SRC_DIR = CURRENT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(CURRENT_DIR))

import earth2grid  # noqa: E402
from earth2grid.healpix import HEALPIX_PAD_XY, PixelOrder  # noqa: E402

from cbottle.datasets.base import BatchInfo, TimeUnit  # noqa: E402
from cbottle.datasets.dataset_2d import MAX_CLASSES  # noqa: E402
from cbottle.training.video.frame_masker import FrameMasker  # noqa: E402

# ---------------------------------------------------------------------------
HPX_LEVEL = 6
NPIX = 12 * 4**HPX_LEVEL
SIGMA_DATA = 0.5

DEFAULT_2D_VARS = ["tos", "sos", "psl", "zos", "tas", "siconc"]
DEFAULT_3D_VARS = ["thetao", "so"]


def _reorder_to_padxy(arr: np.ndarray) -> torch.Tensor:
    tensor = torch.as_tensor(arr, dtype=torch.float32)
    return earth2grid.healpix.reorder(tensor, PixelOrder.NEST, HEALPIX_PAD_XY)


def _minutes_to_datetime(minutes: int) -> datetime.datetime:
    base = datetime.datetime(1900, 1, 1)
    return base + datetime.timedelta(minutes=int(minutes))


def _seconds_of_day(dt: datetime.datetime) -> float:
    start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return (dt - start).total_seconds()


def _day_of_year(dt: datetime.datetime) -> float:
    start = dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    return (dt - start).total_seconds() / 86400.0


def _make_batch_info(channel_names: List[str]) -> BatchInfo:
    # Use Python lists instead of numpy arrays for JSON serialization
    return BatchInfo(
        channels=channel_names,
        time_step=1,
        time_unit=TimeUnit.DAY,
        scales=[1.0] * len(channel_names),
        center=[0.0] * len(channel_names),
    )


@dataclass
class MemberTimeSeries:
    experiment: str
    experiment_idx: int
    member: str
    member_idx: int
    valid_length: int
    start_time_idx: int = 0


class StructuredMIROC6ProfilesLoader:
    def __init__(
        self,
        zarr_path: str,
        experiments: Optional[List[str]] = None,
        members: Optional[List[str]] = None,
        variables_2d: Optional[List[str]] = None,
        variables_3d: Optional[List[str]] = None,
        apply_climatology: bool = False,
        climatology_path: Optional[str] = None,
    ):
        self.store = zarr.open(zarr_path, mode="r")

        self.variables_2d = variables_2d or list(DEFAULT_2D_VARS)
        self.variables_3d = variables_3d or list(DEFAULT_3D_VARS)
        self.depths = np.asarray(self.store["depth"][:], dtype=np.float32)
        self.clim_mean: Optional[np.ndarray] = None  # (12, n_channels, n_space) monthly mean per pixel
        self.clim_std: Optional[np.ndarray] = None  # (n_channels,) global std per channel (EDM style)
        self.clim_monthly = False
        self.land_mask: Optional[np.ndarray] = None  # Per-channel land mask

        self.all_experiments = list(self.store["experiment"][:])
        self.all_members = list(self.store["member"][:])

        if experiments is None:
            self.experiments = self.all_experiments
        else:
            self.experiments = [e for e in experiments if e in self.all_experiments]
            if not self.experiments:
                raise ValueError(
                    f"No matching experiments found. Available: {self.all_experiments}"
                )

        if members is None:
            self.members = self.all_members
        else:
            self.members = [m for m in members if m in self.all_members]
            if not self.members:
                raise ValueError(
                    f"No matching members found. Available: {self.all_members}"
                )

        self.valid_lengths = self.store["time_valid_length"][:]
        self.member_series: List[MemberTimeSeries] = []

        for exp in self.experiments:
            exp_idx = self.all_experiments.index(exp)
            for mem in self.members:
                mem_idx = self.all_members.index(mem)
                valid_len = int(self.valid_lengths[exp_idx, mem_idx])
                if valid_len > 0:
                    self.member_series.append(
                        MemberTimeSeries(
                            experiment=exp,
                            experiment_idx=exp_idx,
                            member=mem,
                            member_idx=mem_idx,
                            valid_length=valid_len,
                        )
                    )

        print(f"StructuredMIROC6ProfilesLoader: {len(self.member_series)} series")
        print(f"  Experiments: {self.experiments}")
        print(f"  Members: {self.members[:5]}... ({len(self.members)} total)")

        expected_channels = len(self.variables_3d) * len(self.depths) + len(
            self.variables_2d
        )
        if climatology_path is not None:
            with open(climatology_path, "rb") as f:
                clim = pickle.load(f)
            self.clim_mean = np.asarray(clim["mean"], dtype=np.float32)
            # Support multiple std formats:
            # 1. New per-pixel: (n_channels, n_space)
            # 2. Old global: (n_channels,)
            # 3. Legacy: (12, n_channels, n_space) or similar
            if "spatial_std" in clim:
                self.clim_std = np.asarray(clim["spatial_std"], dtype=np.float32)
                if clim.get("per_pixel_std", False) or self.clim_std.ndim == 2:
                    print(f"  Using per-pixel spatial_std: shape {self.clim_std.shape}")
                else:
                    print(f"  Using global spatial_std per channel: shape {self.clim_std.shape}")
            elif "std" in clim:
                # Old format: (12, n_channels, n_space) or similar
                old_std = np.asarray(clim["std"], dtype=np.float32)
                if old_std.ndim == 1:
                    self.clim_std = old_std
                elif old_std.ndim == 2:
                    self.clim_std = old_std  # Already (n_channels, n_space)
                else:
                    # Average over time to get (n_channels, n_space)
                    self.clim_std = np.nanmean(old_std, axis=0)
                    print(f"  [WARN] Converted time-varying std to static by averaging")
            self.clim_monthly = bool(clim.get("monthly", False))
            # Load per-channel land mask if available
            if "land_mask" in clim:
                self.land_mask = np.asarray(clim["land_mask"], dtype=bool)
                print(f"  Loaded per-channel land mask from climatology")
            # Print std values for debugging
            if self.clim_std is not None:
                print(f"  Climatology std values (first 5 channels): {self.clim_std[:5]}")
        elif apply_climatology and "climatology" in self.store:
            clim_group = self.store["climatology"]
            self.clim_mean = np.asarray(clim_group["mean"][:], dtype=np.float32)
            self.clim_std = np.asarray(clim_group["std"][:], dtype=np.float32)
            self.clim_monthly = bool(clim_group.attrs.get("monthly", False))
        if self.clim_mean is not None and self.clim_mean.shape[1] != expected_channels:
            print(
                "[WARN] Climatology channel count does not match loader channel "
                f"count ({self.clim_mean.shape[1]} vs {expected_channels})."
            )

    def get_sequence(
        self,
        exp_idx: int,
        mem_idx: int,
        start_time_idx: int,
        length: int,
        frame_step: int = 1,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        end_idx = start_time_idx + length * frame_step
        indices = slice(start_time_idx, end_idx, frame_step)

        data: Dict[str, np.ndarray] = {}
        for var in self.variables_2d:
            data[var] = self.store[var][exp_idx, mem_idx, indices, :]
        for var in self.variables_3d:
            data[var] = self.store[var][exp_idx, mem_idx, indices, :, :]

        times = self.store["time"][exp_idx, mem_idx, indices]
        return data, times


@dataclass
class MirocProfilesTransform:
    variables_2d: List[str]
    variables_3d: List[str]
    depths: np.ndarray
    frame_masker: Optional[FrameMasker]
    channel_names: List[str]
    clim_mean: Optional[np.ndarray] = None
    clim_std: Optional[np.ndarray] = None
    clim_monthly: bool = False
    mask_nan: bool = True
    land_mask: Optional[np.ndarray] = None  # Per-channel land mask (n_channels, n_space)
    _clim_mean_t: Optional[torch.Tensor] = field(default=None, init=False)
    _clim_std_t: Optional[torch.Tensor] = field(default=None, init=False)
    _land_mask_t: Optional[torch.Tensor] = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.clim_mean is not None and self.clim_std is not None:
            self._clim_mean_t = torch.as_tensor(self.clim_mean, dtype=torch.float32)
            # std can be (n_channels,) global or (n_channels, n_space) per-pixel
            self._clim_std_t = torch.as_tensor(self.clim_std, dtype=torch.float32)
            # If global std (n_channels,), reshape for broadcasting: (C,) -> (C, 1)
            if self._clim_std_t.ndim == 1:
                self._clim_std_t = self._clim_std_t.unsqueeze(-1)  # (C, 1) - broadcasts over H
            # If per-pixel std (n_channels, n_space), keep as is - will be reordered in __getitem__
        if self.land_mask is not None:
            # Pre-convert land mask to padxy format for fast masking
            land_mask_t = torch.as_tensor(self.land_mask, dtype=torch.float32)
            self._land_mask_t = earth2grid.healpix.reorder(
                land_mask_t, PixelOrder.NEST, HEALPIX_PAD_XY
            )

    def __call__(self, times: np.ndarray, data: Dict[str, np.ndarray]) -> dict:
        frames = []
        for t_idx in range(times.shape[0]):
            time_val = int(times[t_idx])
            dt = _minutes_to_datetime(time_val)

            channels: List[np.ndarray] = []
            for var in self.variables_3d:
                var_data = data[var][t_idx]  # (depth, space)
                for d_idx in range(var_data.shape[0]):
                    channels.append(var_data[d_idx])

            for var in self.variables_2d:
                channels.append(data[var][t_idx])

            stacked = np.stack(channels, axis=0).astype(np.float32)
            frame = _reorder_to_padxy(stacked)
            
            # Apply climatology normalization if available
            # Formula: (data - monthly_mean) / spatial_std
            # This gives us data with std ≈ 1, then we multiply by SIGMA_DATA (0.5)
            if self._clim_mean_t is not None and self._clim_std_t is not None:
                # Mean is per-pixel monthly: (12, C, H)
                if self.clim_monthly:
                    mean = self._clim_mean_t[dt.month - 1]  # (C, H)
                else:
                    mean = self._clim_mean_t  # (C, H) or (12, C, H)
                    if mean.ndim == 3:
                        mean = mean[dt.month - 1]
                # Reorder mean to padxy format
                mean = earth2grid.healpix.reorder(mean, PixelOrder.NEST, HEALPIX_PAD_XY)
                
                # Std can be (C, 1) global or (C, H) per-pixel
                std = self._clim_std_t
                if std.ndim == 2 and std.shape[1] > 1:
                    # Per-pixel std (C, H) - needs reordering to padxy
                    std = earth2grid.healpix.reorder(std, PixelOrder.NEST, HEALPIX_PAD_XY)
                # else: (C, 1) global - broadcasts over H dimension
                std = torch.where(std == 0, torch.ones_like(std), std)
                
                frame = (frame - mean) / std  # (C, H)

            frame = frame * SIGMA_DATA
            
            # Build mask: True where data is valid (not land, not NaN)
            # Start with finite check
            valid_mask = torch.isfinite(frame)
            
            # If we have a pre-computed land mask, use it (more accurate for ocean vars)
            if self._land_mask_t is not None:
                # _land_mask_t is True for land, so invert for valid ocean pixels
                ocean_mask = ~(self._land_mask_t > 0.5)
                valid_mask = valid_mask & ocean_mask
            
            mask = valid_mask.to(torch.float32)
            
            if self.mask_nan:
                frame = torch.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)

            frames.append(
                {
                    "data": frame,
                    "mask": mask,
                    "second_of_day": torch.tensor(
                        [_seconds_of_day(dt)], dtype=torch.float32
                    ),
                    "day_of_year": torch.tensor(
                        [_day_of_year(dt)], dtype=torch.float32
                    ),
                }
            )

        if len(frames) == 1:
            out = frames[0]
            target = out["data"].unsqueeze(1)
            mask = out["mask"].unsqueeze(1)
            return {
                "target": target,
                "condition": torch.zeros(0, 1, target.shape[-1]),
                "mask": mask,
                "labels": torch.zeros(MAX_CLASSES),
                "second_of_day": out["second_of_day"],
                "day_of_year": out["day_of_year"],
            }

        target = torch.stack([f["data"] for f in frames], dim=1)
        mask = torch.stack([f["mask"] for f in frames], dim=1)
        second_of_day = torch.cat([f["second_of_day"] for f in frames], dim=0)
        day_of_year = torch.cat([f["day_of_year"] for f in frames], dim=0)

        out = {
            "target": target,
            "condition": torch.zeros(0, target.shape[1], target.shape[-1]),
            "mask": mask,
            "labels": torch.zeros(MAX_CLASSES),
            "second_of_day": second_of_day,
            "day_of_year": day_of_year,
        }

        if self.frame_masker is not None:
            out = self.frame_masker(out)
        return out


class StructuredMIROC6ProfilesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        loader: StructuredMIROC6ProfilesLoader,
        split: str = "train",
        time_length: int = 1,
        frame_step: int = 1,
        frame_masker: Optional[FrameMasker] = None,
        train_ratio: float = 0.8,
        mask_nan: bool = True,
        include_metadata: bool = False,
    ):
        self.loader = loader
        self.split = split
        self.time_length = time_length
        self.frame_step = frame_step
        self.frame_masker = frame_masker
        self.include_metadata = include_metadata

        self.frames_per_seq = (time_length - 1) * frame_step + 1
        self.samples: List[Tuple[int, int]] = []

        for series_idx, series in enumerate(loader.member_series):
            n_valid = series.valid_length
            train_end = int(n_valid * train_ratio)

            if split == "train":
                time_start, time_end = 0, train_end
            elif split == "test":
                time_start, time_end = train_end, n_valid
            else:
                time_start, time_end = 0, n_valid

            max_start = time_end - self.frames_per_seq
            for start_idx in range(time_start, max_start + 1):
                self.samples.append((series_idx, start_idx))

        self.channel_names = self._build_channel_names()
        self.transform = MirocProfilesTransform(
            variables_2d=loader.variables_2d,
            variables_3d=loader.variables_3d,
            depths=loader.depths,
            frame_masker=frame_masker if time_length > 1 else None,
            channel_names=self.channel_names,
            clim_mean=loader.clim_mean,
            clim_std=loader.clim_std,
            clim_monthly=loader.clim_monthly,
            mask_nan=mask_nan,
            land_mask=loader.land_mask,  # Pass per-channel land mask
        )

        print(f"StructuredMIROC6ProfilesDataset ({split}): {len(self.samples)} samples")
        print(f"  time_length={time_length}, frame_step={frame_step}")

    def _build_channel_names(self) -> List[str]:
        names: List[str] = []
        for var in self.loader.variables_3d:
            for depth in self.loader.depths:
                depth_label = f"{int(round(depth))}m"
                names.append(f"{var}{depth_label}")
        for var in self.loader.variables_2d:
            names.append(var)
        return names

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        series_idx, start_time_idx = self.samples[idx]
        series = self.loader.member_series[series_idx]

        data, times = self.loader.get_sequence(
            exp_idx=series.experiment_idx,
            mem_idx=series.member_idx,
            start_time_idx=start_time_idx,
            length=self.time_length,
            frame_step=self.frame_step,
        )

        out = self.transform(times, data)
        if self.include_metadata:
            out["experiment"] = series.experiment
            out["member"] = series.member
        return out


def get_miroc6_profiles_dataset(
    *,
    zarr_path: str,
    experiments: Optional[List[str]] = None,
    members: Optional[List[str]] = None,
    split: str = "train",
    time_length: int = 1,
    frame_step: int = 1,
    frame_masker: Optional[FrameMasker] = None,
    train_ratio: float = 0.8,
    variables_2d: Optional[List[str]] = None,
    variables_3d: Optional[List[str]] = None,
    apply_climatology: bool = False,
    mask_nan: bool = True,
    include_metadata: bool = False,
    climatology_path: Optional[str] = None,
) -> StructuredMIROC6ProfilesDataset:
    loader = StructuredMIROC6ProfilesLoader(
        zarr_path=zarr_path,
        experiments=experiments,
        members=members,
        variables_2d=variables_2d,
        variables_3d=variables_3d,
        apply_climatology=apply_climatology,
        climatology_path=climatology_path,
    )

    dataset = StructuredMIROC6ProfilesDataset(
        loader=loader,
        split=split,
        time_length=time_length,
        frame_step=frame_step,
        frame_masker=frame_masker,
        train_ratio=train_ratio,
        mask_nan=mask_nan,
        include_metadata=include_metadata,
    )

    dataset.batch_info = _make_batch_info(dataset.channel_names)
    return dataset


def main():
    import argparse

    parser = argparse.ArgumentParser("Test structured MIROC6 profiles dataset")
    parser.add_argument("--zarr_path", required=True)
    parser.add_argument("--experiments", nargs="+", default=None)
    parser.add_argument("--time_length", type=int, default=1)
    parser.add_argument("--split", default="train")
    parser.add_argument("--apply_climatology", action="store_true")
    parser.add_argument("--no_mask_nan", action="store_true")
    parser.add_argument("--include_metadata", action="store_true")
    parser.add_argument("--climatology_path", default=None)
    args = parser.parse_args()

    ds = get_miroc6_profiles_dataset(
        zarr_path=args.zarr_path,
        experiments=args.experiments,
        split=args.split,
        time_length=args.time_length,
        apply_climatology=args.apply_climatology,
        mask_nan=not args.no_mask_nan,
        include_metadata=args.include_metadata,
        climatology_path=args.climatology_path,
    )

    print(f"Dataset length: {len(ds)}")
    if len(ds) > 0:
        sample = ds[0]
        print("Sample keys:", list(sample.keys()))
        print("target:", sample["target"].shape)
        print("condition:", sample["condition"].shape)
        print("labels:", sample["labels"].shape)
        print("second_of_day:", sample["second_of_day"])
        print("day_of_year:", sample["day_of_year"])
        print("experiment:", sample["experiment"])
        print("member:", sample["member"])


if __name__ == "__main__":
    main()
