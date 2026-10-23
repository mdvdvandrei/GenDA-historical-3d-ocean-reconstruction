#!/usr/bin/env python3
"""Train unconditional MIROC6 profiles diffusion prior (26 ch)."""

import csv
import dataclasses
import logging
import os
import time
from pathlib import Path

import torch
import zarr
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from earth2grid import healpix

# Add src and scripts to path to allow imports
CURRENT_DIR = Path(__file__).parent.absolute()
SRC_DIR = CURRENT_DIR.parent / "src"
import sys

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(CURRENT_DIR))

from cbottle import distributed as dist  # noqa: E402
from cbottle.dataclass_parser import parse_args  # noqa: E402

import train_miroc6_base as train_base  # noqa: E402
import profiles_dataloader  # noqa: E402


def _default_profiles_loop():
    base_loop = train_base.CLI().loop
    defaults = {f.name: getattr(base_loop, f.name) for f in dataclasses.fields(base_loop)}
    if "bf16" in defaults:
        defaults["bf16"] = True
    if "network" in defaults:
        defaults["network"] = dataclasses.replace(defaults["network"], model_channels=128)
    return MirocProfilesTrainingLoop(**defaults)


@dataclasses.dataclass
class MirocProfilesTrainingLoop(train_base.MirocTrainingLoop):
    """
    Training loop for MIROC6 profiles dataset with per-channel masking.
    
    Truly unconditional generation - only calendar embeddings.
    Mask is used only for the loss (to mask land from ocean variables).
    """
    apply_climatology: bool = False
    climatology_path: str | None = None
    _profile_out_channels: int | None = None
    n_val_members: int = 5  # last N historical members held out for validation

    def _get_member_split(self):
        """Split members: all except last N for training, last N for validation.

        Returns (train_members, val_members) as plain Python string lists.
        """
        store = zarr.open(self.miroc_zarr_path, mode="r")
        all_members = [str(m) for m in store["member"][:]]
        n = self.n_val_members
        train_members = all_members[:-n] if n > 0 else all_members
        val_members   = all_members[-n:] if n > 0 else []
        return train_members, val_members

    def get_dataset(self, train: bool):
        train_members, val_members = self._get_member_split()

        if train:
            # Training: historical (excluding last N members) + piControl (all members)
            # We create two datasets and concatenate them.
            members = train_members
            print(f"[TRAIN] historical: {len(members)} members "
                  f"(full time range): {members[:5]}{'…' if len(members)>5 else ''}")

            ds_hist = profiles_dataloader.get_miroc6_profiles_dataset(
                zarr_path=self.miroc_zarr_path,
                experiments=['historical'],
                members=members,
                split="train",
                train_ratio=1.0,
                time_length=self.time_window_months,
                frame_step=1,
                frame_masker=None,
                apply_climatology=self.apply_climatology,
                mask_nan=True,
                climatology_path=self.climatology_path,
            )

            # Add piControl with ALL members (no holdout needed)
            try:
                ds_pi = profiles_dataloader.get_miroc6_profiles_dataset(
                    zarr_path=self.miroc_zarr_path,
                    experiments=['piControl'],
                    members=None,  # all members
                    split="train",
                    train_ratio=1.0,
                    time_length=self.time_window_months,
                    frame_step=1,
                    frame_masker=None,
                    apply_climatology=self.apply_climatology,
                    mask_nan=True,
                    climatology_path=self.climatology_path,
                )
                print(f"[TRAIN] piControl: {len(ds_pi)} samples")
                ds = torch.utils.data.ConcatDataset([ds_hist, ds_pi])
                # Propagate batch_info from the historical dataset
                ds.batch_info = ds_hist.batch_info
                print(f"[TRAIN] Total: {len(ds)} samples "
                      f"(historical {len(ds_hist)} + piControl {len(ds_pi)})")
            except (ValueError, KeyError) as e:
                print(f"[TRAIN] piControl not available ({e}), using historical only")
                ds = ds_hist

            return ds
        else:
            # Validation: last N historical members only (full time range)
            members = val_members
            print(f"[VAL] Using {len(members)} held-out historical members "
                  f"(full time range): {members}")

            ds = profiles_dataloader.get_miroc6_profiles_dataset(
                zarr_path=self.miroc_zarr_path,
                experiments=['historical'],
                members=members,
                split="train",
                train_ratio=1.0,
                time_length=self.time_window_months,
                frame_step=1,
                frame_masker=None,
                apply_climatology=self.apply_climatology,
                mask_nan=True,
                climatology_path=self.climatology_path,
            )
            return ds

    def _ensure_profile_channels(self) -> None:
        if self._profile_out_channels is not None:
            return
        store = zarr.open(self.miroc_zarr_path, mode="r")
        depth_len = int(store["depth"].shape[0])
        n_2d = len(profiles_dataloader.DEFAULT_2D_VARS)
        n_3d = len(profiles_dataloader.DEFAULT_3D_VARS)
        self._profile_out_channels = n_2d + n_3d * depth_len

    @property
    def out_channels(self) -> int:
        self._ensure_profile_channels()
        return int(self._profile_out_channels)

    @property
    def model_config(self):
        """
        Truly unconditional: no condition channels.
        
        Calendar (day of year, time of day) is handled by the model's 
        CalendarEmbedding layer via second_of_day/day_of_year in batch.
        No additional conditioning needed.
        """
        import cbottle.config.models
        
        out_channels = self.out_channels
        time_len = self.time_window_months
        
        # Unconditional: condition_channels = 0
        # Calendar embeddings are injected separately by the model
        config = cbottle.config.models.ModelConfigV1(
            label_dim=0,
            out_channels=out_channels,
            condition_channels=0,  # Truly unconditional
            model_channels=self.network.model_channels,
            time_length=time_len,
            label_dropout=0.0,
            calendar_include_legacy_bug=self.network.include_legacy_calendar_bug,
        )
        print(f"[DEBUG model_config] PROFILES UNCONDITIONAL: out={out_channels}, cond=0, time={time_len}")
        return config

    def _get_batch_info(self):
        self._ensure_profile_channels()
        channel_names = []
        depths = zarr.open(self.miroc_zarr_path, mode="r")["depth"][:]
        for var in profiles_dataloader.DEFAULT_3D_VARS:
            for depth in depths:
                # Convert numpy float32 to Python int for JSON serialization
                channel_names.append(f"{var}{int(round(float(depth)))}m")
        channel_names.extend(profiles_dataloader.DEFAULT_2D_VARS)
        return train_base.BatchInfo(
            channels=channel_names,
            time_step=30,
            time_unit=train_base.TimeUnit.DAY,
        )

    def train_step(self, *, target, timestamp=None, **batch):
        """
        Apply per-channel mask to loss (for land/ocean masking).
        
        The mask is (B, C, T, H) where:
        - 1.0 = valid ocean pixel (include in loss)
        - 0.0 = land pixel or invalid (exclude from loss)
        
        For 3D ocean variables (thetao, so), the land mask varies by depth:
        - Shallow depths have more ocean pixels
        - Deep depths have fewer ocean pixels (ocean bathymetry)
        
        The mask is applied per-channel so each depth level has its own land mask.
        """
        from cbottle import training_stats

        loss: train_base.cbottle.loss.Output = self.loss_fn(
            self._curry_net(self.ddp, batch),
            target,
            classifier_labels=batch.get("classifier_labels"),
        )

        # Apply mask to loss (mask out land from ocean variables)
        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(loss.total.device)
            
            # loss.total has shape (B, C, T, H) or similar
            # Ensure mask broadcasts correctly
            if mask.shape == loss.total.shape:
                # Direct match - multiply element-wise
                masked_loss = loss.total * mask
                # Compute mean only over valid pixels to avoid underweighting
                valid_count = mask.sum()
                if valid_count > 0:
                    loss.total = masked_loss.sum() / valid_count
                else:
                    loss.total = masked_loss.mean()
                    
                if loss.denoising is not None and loss.denoising.shape == mask.shape:
                    masked_denoising = loss.denoising * mask
                    if valid_count > 0:
                        loss.denoising = masked_denoising.sum() / valid_count
                    else:
                        loss.denoising = masked_denoising.mean()
            else:
                # Shape mismatch - try broadcasting
                try:
                    masked_loss = loss.total * mask
                    valid_count = mask.sum()
                    if valid_count > 0:
                        loss.total = masked_loss.sum() / valid_count
                    else:
                        loss.total = masked_loss.mean()
                except RuntimeError:
                    # Fallback: just use the loss as-is
                    print(f"[WARN] Mask shape {mask.shape} != loss shape {loss.total.shape}")

        training_stats.report("Loss/denoising", loss.denoising)
        self._train_sigma_metric.update(loss.sigma, loss.denoising)
        if loss.classification is not None:
            training_stats.report("Loss/classification", loss.classification)
            self._train_classifier_sigma_metric.update(loss.sigma, loss.classification)
        return loss.total

    def _plot_generated_samples(self, samples: torch.Tensor, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        grid = healpix.Grid(6, pixel_order=healpix.PixelOrder.NEST)
        lat = grid.lat
        lon = grid.lon

        sample = samples[0].detach().cpu().numpy()  # (C, T, X)
        time_idx = 0
        channel_names = (
            list(getattr(self.batch_info, "channels", []))
            if hasattr(self, "batch_info")
            else []
        )
        if not channel_names:
            channel_names = [f"ch{idx}" for idx in range(sample.shape[0])]

        for ch in range(sample.shape[0]):
            values = sample[ch, time_idx]
            fig, ax = plt.subplots(figsize=(8, 4.5))
            sc = ax.scatter(lon, lat, c=values, s=1, cmap="coolwarm")
            plt.colorbar(sc, ax=ax, shrink=0.7)
            ax.set_title(f"Generated {channel_names[ch]}")
            fig.tight_layout()
            fig.savefig(
                os.path.join(out_dir, f"generated_{channel_names[ch]}.png"), dpi=200
            )
            plt.close(fig)

    def _run_inline_generation(self, step: int, snapshot_path: str, log_fn):
        # Optional training-time visualization helper (not required for prior training).
        log_fn("  -> Inline generation skipped (not included in this package)")
        return

    def train(self):
        """Training with sigma_data verification before starting."""
        # Verify sigma_data by checking first batch
        if dist.get_rank() == 0:
            print("=" * 60)
            print("MIROC6 Profiles Training")
            print("=" * 60)
            print(f"Model output channels: {self.out_channels}")
            print(f"Time length: {self.time_window_months}")
            print(f"Batch size: {self.batch_size}, per GPU: {self.batch_gpu}")
            
            try:
                _, train_loader, _ = self.get_data_loaders(self.batch_gpu)
                test_batch = next(iter(train_loader))
                target = test_batch['target']
                mask = test_batch.get('mask')
                if mask is not None:
                    valid_data = target[mask > 0.5]
                    if len(valid_data) > 0:
                        actual_sigma = valid_data.std().item()
                        print(f"EDM sigma_data verification: {actual_sigma:.4f} (expected ~0.5)")
                else:
                    actual_sigma = target.std().item()
                    print(f"EDM sigma_data verification: {actual_sigma:.4f} (expected ~0.5)")
            except Exception as e:
                print(f"Could not verify sigma_data: {e}")
            
            print("=" * 60)
        
        # Call parent train method
        super().train()


@dataclasses.dataclass
class CLI(train_base.CLI):
    loop: MirocProfilesTrainingLoop = dataclasses.field(
        default_factory=_default_profiles_loop
    )


def main():
    cli = parse_args(CLI, convert_underscore_to_hyphen=False)
    logging.basicConfig(level=logging.INFO)
    loop = cli.loop

    if not cli.name:
        name = "miroc6-profiles-" + str(int(time.time()))
    else:
        name = cli.name

    loop.run_dir = os.path.join(cli.output_dir, name)
    dist.init()

    # Handle test_fast mode
    if cli.test_fast:
        loop.total_ticks = 2
        loop.steps_per_tick = 1
        loop.state_dump_ticks = 1
        loop.snapshot_ticks = 1
        loop.batch_size = 1 * dist.get_world_size()
        loop.batch_gpu = 1

    loop.setup()

    # Resume logic (same as base)
    resume_dirs_in_priority = [loop.run_dir, cli.resume_dir]
    new_training = True
    for rundir in resume_dirs_in_priority:
        try:
            loop.resume_from_rundir(rundir, require_all=False)
            new_training = False
            break
        except Exception:
            continue

    if new_training:
        print("Starting new training")

    if cli.validate_only:
        loop.validate()
        loop.flush_training_stats()
    elif cli.validate_all:
        loop.validate_all_checkpoints()
    else:
        loop.train()


if __name__ == "__main__":
    main()
