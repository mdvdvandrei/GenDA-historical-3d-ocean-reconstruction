#!/usr/bin/env python3
"""
OSSE reconstruction: real obs masks/grids, values from MIROC6 truth.

  python scripts/reconstruct_osse.py \\
    --checkpoint snapshot.pkl --zarr_path miroc6_profiles.zarr \\
    --member r47i1p1f1 --start_date 1950-01 --end_date 2014-12 \\
    --noise_obs_std 0.15 --gamma 0.001 --steps 128 --corrections 0 \\
    --use_bf16 --output_dir out_osse
"""

import os
import sys
import json
import argparse
import logging
import pickle
import datetime as dt
from pathlib import Path
from io import BytesIO

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import numpy as np
import xarray as xr
from tqdm import tqdm
from PIL import Image

# Add paths
CURRENT_DIR = Path(__file__).parent.absolute()
SRC_DIR = CURRENT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(CURRENT_DIR))


from reconstruct_common import (
    DEFAULT_2D_VARS,
    DEFAULT_3D_VARS,
    OCEAN_VARS,
    SIGMA_DATA,
    CoarseGridMapper,
    EDMWrapperProfiles,
    LongDenoiserProfiles,
    PatchedRealisticScore,
    adjust_L_for_windows,
    get_channel_names,
    is_ocean_var,
    load_checkpoint,
    make_centered_window_indices_reflect,
    _find_time_index,
    _parse_date,
    _sample_with_threshold,
)

import earth2grid
from earth2grid.healpix import PixelOrder, HEALPIX_PAD_XY, Grid

import profiles_dataloader
from sda.score import GaussianScore_from_denoiser, VPSDE_from_denoiser, VPSDE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("profiles_realistic_v4")





# =============================================================================
# Load checkpoint / climatology / GT
# =============================================================================



def load_land_mask_and_depths(climatology_path: str):
    with open(climatology_path, "rb") as f:
        clim = pickle.load(f)
    land_mask = clim.get("land_mask")
    depths = clim.get("depths", np.array([25, 75, 150, 300, 500, 700, 1000, 1500, 2000, 9999], dtype=np.float32))
    channel_names = clim.get("channel_names", get_channel_names(depths))
    logger.info(f"Loaded land_mask shape={land_mask.shape if land_mask is not None else None}, "
                f"{len(channel_names)} channels, {len(depths)} depths")
    return land_mask, depths, channel_names




def _zarr_idx_for_date(times_arr: np.ndarray, base: dt.datetime,
                       target_dt: dt.datetime,
                       valid_range: tuple = None) -> int:
    """Find the zarr time index closest to *target_dt*.

    valid_range: optional (lo, hi) index range to restrict the search.
    """
    target_min = (target_dt - base).total_seconds() / 60.0
    diffs = np.full_like(times_arr, 1e18)
    lo = valid_range[0] if valid_range else 0
    hi = valid_range[1] if valid_range else len(times_arr)
    diffs[lo:hi] = np.abs(times_arr[lo:hi] - target_min)
    return int(np.argmin(diffs))


def load_gt_historical(zarr_path: str, clim_path: str,
                       start_date: str, end_date: str,
                       device, train_ratio: float = 0.8,
                       member: str | None = None):
    """
    Load ground truth from a historical member between *start_date* and
    *end_date* (inclusive).

    When ``member`` is given, load that specific ensemble member (e.g.
    ``"r46i1p1f1"``).  Otherwise the first available member is used.

    When ``train_ratio < 1.0`` the accessible time range is restricted to
    the test subset ``[int(valid_length * train_ratio), valid_length)``.
    Set ``train_ratio=1.0`` to use the **full** time range (useful for
    held-out member validation).
    """
    start_dt = _parse_date(start_date)
    end_dt   = _parse_date(end_date)
    logger.info(f"Loading historical GT: {start_dt:%Y-%m} → {end_dt:%Y-%m}")

    loader = profiles_dataloader.StructuredMIROC6ProfilesLoader(
        zarr_path=zarr_path,
        experiments=['historical'],
        members=[member] if member else None,
        climatology_path=clim_path,
    )

    series = loader.member_series[0]
    exp_idx = series.experiment_idx
    mem_idx = series.member_idx
    logger.info(f"  Member: {series.member} (idx={mem_idx}), "
                f"{series.valid_length} months")

    # Time array & base epoch  (zarr stores minutes since 1900-01-01)
    times_arr = np.array(loader.store["time"][exp_idx, mem_idx, :], dtype=np.float64)
    base = dt.datetime(1900, 1, 1)

    # Accessible time range
    if train_ratio >= 1.0:
        # Full range (member-based validation)
        range_start_idx = 0
        range_end_idx = series.valid_length
        range_start_dt = base + dt.timedelta(minutes=float(times_arr[0]))
        range_end_dt = base + dt.timedelta(
            minutes=float(times_arr[series.valid_length - 1])
        )
        logger.info(f"  Full time range: idx [0, {series.valid_length}) "
                    f"→ {range_start_dt:%Y-%m} to {range_end_dt:%Y-%m}")
    else:
        # Time-based split: only test subset
        range_start_idx = int(series.valid_length * train_ratio)
        range_end_idx = series.valid_length
        range_start_dt = base + dt.timedelta(
            minutes=float(times_arr[range_start_idx])
        )
        range_end_dt = base + dt.timedelta(
            minutes=float(times_arr[series.valid_length - 1])
        )
        logger.info(f"  Test subset: idx [{range_start_idx}, {series.valid_length}) "
                    f"→ {range_start_dt:%Y-%m} to {range_end_dt:%Y-%m}")

    # Map requested dates to zarr indices
    valid_range = (range_start_idx, range_end_idx)
    start_idx = _zarr_idx_for_date(times_arr, base, start_dt, valid_range)
    end_idx   = _zarr_idx_for_date(times_arr, base, end_dt,   valid_range)

    if end_idx >= series.valid_length:
        end_idx = series.valid_length - 1
    if end_idx < start_idx:
        raise ValueError(
            f"Requested range {start_date} → {end_date} could not be "
            f"satisfied within the accessible range "
            f"({range_start_dt:%Y-%m} to {range_end_dt:%Y-%m})"
        )

    total_months = end_idx - start_idx + 1

    actual_start = base + dt.timedelta(minutes=int(times_arr[start_idx]))
    actual_end   = base + dt.timedelta(minutes=int(times_arr[end_idx]))
    logger.info(f"  zarr [{start_idx}:{start_idx + total_months}] → "
                f"{actual_start:%Y-%m} to {actual_end:%Y-%m} "
                f"({total_months} months)")

    data, raw_times = loader.get_sequence(exp_idx, mem_idx,
                                          start_idx, total_months)

    # Channel names
    ch_names = []
    for var in loader.variables_3d:
        for depth in loader.depths:
            ch_names.append(f"{var}{int(round(depth))}m")
    for var in loader.variables_2d:
        ch_names.append(var)

    # Climatology transform
    transform = profiles_dataloader.MirocProfilesTransform(
        variables_2d=loader.variables_2d,
        variables_3d=loader.variables_3d,
        depths=loader.depths,
        frame_masker=None,
        channel_names=ch_names,
        clim_mean=loader.clim_mean,
        clim_std=loader.clim_std,
        clim_monthly=loader.clim_monthly,
        mask_nan=True,
        land_mask=loader.land_mask,
    )

    frames, months = [], []
    for t in range(total_months):
        single_data = {}
        for var in loader.variables_2d:
            single_data[var] = data[var][t:t + 1]
        for var in loader.variables_3d:
            single_data[var] = data[var][t:t + 1]
        single_times = raw_times[t:t + 1]

        result = transform(single_times, single_data)
        frames.append(result['target'])

        time_dt = base + dt.timedelta(minutes=int(raw_times[t]))
        months.append(time_dt.month - 1)

    gt = torch.cat(frames, dim=1).unsqueeze(0).to(device)
    months_t = torch.tensor(months, device=device, dtype=torch.long)

    logger.info(f"  GT shape: {gt.shape}, months: {months[:12]}...")
    # Return actual_start so obs masks can align to the same dates
    return gt, months_t, ch_names, actual_start




# =============================================================================
# Observation mask loaders
# =============================================================================



def load_hadsst_masks(path, mapper, start_dt: dt.datetime, n_months):
    logger.info(f"  Loading HadSST masks: {path}")
    ds = xr.open_dataset(path)
    t0 = _find_time_index(ds, start_dt)
    lat, lon = ds['latitude'].values, ds['longitude'].values
    masks = []
    for i in range(n_months):
        ti = min(t0 + i, len(ds.time) - 1)
        field = ds['tos'].isel(time=ti).values
        masks.append(mapper.regrid_mask(lat, lon, ~np.isnan(field)))
    ds.close()
    pct = np.mean([m.sum() / mapper.n_cells * 100 for m in masks])
    logger.info(f"    avg {pct:.1f}% coarse cells observed / month")
    return masks


def load_crutem_masks(path, mapper, start_dt: dt.datetime, n_months):
    logger.info(f"  Loading CRUTEM masks: {path}")
    ds = xr.open_dataset(path)
    t0 = _find_time_index(ds, start_dt)
    lat, lon = ds['latitude'].values, ds['longitude'].values
    masks = []
    for i in range(n_months):
        ti = min(t0 + i, len(ds.time) - 1)
        field = ds['tas'].isel(time=ti).values
        masks.append(mapper.regrid_mask(lat, lon, ~np.isnan(field)))
    ds.close()
    pct = np.mean([m.sum() / mapper.n_cells * 100 for m in masks])
    logger.info(f"    avg {pct:.1f}% coarse cells observed / month")
    return masks


def load_icoads_masks(path, mapper, start_dt: dt.datetime, n_months):
    logger.info(f"  Loading ICOADS masks: {path}")
    ds = xr.open_dataset(path)
    t0 = _find_time_index(ds, start_dt)
    lat, lon = ds['lat'].values, ds['lon'].values
    masks = []
    for i in range(n_months):
        ti = min(t0 + i, len(ds.time) - 1)
        field = ds['psl'].isel(time=ti).values
        masks.append(mapper.regrid_mask(lat, lon, ~np.isnan(field)))
    ds.close()
    pct = np.mean([m.sum() / mapper.n_cells * 100 for m in masks])
    logger.info(f"    avg {pct:.1f}% coarse cells observed / month")
    return masks


def load_en4_masks(path, mapper, start_dt: dt.datetime, n_months, var_name):
    logger.info(f"  Loading EN4 masks ({var_name}): {path}")
    ds = xr.open_dataset(path)
    t0 = _find_time_index(ds, start_dt)
    lat, lon = ds['lat'].values, ds['lon'].values
    n_depths = len(ds['depth'])

    masks_per_depth = []
    for d in range(n_depths):
        depth_masks = []
        for i in range(n_months):
            ti = min(t0 + i, len(ds.time) - 1)
            field = ds[var_name].isel(time=ti).values[:, :, d]
            depth_masks.append(mapper.regrid_mask(lat, lon, ~np.isnan(field)))
        masks_per_depth.append(depth_masks)
        pct = np.mean([m.sum() / mapper.n_cells * 100 for m in depth_masks])
        logger.info(f"    depth {d} ({float(ds['depth'].values[d]):.0f}m): "
                     f"avg {pct:.1f}% observed / month")
    ds.close()
    return masks_per_depth


def _med_coarse_cells(mapper):
    """Boolean (n_cells,) mask of coarse cells whose center lies in the
    Mediterranean/Black Sea bbox (lat 30-47N, lon 6W-42E)."""
    latc = 0.5 * (mapper.lat_edges[:-1] + mapper.lat_edges[1:])
    lonc = 0.5 * (mapper.lon_edges[:-1] + mapper.lon_edges[1:])
    LATc, LONc = np.meshgrid(latc, lonc, indexing='ij')
    latf = LATc.ravel(); lonf = LONc.ravel()
    return (latf >= 30.0) & (latf <= 47.0) & (lonf >= -6.0) & (lonf <= 42.0)


def load_duacs_masks(path, mapper, start_dt: dt.datetime, n_months,
                     var: str = 'sla', exclude_med: bool = True):
    """Per-month coarse-cell coverage masks from DUACS altimetry (where the
    product has data), used to place GT-sampled SSH pseudo-obs in the OSSE."""
    logger.info(f"  Loading DUACS SSH masks ({var}): {path}")
    ds = xr.open_dataset(path)
    t0 = _find_time_index(ds, start_dt)
    latn = 'lat' if 'lat' in ds else 'latitude'
    lonn = 'lon' if 'lon' in ds else 'longitude'
    lat, lon = ds[latn].values, ds[lonn].values
    med = _med_coarse_cells(mapper) if exclude_med else None
    masks = []
    for i in range(n_months):
        ti = min(t0 + i, len(ds.time) - 1)
        field = ds[var].isel(time=ti).values
        m = mapper.regrid_mask(lat, lon, ~np.isnan(field))
        if med is not None:
            m = m & ~med
        masks.append(m)
    ds.close()
    pct = np.mean([m.sum() / mapper.n_cells * 100 for m in masks])
    logger.info(f"    avg {pct:.1f}% coarse cells observed / month"
                + (" (Med excluded)" if exclude_med else ""))
    return masks


# =============================================================================
# Realistic Multi-Variable Observation Operator
# =============================================================================

class RealisticMultiVarObs:
    """
    Multi-variable observation operator with per-channel coarse-graining
    and real observation masks.

    Builds:
      - y: flat observation vector (coarse-grained GT + noise)
      - A: differentiable callable  x(1,C,L,H) → y_pred(n_obs,)
      - Per-channel mask info for visualization
    """

    def __init__(
        self,
        gt: torch.Tensor,          # (1, C, L, H)
        n_depths: int,
        hadsst_path: str,
        crutem_path: str,
        icoads_path: str,
        en4_path: str,
        start_dt: dt.datetime,
        noise_std: float = 0.15,
        device: str = 'cuda',
        condition_months: int | None = None,
        observe_ssh: bool = False,
        duacs_path: str | None = None,
        ssh_coarsen_deg: float = 5.0,
        ssh_var: str = 'sla',
        ssh_exclude_med: bool = True,
    ):
        B, C, L, H = gt.shape
        self.C, self.L, self.H = C, L, H
        self.device = device
        self.condition_months = condition_months

        # ---- channel indices --------------------------------------------------
        self.thetao_chs = list(range(0, n_depths))
        self.so_chs     = list(range(n_depths, 2 * n_depths))
        base_2d = 2 * n_depths
        self.tos_ch  = base_2d + DEFAULT_2D_VARS.index("tos")
        self.psl_ch  = base_2d + DEFAULT_2D_VARS.index("psl")
        self.tas_ch  = base_2d + DEFAULT_2D_VARS.index("tas")
        sos_ch       = base_2d + DEFAULT_2D_VARS.index("sos")
        zos_ch       = base_2d + DEFAULT_2D_VARS.index("zos")
        siconc_ch    = base_2d + DEFAULT_2D_VARS.index("siconc")
        self.unobserved = {sos_ch, zos_ch, siconc_ch}

        # ---- coarse grid mappers ----------------------------------------------
        logger.info("Creating coarse-grid mappers ...")
        self.mapper_2deg = CoarseGridMapper(2.0, device)
        self.mapper_4deg = CoarseGridMapper(4.0, device)
        self.mapper_5deg = CoarseGridMapper(5.0, device)

        # ---- load masks (aligned to the same start date as GT) ----------------
        logger.info(f"Loading observation masks from data products "
                    f"(start={start_dt:%Y-%m}, L={L}) ...")
        hadsst_masks = load_hadsst_masks(hadsst_path, self.mapper_5deg, start_dt, L)
        crutem_masks = load_crutem_masks(crutem_path,  self.mapper_5deg, start_dt, L)
        icoads_masks = load_icoads_masks(icoads_path,  self.mapper_2deg, start_dt, L)
        en4_thetao   = load_en4_masks(en4_path, self.mapper_4deg, start_dt, L, 'potm_mean')
        en4_so       = load_en4_masks(en4_path, self.mapper_4deg, start_dt, L, 'psal_mean')

        # ---- per-channel obs: ch -> (mapper, [masks_t]) ----------------------
        self.channel_obs: dict = {}
        for i, ch in enumerate(self.thetao_chs):
            if i < len(en4_thetao):
                self.channel_obs[ch] = (self.mapper_4deg, en4_thetao[i])
        for i, ch in enumerate(self.so_chs):
            if i < len(en4_so):
                self.channel_obs[ch] = (self.mapper_4deg, en4_so[i])
        self.channel_obs[self.tos_ch] = (self.mapper_5deg, hadsst_masks)
        self.channel_obs[self.psl_ch] = (self.mapper_2deg, icoads_masks)
        self.channel_obs[self.tas_ch] = (self.mapper_5deg, crutem_masks)

        # ---- optional SSH (zos) pseudo-obs from GT over DUACS coverage (OSSE) -
        if observe_ssh:
            if duacs_path is None:
                raise ValueError("observe_ssh=True requires duacs_path")
            ssh_mapper = (self.mapper_5deg
                          if abs(ssh_coarsen_deg - 5.0) < 1e-6
                          else CoarseGridMapper(ssh_coarsen_deg, device))
            duacs_masks = load_duacs_masks(
                duacs_path, ssh_mapper, start_dt, L, var=ssh_var,
                exclude_med=ssh_exclude_med)
            self.channel_obs[zos_ch] = (ssh_mapper, duacs_masks)
            self.unobserved = self.unobserved - {zos_ch}
            logger.info(f"SSH OSSE: observing zos (ch {zos_ch}) at "
                        f"{ssh_coarsen_deg}deg from GT over DUACS coverage "
                        f"(var={ssh_var}, exclude_med={ssh_exclude_med})")

        # ---- forecast masking: keep obs only for first condition_months ------
        if condition_months is not None and condition_months < L:
            n_zeroed = 0
            for ch, (mapper, masks) in self.channel_obs.items():
                for t in range(L):
                    if t >= condition_months and masks[t].any():
                        masks[t][:] = False
                        n_zeroed += 1
            logger.info(f"Forecast mode: conditioning on first "
                        f"{condition_months}/{L} months; masked obs for "
                        f"the remaining {L - condition_months} months "
                        f"({n_zeroed} channel-months zeroed).")

        # ---- build flat y vector AND A(x) mapping ----------------------------
        # Group (ch, t) pairs by mapper for efficient batched coarsening
        logger.info("Pre-computing coarse-grained GT observations ...")
        mapper_to_items: dict = {}   # id(mapper) -> (mapper, [(ch, t, obs_idx)])

        for ch, (mapper, masks) in self.channel_obs.items():
            mid = id(mapper)
            if mid not in mapper_to_items:
                mapper_to_items[mid] = (mapper, [])
            for t in range(L):
                if not masks[t].any():
                    continue
                obs_idx = torch.where(
                    torch.as_tensor(masks[t], device=device, dtype=torch.bool)
                )[0]
                mapper_to_items[mid][1].append((ch, t, obs_idx))

        self.mapper_groups = list(mapper_to_items.values())

        # Build y in the same grouped order as A(x) will produce
        y_parts = []
        for mapper, items in self.mapper_groups:
            for ch, t, obs_idx in items:
                gt_field  = gt[0, ch, t]              # (H,) on device
                gt_coarse = mapper.coarsen(gt_field)   # (n_cells,)
                y_parts.append(gt_coarse[obs_idx].detach())

        self.y_clean = torch.cat(y_parts).to(device)
        self.noise_std = noise_std
        self.y = self.y_clean.clone()
        if noise_std > 0:
            self.y = self.y + torch.randn_like(self.y) * noise_std

        self.n_obs = len(self.y)
        logger.info(f"RealisticMultiVarObs: {len(self.channel_obs)} observed channels, "
                     f"{self.n_obs:,} total observations")
        logger.info(f"  Unobserved channels: {sorted(self.unobserved)}")

    def perturb_observations(self, seed: int):
        """Re-sample observation noise for ensemble diversity (perturbed-obs EnKF style)."""
        g = torch.Generator(device=self.y_clean.device).manual_seed(seed)
        self.y = self.y_clean + torch.randn(
            self.y_clean.shape, device=self.y_clean.device,
            dtype=self.y_clean.dtype, generator=g,
        ) * self.noise_std

    def A(self, x: torch.Tensor) -> torch.Tensor:
        """
        Differentiable observation operator.

        x: (1, C, L, H) denoised prediction
        Returns: (n_obs,) predicted observation values
        """
        parts = []
        for mapper, items in self.mapper_groups:
            # Batch all fields for this mapper
            fields = torch.stack([x[0, ch, t, :] for ch, t, _ in items])
            coarsened = mapper.coarsen_batch(fields)   # (N, n_cells)
            for i, (_, _, obs_idx) in enumerate(items):
                parts.append(coarsened[i][obs_idx])
        return torch.cat(parts)


# =============================================================================
# Observation field builder (for saving to results.pt)
# =============================================================================

def build_obs_field(obs, gt):
    """Build (C, L, H) observation field in PAD_XY order for saving.

    Each HEALPix pixel gets the coarse-grained GT value of its coarse cell
    when that cell is observed, NaN otherwise.  Unobserved channels are
    entirely NaN.
    """
    C, L, H = obs.C, obs.L, obs.H
    obs_field = np.full((C, L, H), np.nan, dtype=np.float32)
    gt_np = gt[0].cpu().numpy()

    for ch, (mapper, masks) in obs.channel_obs.items():
        h2c = mapper.hpx_to_coarse.cpu().numpy()
        cnt = mapper.counts.cpu().numpy()
        for t in range(L):
            if not masks[t].any():
                continue
            gt_c = np.zeros(mapper.n_cells, dtype=np.float64)
            np.add.at(gt_c, h2c, gt_np[ch, t])
            gt_c /= cnt
            o_hpx = gt_c[h2c].astype(np.float32)
            o_hpx[~masks[t][h2c]] = np.nan
            obs_field[ch, t] = o_hpx

    return torch.from_numpy(obs_field)



# =============================================================================
# Visualization
# =============================================================================

def _prepare_ring_data(gen, gt, ch_names, land_mask_nest):
    gen_np = gen[0].cpu().numpy()
    gt_np  = gt[0].cpu().numpy()

    grid = Grid(level=6, pixel_order=PixelOrder.RING)
    lon = np.where(np.array(grid.lon) > 180,
                   np.array(grid.lon) - 360, np.array(grid.lon))
    lat = np.array(grid.lat)

    gen_ring = earth2grid.healpix.reorder(
        torch.from_numpy(gen_np), HEALPIX_PAD_XY, PixelOrder.RING).numpy()
    gt_ring = earth2grid.healpix.reorder(
        torch.from_numpy(gt_np), HEALPIX_PAD_XY, PixelOrder.RING).numpy()

    lm_ring = None
    if land_mask_nest is not None:
        lm_ring = earth2grid.healpix.reorder(
            torch.as_tensor(land_mask_nest, dtype=torch.float32),
            PixelOrder.NEST, PixelOrder.RING).numpy() > 0.5

    return gen_np, gt_np, gen_ring, gt_ring, lm_ring, lon, lat


def _build_obs_ring(channels, obs, gt_np, L, H):
    """Pre-compute coarse-grained obs mapped back to RING for visualization."""
    obs_ring: dict = {}
    for ch in channels:
        if ch not in obs.channel_obs:
            continue
        mapper, masks = obs.channel_obs[ch]
        h2c = mapper.hpx_to_coarse.cpu().numpy()
        cnt = mapper.counts.cpu().numpy()
        for t in range(L):
            mask = masks[t]
            if not mask.any():
                obs_ring[(ch, t)] = np.full(H, np.nan)
                continue
            gt_c = np.zeros(mapper.n_cells)
            np.add.at(gt_c, h2c, gt_np[ch, t])
            gt_c /= cnt
            o_hpx = gt_c[h2c]
            o_hpx[~mask[h2c]] = np.nan
            obs_ring[(ch, t)] = earth2grid.healpix.reorder(
                torch.from_numpy(o_hpx.astype(np.float32)),
                HEALPIX_PAD_XY, PixelOrder.RING).numpy()
    return obs_ring


def _make_one_gif(channels, group_label, gen_ring, gt_ring, lm_ring,
                  lon, lat, gt_np, obs, obs_ring, ch_names,
                  L, H, out_dir, fps):
    n_cols = len(channels)
    if n_cols == 0:
        return

    try:
        import cartopy.crs as ccrs
        has_carto = True
    except ImportError:
        has_carto = False

    vr = {}
    for ch in channels:
        nm = ch_names[ch] if ch < len(ch_names) else f"ch{ch}"
        vals = np.concatenate([gen_ring[ch].ravel(), gt_ring[ch].ravel()])
        if lm_ring is not None and is_ocean_var(nm):
            ocean_tile = ~np.tile(lm_ring[ch], 2 * L)
            if ocean_tile.shape[0] == vals.shape[0]:
                vals = vals[ocean_tile]
        v = vals[np.isfinite(vals)]
        vr[ch] = (float(np.percentile(v, 2)),
                  float(np.percentile(v, 98))) if len(v) > 0 else (-1, 1)

    frames_imgs = []
    for t in tqdm(range(L), desc=f"GIF [{group_label}]"):
        kw = {'subplot_kw': {'projection': ccrs.Robinson()}} if has_carto else {}
        fig, axes = plt.subplots(3, n_cols, figsize=(3.5 * n_cols, 9), **kw)
        if n_cols == 1:
            axes = axes[:, np.newaxis]

        for ci, ch in enumerate(channels):
            nm = ch_names[ch] if ch < len(ch_names) else f"ch{ch}"
            vmin, vmax = vr[ch]
            is_oc = is_ocean_var(nm)

            # --- Row 0: Observations ---
            ax = axes[0, ci]
            d = obs_ring.get((ch, t), np.full(H, np.nan))
            vi = np.isfinite(d)
            if has_carto:
                ax.set_global()
                ax.coastlines(linewidth=0.3)
                if vi.any():
                    ax.scatter(lon[vi], lat[vi], c=d[vi], s=0.3,
                               transform=ccrs.PlateCarree(),
                               cmap="coolwarm", vmin=vmin, vmax=vmax)
            else:
                ax.set_xlim(-180, 180)
                ax.set_ylim(-90, 90)
                if vi.any():
                    ax.scatter(lon[vi], lat[vi], c=d[vi], s=0.3,
                               cmap="coolwarm", vmin=vmin, vmax=vmax)
            n_obs_str = ""
            if ch in obs.channel_obs:
                _, ms = obs.channel_obs[ch]
                n_obs_str = f" ({ms[t].sum()})"
            ax.set_title(f"Obs: {nm}{n_obs_str}", fontsize=6)

            # --- Rows 1-2: Truth / Generated ---
            for row, (src, lbl) in enumerate(
                [(gt_ring, "Truth"), (gen_ring, "Gen")]
            ):
                ax = axes[row + 1, ci]
                d = src[ch, t].copy()
                if lm_ring is not None and is_oc:
                    d[lm_ring[ch]] = np.nan
                vi = np.isfinite(d)
                if has_carto:
                    ax.set_global()
                    ax.coastlines(linewidth=0.3)
                    if vi.any():
                        ax.scatter(lon[vi], lat[vi], c=d[vi], s=0.3,
                                   transform=ccrs.PlateCarree(),
                                   cmap="coolwarm", vmin=vmin, vmax=vmax)
                else:
                    ax.set_xlim(-180, 180)
                    ax.set_ylim(-90, 90)
                    if vi.any():
                        ax.scatter(lon[vi], lat[vi], c=d[vi], s=0.3,
                                   cmap="coolwarm", vmin=vmin, vmax=vmax)
                ax.set_title(f"{lbl}: {nm}", fontsize=6)

        fig.suptitle(f"{group_label}  t={t+1}/{L}", fontsize=10, fontweight='bold')
        plt.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        frames_imgs.append(Image.open(buf).copy())
        buf.close()

    safe_label = group_label.lower().replace(" ", "_").replace("/", "_")
    gif_path = os.path.join(out_dir, f"realistic_obs_{safe_label}.gif")
    frames_imgs[0].save(gif_path, save_all=True,
                        append_images=frames_imgs[1:],
                        duration=int(1000 / fps), loop=0)
    logger.info(f"Saved {gif_path}  ({n_cols} channels)")


def _make_polar_gif(channels, group_label, gen_ring, gt_ring, lm_ring,
                    lon, lat, gt_np, obs, obs_ring, ch_names,
                    L, H, out_dir, fps):
    """Create GIF with North-Pole and South-Pole stereographic views for sea-ice."""
    n_vars = len(channels)
    if n_vars == 0:
        return

    try:
        import cartopy.crs as ccrs
        has_carto = True
    except ImportError:
        has_carto = False
        logger.warning("cartopy not available – falling back to Robinson for polar plots")
        _make_one_gif(channels, group_label, gen_ring, gt_ring, lm_ring,
                      lon, lat, gt_np, obs, obs_ring, ch_names,
                      L, H, out_dir, fps)
        return

    lat_np = np.asarray(lat)
    lon_np = np.asarray(lon)

    # Polar latitude thresholds
    north_mask = lat_np >= 40.0
    south_mask = lat_np <= -40.0

    # Colour range
    vr = {}
    for ch in channels:
        nm = ch_names[ch] if ch < len(ch_names) else f"ch{ch}"
        vals = np.concatenate([gen_ring[ch].ravel(), gt_ring[ch].ravel()])
        if lm_ring is not None and is_ocean_var(nm):
            ocean_tile = ~np.tile(lm_ring[ch], 2 * L)
            if ocean_tile.shape[0] == vals.shape[0]:
                vals = vals[ocean_tile]
        v = vals[np.isfinite(vals)]
        vr[ch] = (float(np.percentile(v, 2)),
                  float(np.percentile(v, 98))) if len(v) > 0 else (0, 1)

    # Two polar projections
    proj_north = ccrs.NorthPolarStereo()
    proj_south = ccrs.SouthPolarStereo()

    # Layout: 3 rows (Obs / Truth / Gen) x 2*n_vars cols (North, South per var)
    n_cols = 2 * n_vars
    frames_imgs = []

    for t in tqdm(range(L), desc=f"GIF [{group_label}]"):
        fig, axes = plt.subplots(
            3, n_cols,
            figsize=(3.2 * n_cols, 9),
            subplot_kw={'projection': proj_north},  # placeholder, overridden below
        )
        if n_cols == 1:
            axes = axes[:, np.newaxis]

        for vi_idx, ch in enumerate(channels):
            nm = ch_names[ch] if ch < len(ch_names) else f"ch{ch}"
            vmin, vmax = vr[ch]
            is_oc = is_ocean_var(nm)

            for pole_idx, (proj, pmask, pole_lbl, extent) in enumerate([
                (proj_north, north_mask, "N", [-180, 180, 40, 90]),
                (proj_south, south_mask, "S", [-180, 180, -90, -40]),
            ]):
                col = vi_idx * 2 + pole_idx

                # --- Row 0: Observations ---
                ax = axes[0, col]
                ax.remove()
                ax = fig.add_subplot(3, n_cols, 0 * n_cols + col + 1, projection=proj)
                axes[0, col] = ax
                ax.set_extent(extent, crs=ccrs.PlateCarree())
                ax.coastlines(linewidth=0.4)
                # Land pixels are already NaN-masked via lm_ring; plot
                # land-mask dots as light grey background so coast is visible
                if lm_ring is not None:
                    land_sel = pmask & lm_ring[ch]
                    if land_sel.any():
                        ax.scatter(lon_np[land_sel], lat_np[land_sel],
                                   c='#d9d9d9', s=0.4, transform=ccrs.PlateCarree(),
                                   zorder=0, edgecolors='none')
                d = obs_ring.get((ch, t), np.full(H, np.nan))
                sel = pmask & np.isfinite(d)
                if sel.any():
                    ax.scatter(lon_np[sel], lat_np[sel], c=d[sel], s=0.4,
                               transform=ccrs.PlateCarree(),
                               cmap="coolwarm", vmin=vmin, vmax=vmax, zorder=2)
                n_obs_str = ""
                if ch in obs.channel_obs:
                    _, ms = obs.channel_obs[ch]
                    n_obs_str = f" ({ms[t].sum()})"
                ax.set_title(f"Obs {pole_lbl}: {nm}{n_obs_str}", fontsize=6)

                # --- Rows 1-2: Truth / Generated ---
                for row, (src, lbl) in enumerate(
                    [(gt_ring, "Truth"), (gen_ring, "Gen")]
                ):
                    ax = axes[row + 1, col]
                    ax.remove()
                    ax = fig.add_subplot(3, n_cols, (row + 1) * n_cols + col + 1, projection=proj)
                    axes[row + 1, col] = ax
                    ax.set_extent(extent, crs=ccrs.PlateCarree())
                    ax.coastlines(linewidth=0.4)
                    if lm_ring is not None:
                        land_sel = pmask & lm_ring[ch]
                        if land_sel.any():
                            ax.scatter(lon_np[land_sel], lat_np[land_sel],
                                       c='#d9d9d9', s=0.4, transform=ccrs.PlateCarree(),
                                       zorder=0, edgecolors='none')
                    d = src[ch, t].copy()
                    if lm_ring is not None and is_oc:
                        d[lm_ring[ch]] = np.nan
                    sel = pmask & np.isfinite(d)
                    if sel.any():
                        ax.scatter(lon_np[sel], lat_np[sel], c=d[sel], s=0.4,
                                   transform=ccrs.PlateCarree(),
                                   cmap="coolwarm", vmin=vmin, vmax=vmax, zorder=2)
                    ax.set_title(f"{lbl} {pole_lbl}: {nm}", fontsize=6)

        fig.suptitle(f"{group_label}  t={t+1}/{L}", fontsize=10, fontweight='bold')
        plt.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        frames_imgs.append(Image.open(buf).copy())
        buf.close()

    safe_label = group_label.lower().replace(" ", "_").replace("/", "_")
    gif_path = os.path.join(out_dir, f"realistic_obs_{safe_label}.gif")
    frames_imgs[0].save(gif_path, save_all=True,
                        append_images=frames_imgs[1:],
                        duration=int(1000 / fps), loop=0)
    logger.info(f"Saved {gif_path}  ({n_vars} channels, polar stereo)")


def create_all_gifs(gen, gt, obs, ch_names, land_mask_nest,
                    out_dir, n_depths, fps=2, custom_channels=None):
    os.makedirs(out_dir, exist_ok=True)
    C = gen.shape[1]
    L = gen.shape[2]
    H = gen.shape[3]

    gen_np, gt_np, gen_ring, gt_ring, lm_ring, lon, lat = \
        _prepare_ring_data(gen, gt, ch_names, land_mask_nest)

    if custom_channels is not None:
        channels = [c for c in custom_channels if c < C]
        obs_ring = _build_obs_ring(channels, obs, gt_np, L, H)
        _make_one_gif(channels, "Custom", gen_ring, gt_ring, lm_ring,
                      lon, lat, gt_np, obs, obs_ring, ch_names,
                      L, H, out_dir, fps)
        return

    # Group 1: thetao depths
    thetao_chs = list(range(0, min(n_depths, C)))
    obs_ring_th = _build_obs_ring(thetao_chs, obs, gt_np, L, H)
    _make_one_gif(thetao_chs, "Thetao (depth profiles)",
                  gen_ring, gt_ring, lm_ring,
                  lon, lat, gt_np, obs, obs_ring_th, ch_names,
                  L, H, out_dir, fps)

    # Group 2: so depths
    so_chs = list(range(n_depths, min(2 * n_depths, C)))
    obs_ring_so = _build_obs_ring(so_chs, obs, gt_np, L, H)
    _make_one_gif(so_chs, "So (depth profiles)",
                  gen_ring, gt_ring, lm_ring,
                  lon, lat, gt_np, obs, obs_ring_so, ch_names,
                  L, H, out_dir, fps)

    # Group 3: 2D surface vars (excluding siconc)
    surf_chs_all = list(range(2 * n_depths, C))
    siconc_chs = [ch for ch in surf_chs_all
                  if ch < len(ch_names) and ch_names[ch] == "siconc"]
    surf_chs = [ch for ch in surf_chs_all if ch not in siconc_chs]
    obs_ring_2d = _build_obs_ring(surf_chs, obs, gt_np, L, H)
    _make_one_gif(surf_chs, "Surface 2D vars",
                  gen_ring, gt_ring, lm_ring,
                  lon, lat, gt_np, obs, obs_ring_2d, ch_names,
                  L, H, out_dir, fps)

    # Group 4: Sea-ice (siconc) — polar stereographic
    if siconc_chs:
        obs_ring_ice = _build_obs_ring(siconc_chs, obs, gt_np, L, H)
        _make_polar_gif(siconc_chs, "Sea ice (siconc)",
                        gen_ring, gt_ring, lm_ring,
                        lon, lat, gt_np, obs, obs_ring_ice, ch_names,
                        L, H, out_dir, fps)




def sample_sda_realistic(
    net,
    gt: torch.Tensor,
    months: torch.Tensor,
    obs: RealisticMultiVarObs,
    sigma_obs: float,
    steps: int,
    corrections: int,
    tau: float,
    chunk_windows: int,
    device: torch.device,
    output_frames: int = 3,
    gamma: float = 0.1,
    use_bf16: bool = False,
    world_size: int = 1,
    rank: int = 0,
    parallel_sda: bool = True,
    use_checkpoint: bool = True,
    corrector_t_threshold: float = 0.0,
):
    """
    Run SDA — either with the new patched/parallel approach or the old
    global approach.

    When ``parallel_sda=True`` (default), uses ``PatchedRealisticScore``
    with a shared computation graph so that ``autograd.grad`` on one
    shared ``x`` tensor gives cross-window gradient coupling (≈ global
    quality).  ``use_checkpoint`` toggles gradient checkpointing (trades
    ~1.5× compute for constant memory).

    ``corrector_t_threshold`` (0–1): only run corrector steps when
    diffusion time ``t > threshold``.  E.g. 0.5 = corrections only in
    the noisy first half of sampling.  Default 0.0 = corrections at all
    noise levels.

    When ``parallel_sda=False``, falls back to the original global
    ``VPSDE_from_denoiser`` + ``GaussianScore_from_denoiser`` pipeline.
    """
    B, C, L, H = gt.shape
    assert B == 1

    if parallel_sda:
        # ---- New patched approach ----
        T = int(net.time_length)
        edm = EDMWrapperProfiles(net, use_bf16=use_bf16).to(device)

        score = PatchedRealisticScore(
            edm=edm, obs=obs, months=months,
            T=T, C=C, H=H, L=L,
            output_frames=output_frames,
            chunk_windows=chunk_windows,
            gamma=gamma, sigma_obs=sigma_obs,
            world_size=world_size, rank=rank,
            use_checkpoint=use_checkpoint,
        ).to(device)

        posterior_sde = VPSDE(
            eps=score, shape=torch.Size((C, L, H)),
        ).to(device)

        t_thr_str = (f", corrector only t>{corrector_t_threshold:.2f}"
                     if corrections > 0 and corrector_t_threshold < 1.0 else "")
        logger.info(
            f"Running patched SDA: steps={steps}, corrections={corrections}, "
            f"tau={tau}, gamma={gamma}, n_obs={obs.n_obs:,}, "
            f"n_windows={score.n_windows}, world_size={world_size}"
            f"{t_thr_str}"
        )

        x = _sample_with_threshold(
            posterior_sde, score, B, C, L, H,
            steps=steps, corrections=corrections, tau=tau,
            corrector_t_threshold=corrector_t_threshold,
        )
        return x

    else:
        # ---- Old global approach ----
        denoiser = LongDenoiserProfiles(
            net=net, months=months, C=C, H=H, device=device,
            chunk_windows=chunk_windows, output_frames=output_frames,
            use_bf16=use_bf16,
        ).to(device)

        prior_sde = VPSDE_from_denoiser(
            denoiser, shape=torch.Size(()),
        ).to(device)

        y = obs.y
        A = obs.A
        std_eff = torch.as_tensor(float(sigma_obs), device=device,
                                  dtype=gt.dtype)

        post_eps = GaussianScore_from_denoiser(
            y=y, A=A, std=std_eff, sde=prior_sde, gamma=gamma, detach=True,
        ).to(device)

        posterior_sde = VPSDE(
            eps=post_eps, shape=torch.Size((C, L, H)),
        ).to(device)

        logger.info(
            f"Running global SDA: steps={steps}, corrections={corrections}, "
            f"tau={tau}, gamma={gamma}, n_obs={obs.n_obs:,}"
        )
        x = posterior_sde.sample(
            shape=torch.Size((B,)), c=None, steps=steps,
            corrections=corrections, tau=tau, makefigs=False,
        )
        return x


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="MIROC6 profiles SDA with realistic obs (library-based sampler)"
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--zarr_path", required=True)
    p.add_argument("--climatology", required=True)

    # Data product paths
    p.add_argument("--hadsst_path", default=None)
    p.add_argument("--crutem_path", default=None)
    p.add_argument("--icoads_path", default=None)
    p.add_argument("--en4_path", default=None)

    # Member selection
    p.add_argument("--member", type=str, default="r47i1p1f1",
                   help="Ensemble member ID (e.g. r47i1p1f1). Default: r47i1p1f1.")
    p.add_argument("--train_ratio", type=float, default=1.0,
                   help="Time-based train/test split ratio. "
                        "Set to 1.0 for full range (member-based validation).")

    # Time range (YYYY-MM format)
    p.add_argument("--start_date", type=str, default="1990-01",
                   help="Start date YYYY-MM")
    p.add_argument("--end_date", type=str, default="1991-12",
                   help="End date YYYY-MM (inclusive)")

    # Observation parameters
    p.add_argument("--noise_obs_std", type=float, default=0.15)
    p.add_argument("--gamma", type=float, default=0.001)

    # SSH pseudo-obs (perfect-model OSSE: zos sampled from GT)
    p.add_argument("--observe_ssh", action="store_true", default=False,
                   help="Assimilate zos (SSH) sampled from the MIROC6 truth "
                        "over DUACS spatial coverage (OSSE).")
    p.add_argument("--duacs_path", default=None)
    p.add_argument("--ssh_coarsen_deg", type=float, default=5.0)
    p.add_argument("--ssh_var", default="sla")
    p.add_argument("--ssh_keep_med", action="store_true", default=False,
                   help="Keep Mediterranean SSH obs (default: excluded).")

    # Sampling
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--corrections", type=int, default=0)
    p.add_argument("--tau", type=float, default=0.3)
    p.add_argument("--corrector_t_threshold", type=float, default=0.0,
                   help="Min diffusion time for corrections (0=all, 0.5=high-noise only)")
    p.add_argument("--output_frames", type=int, default=5)
    p.add_argument("--chunk_windows", type=int, default=2,
                   help="Windows per GPU micro-batch")

    # Parallel SDA (new default)
    p.add_argument("--parallel_sda", action="store_true", default=True,
                   help="Use patched parallel SDA (default)")
    p.add_argument("--no_parallel_sda", dest="parallel_sda",
                   action="store_false",
                   help="Fall back to old global SDA")
    p.add_argument("--no_checkpoint", dest="use_checkpoint",
                   action="store_false", default=True,
                   help="Disable gradient checkpointing (uses more memory, ~0.75x time)")

    # Ensemble
    p.add_argument("--n_ensemble", type=int, default=1,
                   help="Total number of ensemble members to generate")
    p.add_argument("--ensemble_id", type=int, default=None,
                   help="Generate only this member (0-indexed, for parallel "
                        "jobs). If not set, generates all n_ensemble members "
                        "sequentially.")

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="out_profiles_realistic_obs_v2")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fps", type=int, default=2)
    p.add_argument("--use_bf16", action="store_true")
    p.add_argument("--no_gifs", action="store_true",
                   help="Skip GIF generation (only save results.pt and args.json)")
    p.add_argument("--channels_to_plot", type=int, nargs="+", default=None)
    args = p.parse_args()

    # ---- multi-GPU setup ------------------------------------------------------
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        if rank != 0:
            logging.getLogger("profiles_realistic_v4").setLevel(logging.WARNING)
        logger.info(f"Distributed: rank={rank}/{world_size}, device={device}")
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- load model -----------------------------------------------------------
    net = load_checkpoint(args.checkpoint, device)
    land_mask_nest, depths, ch_names_clim = load_land_mask_and_depths(args.climatology)
    n_depths = len(depths)

    T = int(net.time_length)
    C = int(net.img_channels)
    H = int(net.domain.numel())
    logger.info(f"Model: C={C}, T={T}, H={H}")

    # ---- load GT from historical member -----------------------------------------
    gt, months, ch_names, actual_start_dt = load_gt_historical(
        args.zarr_path, args.climatology,
        args.start_date, args.end_date, device,
        train_ratio=args.train_ratio,
        member=args.member,
    )
    L = gt.shape[2]
    if L < T:
        raise ValueError(
            f"Sequence length L={L} is shorter than model window T={T}. "
            f"Request a longer date range (>= {T} months) or use "
            f"--train_ratio 1.0 if the requested dates fall outside the "
            f"test subset."
        )

    # ---- adjust L for clean windowing (parallel_sda mode) ---------------------
    if args.parallel_sda:
        L_eff, n_windows, L_output = adjust_L_for_windows(
            L, T, args.output_frames,
        )
        if L_eff < L:
            logger.info(
                f"Trimming sequence for windowing: L={L} -> L_eff={L_eff} "
                f"({n_windows} windows, {L_output} output frames)"
            )
            gt = gt[:, :, :L_eff, :]
            months = months[:L_eff]
            L = L_eff

    # ---- build realistic observation operator (aligned to same dates) ---------
    obs = RealisticMultiVarObs(
        gt=gt, n_depths=n_depths,
        hadsst_path=args.hadsst_path,
        crutem_path=args.crutem_path,
        icoads_path=args.icoads_path,
        en4_path=args.en4_path,
        start_dt=actual_start_dt,
        noise_std=args.noise_obs_std,
        device=str(device),
        observe_ssh=args.observe_ssh,
        duacs_path=args.duacs_path,
        ssh_coarsen_deg=args.ssh_coarsen_deg,
        ssh_var=args.ssh_var,
        ssh_exclude_med=not args.ssh_keep_med,
    )

    if args.use_bf16:
        logger.info("BF16 autocast ENABLED")

    # ---- determine ensemble members to generate --------------------------------
    if args.ensemble_id is not None:
        member_ids = [args.ensemble_id]
    else:
        member_ids = list(range(args.n_ensemble))

    logger.info(f"Ensemble members to generate: {member_ids} "
                f"(n_ensemble={args.n_ensemble})")

    # ---- build observation field for saving ------------------------------------
    obs_field = build_obs_field(obs, gt)
    n_obs_px = int(obs_field.isfinite().sum())
    logger.info(f"Observation field: {obs_field.shape}, "
                f"observed pixels: {n_obs_px:,}/{obs_field.numel():,}")

    # ---- run SDA for each ensemble member --------------------------------------
    generated_members = []
    for mid in member_ids:
        member_seed = args.seed + 1_000_000 + mid
        torch.manual_seed(member_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(member_seed)

        if args.n_ensemble > 1 and args.noise_obs_std > 0:
            obs.perturb_observations(seed=member_seed + 777)
            logger.info(f"Generating ensemble member {mid} "
                        f"(seed={member_seed}, perturbed obs)")
        else:
            logger.info(f"Generating ensemble member {mid} "
                        f"(seed={member_seed})")

        x = sample_sda_realistic(
            net=net, gt=gt, months=months, obs=obs,
            sigma_obs=args.noise_obs_std,
            steps=args.steps, corrections=args.corrections,
            tau=args.tau, chunk_windows=args.chunk_windows,
            device=device, output_frames=args.output_frames,
            gamma=args.gamma, use_bf16=args.use_bf16,
            world_size=world_size, rank=rank,
            parallel_sda=args.parallel_sda,
            use_checkpoint=args.use_checkpoint,
            corrector_t_threshold=args.corrector_t_threshold,
        )
        logger.info(f"  Member {mid} done: {x.shape}")
        generated_members.append(x.cpu())

    # (n_members, C, L, H)
    x_ensemble = torch.cat(generated_members, dim=0)
    gt_out = gt
    months_out = months

    # ---- save + GIF (rank 0 only) -----------------------------------------------
    if rank == 0:
        result_dict = {
            "generated": x_ensemble,
            "ground_truth": gt_out.cpu(),
            "observations": obs_field,
            "observations_y": obs.y.cpu(),
            "months": months_out.cpu(),
            "ensemble_member_ids": member_ids,
            "args": vars(args),
        }

        if args.ensemble_id is not None:
            fname = f"results_e{args.ensemble_id:04d}.pt"
        else:
            fname = "results.pt"

        torch.save(result_dict, os.path.join(args.output_dir, fname))

        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

        logger.info(f"Results saved to {args.output_dir}/{fname}")

        if not args.no_gifs:
            x_for_viz = x_ensemble[0:1].to(device)
            create_all_gifs(
                x_for_viz, gt_out, obs, ch_names, land_mask_nest,
                args.output_dir, n_depths=n_depths, fps=args.fps,
                custom_channels=args.channels_to_plot,
            )

        logger.info(f"Done! Results in {args.output_dir}/")

    # ---- cleanup distributed --------------------------------------------------
    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
