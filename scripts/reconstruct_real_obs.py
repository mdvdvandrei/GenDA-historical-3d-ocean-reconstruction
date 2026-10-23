#!/usr/bin/env python3
"""
Real-obs reconstruction (HadSST / CRUTEM / ICOADS / EN4, optional DUACS SSH).

  python scripts/reconstruct_real_obs.py \\
    --checkpoint snapshot.pkl --start_date 1993-01 --end_date 2020-12 \\
    --use_ssh_obs --noise_obs_std 0.15 --gamma 0.001 \\
    --steps 128 --corrections 0 --use_bf16 --output_dir out_real_obs
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
import pandas as pd
from tqdm import tqdm
from PIL import Image

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

from sda.score import GaussianScore_from_denoiser, VPSDE_from_denoiser, VPSDE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("profiles_real_obs_v1")

OBS_CLIP = 5.0  # clamp normalised obs to ±OBS_CLIP
INV_STD_CAP_FACTOR = 3.0  # cap per-cell inv_std at factor × median





# =============================================================================
# Load checkpoint / climatology
# =============================================================================



def load_climatology(climatology_path: str):
    with open(climatology_path, "rb") as f:
        clim = pickle.load(f)
    land_mask = clim.get("land_mask")
    depths = clim.get("depths", np.array([25, 75, 150, 300, 500, 700, 1000, 1500, 2000, 9999], dtype=np.float32))
    channel_names = clim.get("channel_names", get_channel_names(depths))

    # Per-pixel std used during training: (C, H) in NEST order
    spatial_std_nest = clim.get("spatial_std")
    # Per-pixel monthly mean used during training: (12, C, H) in NEST order
    monthly_mean_nest = clim.get("mean")

    logger.info(f"Loaded climatology: land_mask={land_mask.shape if land_mask is not None else None}, "
                f"{len(channel_names)} channels, {len(depths)} depths, "
                f"spatial_std={spatial_std_nest.shape if spatial_std_nest is not None else None}, "
                f"monthly_mean={monthly_mean_nest.shape if monthly_mean_nest is not None else None}")
    return land_mask, depths, channel_names, spatial_std_nest, monthly_mean_nest





# =============================================================================
# Observation utilities
# =============================================================================

_native_to_hpx_cache: dict = {}


def _build_native_to_hpx_map(obs_lat, obs_lon):
    """Map each (lat, lon) native grid point to its nearest HEALPix PAD_XY pixel.

    Returns (n_lat, n_lon) int64 array of HEALPix pixel indices.
    Uses an approximate-great-circle distance, processed in chunks.
    Result is cached by (lat tuple, lon tuple) for reuse across loaders.
    """
    key = (tuple(np.round(obs_lat, 6)), tuple(np.round(obs_lon, 6)))
    if key in _native_to_hpx_cache:
        return _native_to_hpx_cache[key]

    hpx_grid = Grid(level=6, pixel_order=HEALPIX_PAD_XY)
    hpx_lat = np.array(hpx_grid.lat, dtype=np.float64)
    hpx_lon = np.array(hpx_grid.lon, dtype=np.float64)
    hpx_lon = np.where(hpx_lon > 180, hpx_lon - 360, hpx_lon)

    obs_lon_adj = np.where(np.asarray(obs_lon) > 180,
                           np.asarray(obs_lon) - 360.0, np.asarray(obs_lon))
    lon_m, lat_m = np.meshgrid(obs_lon_adj.astype(np.float64),
                               np.asarray(obs_lat, dtype=np.float64))
    flat_lat = lat_m.ravel()
    flat_lon = lon_m.ravel()

    result = np.empty(len(flat_lat), dtype=np.int64)
    chunk = 64
    for s in range(0, len(flat_lat), chunk):
        e = min(s + chunk, len(flat_lat))
        dlat = flat_lat[s:e, None] - hpx_lat[None, :]
        dlon = flat_lon[s:e, None] - hpx_lon[None, :]
        cos_lat = np.cos(np.deg2rad(flat_lat[s:e, None]))
        dist_sq = dlat ** 2 + (dlon * cos_lat) ** 2
        result[s:e] = np.argmin(dist_sq, axis=1)

    mapping = result.reshape(len(obs_lat), len(obs_lon))
    _native_to_hpx_cache[key] = mapping
    logger.info(f"    Native-to-HEALPix mapping: {mapping.shape} -> "
                f"{len(hpx_lat)} pixels (cached)")
    return mapping


def _pixel_normalize_field(anom_2d, std_hpx_ch, native_to_hpx):
    """Normalize a native-grid anomaly field per-pixel using HEALPix spatial_std.

    anom_2d:          (n_lat, n_lon)  raw anomaly (NaN where missing)
    std_hpx_ch:       (H,)           per-HEALPix-pixel spatial std
    native_to_hpx:    (n_lat, n_lon) int  nearest HEALPix pixel index

    Returns (n_lat, n_lon) float32 with NaN preserved, values clamped to ±OBS_CLIP.
    """
    norm = np.full_like(anom_2d, np.nan, dtype=np.float32)
    valid = np.isfinite(anom_2d)
    hpx_idx = native_to_hpx[valid]
    std_at_obs = std_hpx_ch[hpx_idx]
    norm[valid] = np.clip(
        anom_2d[valid] / std_at_obs * SIGMA_DATA,
        -OBS_CLIP, OBS_CLIP,
    ).astype(np.float32)
    return norm




def map_obs_to_coarse_cells(field_2d, obs_lat, obs_lon, mapper):
    """Map 2D observation field (lat, lon) to mapper's flat (n_cells,) grid.

    Handles arbitrary lat ordering and 0-360 / -180-180 lon conventions.
    Returns (n_cells,) float32 array with NaN for unobserved cells.
    Multiple obs in the same coarse cell are averaged.
    """
    obs_lon_adj = np.where(np.asarray(obs_lon) > 180,
                           np.asarray(obs_lon) - 360,
                           np.asarray(obs_lon))
    lon_mesh, lat_mesh = np.meshgrid(obs_lon_adj, np.asarray(obs_lat))
    lat_flat = lat_mesh.ravel()
    lon_flat = lon_mesh.ravel()
    vals = np.asarray(field_2d, dtype=np.float64).ravel()

    lat_bin = np.clip(np.digitize(lat_flat, mapper.lat_edges) - 1,
                      0, mapper.n_lat - 1)
    lon_bin = np.clip(np.digitize(lon_flat, mapper.lon_edges) - 1,
                      0, mapper.n_lon - 1)
    cell_idx = lat_bin * mapper.n_lon + lon_bin

    valid = np.isfinite(vals)
    accum = np.zeros(mapper.n_cells, dtype=np.float64)
    count = np.zeros(mapper.n_cells, dtype=np.int32)
    np.add.at(accum, cell_idx[valid], vals[valid])
    np.add.at(count, cell_idx[valid], 1)

    result = np.full(mapper.n_cells, np.nan, dtype=np.float32)
    has = count > 0
    result[has] = (accum[has] / count[has]).astype(np.float32)
    return result


# =============================================================================
# Coarsen model's per-pixel std to observation grids
# =============================================================================

def build_norm_stats(spatial_std_nest, monthly_mean_nest, channel_names,
                     n_depths, mappers):
    """Build per-cell inv_std and coarsened model monthly mean.

    For each coarse cell c:
        inv_std_coarse[c] = mean_{p in c}(1 / std[p])

    Capped at INV_STD_CAP_FACTOR × median to prevent outlier cells
    (land margins, low-variability deep ocean) from dominating the DPS
    likelihood gradient.

    Also coarsens the model monthly mean to each obs grid for optional
    bias correction (--use_model_clim).

    Returns:
        norm_stats dict  — keys like 'tos', 'thetao_0', …;
        each value is a dict with:
            'inv_std': (n_cells,) float32
            'model_mean_coarse': (12, n_cells) float32  — coarsened model monthly mean
    """
    import earth2grid.healpix

    std_padxy = earth2grid.healpix.reorder(
        torch.from_numpy(spatial_std_nest),
        PixelOrder.NEST, HEALPIX_PAD_XY,
    ).numpy()  # (C, H)

    mean_padxy = None
    if monthly_mean_nest is not None:
        mean_padxy = np.array([
            earth2grid.healpix.reorder(
                torch.from_numpy(monthly_mean_nest[m]),
                PixelOrder.NEST, HEALPIX_PAD_XY,
            ).numpy()
            for m in range(12)
        ])  # (12, C, H)

    stats: dict = {}

    for ch_idx, name in enumerate(channel_names):
        std_hpx = std_padxy[ch_idx].copy()  # (H,)
        std_hpx = np.maximum(std_hpx, 1e-6)

        if ch_idx < n_depths:
            key, mkey = f'thetao_{ch_idx}', 'thetao'
        elif ch_idx < 2 * n_depths:
            key, mkey = f'so_{ch_idx - n_depths}', 'so'
        elif name == "tos":
            key, mkey = 'tos', 'tos'
        elif name == "psl":
            key, mkey = 'psl', 'psl'
        elif name == "tas":
            key, mkey = 'tas', 'tas'
        elif name == "zos" and 'zos' in mappers:
            key, mkey = 'zos', 'zos'
        else:
            continue

        mapper = mappers[mkey]

        inv_std_hpx = 1.0 / std_hpx  # (H,)
        inv_std_coarse = mapper.coarsen(
            torch.from_numpy(inv_std_hpx).float()
        ).cpu().numpy()  # (n_cells,)

        ocean = inv_std_coarse[inv_std_coarse > 0]
        med_inv = float(np.median(ocean)) if len(ocean) > 0 else 1.0
        cap = med_inv * INV_STD_CAP_FACTOR
        inv_std_coarse = np.clip(inv_std_coarse, 0, cap)
        inv_std_coarse = np.where(
            inv_std_coarse > 0, inv_std_coarse, med_inv
        ).astype(np.float32)

        model_mean_coarse = None
        if mean_padxy is not None:
            model_mean_coarse = np.array([
                mapper.coarsen(
                    torch.from_numpy(mean_padxy[m, ch_idx].copy()).float()
                ).cpu().numpy()
                for m in range(12)
            ]).astype(np.float32)  # (12, n_cells)

        stats[key] = {
            'inv_std': inv_std_coarse,
            'model_mean_coarse': model_mean_coarse,
            'ch_idx': ch_idx,
            'std_hpx': std_hpx.copy(),
        }
        logger.info(
            f"  {name} (ch{ch_idx}) -> {key}: "
            f"inv_std range [{inv_std_coarse.min():.4f}, {inv_std_coarse.max():.4f}], "
            f"median={med_inv:.4f}, cap={cap:.4f}"
        )

    return stats, std_padxy


# =============================================================================
# Real observation loaders  (return normalized values + masks on coarse grid)
# =============================================================================

def _normalize_coarse(anom_coarse, inv_std_coarse, mask):
    """Normalize anomaly observations: anom * inv_std * SIGMA_DATA, clamped.

    Anomalies should be relative to the model's monthly mean (when
    --use_model_clim is on) or to the obs product's own 1993-2020
    climatology (default).
    """
    raw = anom_coarse * inv_std_coarse * SIGMA_DATA
    return np.where(mask, np.clip(raw, -OBS_CLIP, OBS_CLIP), np.nan
                    ).astype(np.float32)


def load_hadsst_obs(path, mapper, start_dt, n_months, norm_entry,
                    use_model_clim=False, pixel_normalize=False):
    """Load HadSST anomalies, re-baseline, normalize per-cell."""
    inv_std = norm_entry['inv_std']
    model_mean_coarse = norm_entry.get('model_mean_coarse')
    std_hpx_ch = norm_entry.get('std_hpx')
    logger.info(f"  Loading HadSST observations: {path}")
    ds = xr.open_dataset(path)
    t0 = _find_time_index(ds, start_dt)
    lat, lon = ds['latitude'].values, ds['longitude'].values

    tos_clim_sel = ds['tos'].sel(time=slice('1993', '2020'))
    rebase = tos_clim_sel.groupby('time.month').mean('time')

    obs_clim_coarse = None
    if use_model_clim and model_mean_coarse is not None:
        logger.info("    Computing obs 1993-2020 clim on coarse grid for bias correction ...")
        obs_clim_coarse = np.zeros((12, mapper.n_cells), dtype=np.float32)
        for m_idx in range(12):
            field = rebase.sel(month=m_idx + 1).values
            obs_clim_coarse[m_idx] = map_obs_to_coarse_cells(field, lat, lon, mapper)

    native_to_hpx = None
    if pixel_normalize:
        assert std_hpx_ch is not None, "std_hpx required for pixel_normalize"
        native_to_hpx = _build_native_to_hpx_map(lat, lon)

    values_list, masks_list = [], []
    for i in range(n_months):
        ti = min(t0 + i, len(ds.time) - 1)
        time_val = pd.Timestamp(ds.time.values[ti])
        month = time_val.month

        field = ds['tos'].isel(time=ti).values
        rebased = field - rebase.sel(month=month).values

        if pixel_normalize:
            norm_field = _pixel_normalize_field(rebased, std_hpx_ch, native_to_hpx)
            coarse = map_obs_to_coarse_cells(norm_field, lat, lon, mapper)
            mask = np.isfinite(coarse)
            values_list.append(np.where(mask, coarse, np.nan).astype(np.float32))
        else:
            coarse = map_obs_to_coarse_cells(rebased, lat, lon, mapper)
            mask = np.isfinite(coarse)
            if obs_clim_coarse is not None:
                correction = model_mean_coarse[month - 1] - obs_clim_coarse[month - 1]
                coarse = np.where(mask, coarse - correction, np.nan)
            values_list.append(_normalize_coarse(coarse, inv_std, mask))
        masks_list.append(mask)

    ds.close()
    pct = np.mean([m.sum() / mapper.n_cells * 100 for m in masks_list])
    mode = "pixel-norm" if pixel_normalize else (
        "model-clim" if obs_clim_coarse is not None else "standard")
    logger.info(f"    avg {pct:.1f}% coarse cells observed / month ({mode})")
    return values_list, masks_list


def load_duacs_obs(path, mapper, start_dt, n_months, norm_entry,
                   use_model_clim=False, pixel_normalize=False,
                   remove_global_mean=True, duacs_var="sla",
                   exclude_med=True):
    """Load DUACS altimetry SSH (adt/sla), re-baseline to its own 1993-2020
    monthly climatology -> anomaly (m), optionally remove the per-timestep
    global ocean-area mean (so global mean sea-level rise / steric drift, which
    the detrended model does not represent, is not forced onto the prior), then
    coarsen onto `mapper`'s grid and normalize with the model zos per-cell stats.

    The coarse grid (mapper) is intentionally chosen coarse so altimetric
    mesoscale eddies are spatially averaged out and do not pull the
    reconstruction far from the diffusion prior.
    """
    inv_std = norm_entry['inv_std']
    model_mean_coarse = norm_entry.get('model_mean_coarse')
    std_hpx_ch = norm_entry.get('std_hpx')
    logger.info(f"  Loading DUACS SSH observations: {path}")
    ds = xr.open_dataset(path)
    if duacs_var in ds.data_vars:
        var = duacs_var
    else:
        var = next(v for v in ('sla', 'adt', 'ssh', 'zos') if v in ds.data_vars)
        logger.warning(f"    requested duacs_var='{duacs_var}' not found; using '{var}'")
    logger.info(f"    using variable '{var}', coarsen grid "
                f"{mapper.resolution}°, remove_global_mean={remove_global_mean}")
    t0 = _find_time_index(ds, start_dt)
    lat, lon = ds['lat'].values, ds['lon'].values
    lon180 = np.where(lon > 180, lon - 360, lon)

    med_native = None
    if exclude_med:
        med_native = ((lat[:, None] >= 30.0) & (lat[:, None] <= 47.0) &
                      (lon180[None, :] >= -6.0) & (lon180[None, :] <= 42.0))
        logger.info(f"    excluding Mediterranean/Black Sea: "
                    f"{int(med_native.sum())} native cells masked")

    ssh_clim_sel = ds[var].sel(time=slice('1993', '2020'))
    rebase = ssh_clim_sel.groupby('time.month').mean('time')

    cos_w = np.cos(np.deg2rad(lat))[:, None]  # (n_lat, 1) area weight

    obs_clim_coarse = None
    if use_model_clim and model_mean_coarse is not None:
        obs_clim_coarse = np.zeros((12, mapper.n_cells), dtype=np.float32)
        for m_idx in range(12):
            field = rebase.sel(month=m_idx + 1).values
            obs_clim_coarse[m_idx] = map_obs_to_coarse_cells(field, lat, lon, mapper)

    native_to_hpx = None
    if pixel_normalize:
        assert std_hpx_ch is not None, "std_hpx required for pixel_normalize"
        native_to_hpx = _build_native_to_hpx_map(lat, lon)

    values_list, masks_list = [], []
    for i in range(n_months):
        ti = min(t0 + i, len(ds.time) - 1)
        month = pd.Timestamp(ds.time.values[ti]).month
        field = ds[var].isel(time=ti).values
        rebased = field - rebase.sel(month=month).values

        if med_native is not None:
            rebased = np.where(med_native, np.nan, rebased)

        if remove_global_mean:
            ok = np.isfinite(rebased)
            if ok.any():
                w = np.broadcast_to(cos_w, rebased.shape)
                gmean = np.average(rebased[ok], weights=w[ok])
                rebased = rebased - gmean

        if pixel_normalize:
            norm_field = _pixel_normalize_field(rebased, std_hpx_ch, native_to_hpx)
            coarse = map_obs_to_coarse_cells(norm_field, lat, lon, mapper)
            mask = np.isfinite(coarse)
            values_list.append(np.where(mask, coarse, np.nan).astype(np.float32))
        else:
            coarse = map_obs_to_coarse_cells(rebased, lat, lon, mapper)
            mask = np.isfinite(coarse)
            if obs_clim_coarse is not None:
                correction = model_mean_coarse[month - 1] - obs_clim_coarse[month - 1]
                coarse = np.where(mask, coarse - correction, np.nan)
            values_list.append(_normalize_coarse(coarse, inv_std, mask))
        masks_list.append(mask)

    ds.close()
    pct = np.mean([m.sum() / mapper.n_cells * 100 for m in masks_list])
    mode = "pixel-norm" if pixel_normalize else (
        "model-clim" if obs_clim_coarse is not None else "standard")
    logger.info(f"    avg {pct:.1f}% coarse cells observed / month ({mode})")
    return values_list, masks_list


def load_crutem_obs(path, mapper, start_dt, n_months, norm_entry,
                    use_model_clim=False, pixel_normalize=False):
    """Load CRUTEM anomalies, re-baseline, normalize per-cell."""
    inv_std = norm_entry['inv_std']
    model_mean_coarse = norm_entry.get('model_mean_coarse')
    std_hpx_ch = norm_entry.get('std_hpx')
    logger.info(f"  Loading CRUTEM observations: {path}")
    ds = xr.open_dataset(path)
    t0 = _find_time_index(ds, start_dt)
    lat, lon = ds['latitude'].values, ds['longitude'].values

    tas_clim_sel = ds['tas'].sel(time=slice('1993', '2020'))
    rebase = tas_clim_sel.groupby('time.month').mean('time')

    obs_clim_coarse = None
    if use_model_clim and model_mean_coarse is not None:
        logger.info("    Computing obs 1993-2020 clim on coarse grid for bias correction ...")
        obs_clim_coarse = np.zeros((12, mapper.n_cells), dtype=np.float32)
        for m_idx in range(12):
            field = rebase.sel(month=m_idx + 1).values
            obs_clim_coarse[m_idx] = map_obs_to_coarse_cells(field, lat, lon, mapper)

    native_to_hpx = None
    if pixel_normalize:
        assert std_hpx_ch is not None, "std_hpx required for pixel_normalize"
        native_to_hpx = _build_native_to_hpx_map(lat, lon)

    values_list, masks_list = [], []
    for i in range(n_months):
        ti = min(t0 + i, len(ds.time) - 1)
        time_val = pd.Timestamp(ds.time.values[ti])
        month = time_val.month

        field = ds['tas'].isel(time=ti).values
        rebased = field - rebase.sel(month=month).values

        if pixel_normalize:
            norm_field = _pixel_normalize_field(rebased, std_hpx_ch, native_to_hpx)
            coarse = map_obs_to_coarse_cells(norm_field, lat, lon, mapper)
            mask = np.isfinite(coarse)
            values_list.append(np.where(mask, coarse, np.nan).astype(np.float32))
        else:
            coarse = map_obs_to_coarse_cells(rebased, lat, lon, mapper)
            mask = np.isfinite(coarse)
            if obs_clim_coarse is not None:
                correction = model_mean_coarse[month - 1] - obs_clim_coarse[month - 1]
                coarse = np.where(mask, coarse - correction, np.nan)
            values_list.append(_normalize_coarse(coarse, inv_std, mask))
        masks_list.append(mask)

    ds.close()
    pct = np.mean([m.sum() / mapper.n_cells * 100 for m in masks_list])
    mode = "pixel-norm" if pixel_normalize else (
        "model-clim" if obs_clim_coarse is not None else "standard")
    logger.info(f"    avg {pct:.1f}% coarse cells observed / month ({mode})")
    return values_list, masks_list


def load_icoads_obs(path, mapper, start_dt, n_months, norm_entry,
                    use_model_clim=False, pixel_normalize=False):
    """Load ICOADS SLP (Pa, absolute) -> anomalies -> normalize."""
    inv_std = norm_entry['inv_std']
    model_mean_coarse = norm_entry.get('model_mean_coarse')
    std_hpx_ch = norm_entry.get('std_hpx')
    logger.info(f"  Loading ICOADS observations: {path}")
    ds = xr.open_dataset(path)
    t0 = _find_time_index(ds, start_dt)
    lat, lon = ds['lat'].values, ds['lon'].values

    logger.info("    Computing ICOADS 1993-2020 monthly climatology ...")
    psl_clim = ds['psl'].sel(time=slice('1993', '2020')).groupby('time.month').mean('time')

    obs_clim_coarse = None
    if use_model_clim and model_mean_coarse is not None:
        logger.info("    Computing obs clim on coarse grid for bias correction ...")
        obs_clim_coarse = np.zeros((12, mapper.n_cells), dtype=np.float32)
        for m_idx in range(12):
            field = psl_clim.sel(month=m_idx + 1).values
            obs_clim_coarse[m_idx] = map_obs_to_coarse_cells(field, lat, lon, mapper)

    native_to_hpx = None
    if pixel_normalize:
        assert std_hpx_ch is not None, "std_hpx required for pixel_normalize"
        native_to_hpx = _build_native_to_hpx_map(lat, lon)

    values_list, masks_list = [], []
    for i in range(n_months):
        ti = min(t0 + i, len(ds.time) - 1)
        time_val = pd.Timestamp(ds.time.values[ti])
        month = time_val.month

        field = ds['psl'].isel(time=ti).values
        anom = field - psl_clim.sel(month=month).values

        if pixel_normalize:
            norm_field = _pixel_normalize_field(anom, std_hpx_ch, native_to_hpx)
            coarse = map_obs_to_coarse_cells(norm_field, lat, lon, mapper)
            mask = np.isfinite(coarse)
            values_list.append(np.where(mask, coarse, np.nan).astype(np.float32))
        else:
            coarse = map_obs_to_coarse_cells(anom, lat, lon, mapper)
            mask = np.isfinite(coarse)
            if obs_clim_coarse is not None:
                correction = model_mean_coarse[month - 1] - obs_clim_coarse[month - 1]
                coarse = np.where(mask, coarse - correction, np.nan)
            values_list.append(_normalize_coarse(coarse, inv_std, mask))
        masks_list.append(mask)

    ds.close()
    pct = np.mean([m.sum() / mapper.n_cells * 100 for m in masks_list])
    mode = "pixel-norm" if pixel_normalize else (
        "model-clim" if obs_clim_coarse is not None else "standard")
    logger.info(f"    avg {pct:.1f}% coarse cells observed / month ({mode})")
    return values_list, masks_list


def load_en4_zarr_obs(path, mapper, start_dt, n_months, model_depths,
                      norm_stats, min_profiles=3, use_model_clim=False,
                      pixel_normalize=False):
    """Load EN4 zarr T/S profiles -> anomalies -> normalize."""
    logger.info(f"  Loading EN4 zarr observations: {path}")
    ds = xr.open_zarr(path)
    t0 = _find_time_index(ds, start_dt)
    lat, lon = ds['lat'].values, ds['lon'].values
    en4_depths = ds['depth'].values

    depth_map = [int(np.argmin(np.abs(en4_depths - md))) for md in model_depths]
    logger.info(f"    Depth mapping: model -> EN4 zarr")
    for i, (md, ed) in enumerate(zip(model_depths, [en4_depths[j] for j in depth_map])):
        logger.info(f"      [{i}] {md:.0f}m -> {ed:.1f}m")

    logger.info("    Computing EN4 zarr 1993-2020 monthly climatology ...")
    ds_clim = ds.sel(time=slice('1993', '2020'))
    potm_clim = ds_clim['potm_mean'].groupby('time.month').mean('time')
    psal_clim = ds_clim['psal_mean'].groupby('time.month').mean('time')

    native_to_hpx = None
    if pixel_normalize:
        native_to_hpx = _build_native_to_hpx_map(lat, lon)

    thetao_per_depth, so_per_depth = [], []

    for d_model_idx, d_en4_idx in enumerate(depth_map):
        inv_t = norm_stats[f'thetao_{d_model_idx}']['inv_std']
        inv_s = norm_stats[f'so_{d_model_idx}']['inv_std']
        std_hpx_t = norm_stats[f'thetao_{d_model_idx}'].get('std_hpx')
        std_hpx_s = norm_stats[f'so_{d_model_idx}'].get('std_hpx')
        model_mean_t = norm_stats[f'thetao_{d_model_idx}'].get('model_mean_coarse')
        model_mean_s = norm_stats[f'so_{d_model_idx}'].get('model_mean_coarse')

        obs_clim_t_coarse = None
        obs_clim_s_coarse = None
        if use_model_clim and not pixel_normalize and model_mean_t is not None:
            obs_clim_t_coarse = np.zeros((12, mapper.n_cells), dtype=np.float32)
            obs_clim_s_coarse = np.zeros((12, mapper.n_cells), dtype=np.float32)
            for m_idx in range(12):
                ft = potm_clim.sel(month=m_idx + 1).isel(depth=d_en4_idx).values
                obs_clim_t_coarse[m_idx] = map_obs_to_coarse_cells(ft, lat, lon, mapper)
                fs = psal_clim.sel(month=m_idx + 1).isel(depth=d_en4_idx).values
                obs_clim_s_coarse[m_idx] = map_obs_to_coarse_cells(fs, lat, lon, mapper)

        t_vals_list, t_masks_list = [], []
        s_vals_list, s_masks_list = [], []

        for i in range(n_months):
            ti = min(t0 + i, len(ds.time) - 1)
            time_val = pd.Timestamp(ds.time.values[ti])
            month = time_val.month

            # --- temperature ---
            potm = ds['potm_mean'].isel(time=ti, depth=d_en4_idx).values
            n_prof_t = ds['potm_n_profiles'].isel(time=ti, depth=d_en4_idx).values
            mask_t_native = np.isfinite(potm) & (n_prof_t >= min_profiles)

            clim_t = potm_clim.sel(month=month).isel(depth=d_en4_idx).values
            anom_t = np.where(mask_t_native, potm - clim_t, np.nan)

            if pixel_normalize:
                norm_t = _pixel_normalize_field(anom_t, std_hpx_t, native_to_hpx)
                coarse_t = map_obs_to_coarse_cells(norm_t, lat, lon, mapper)
                cmask_t = np.isfinite(coarse_t)
                t_vals_list.append(np.where(cmask_t, coarse_t, np.nan).astype(np.float32))
            else:
                coarse_t = map_obs_to_coarse_cells(anom_t, lat, lon, mapper)
                cmask_t = np.isfinite(coarse_t)
                if obs_clim_t_coarse is not None:
                    corr = model_mean_t[month - 1] - obs_clim_t_coarse[month - 1]
                    coarse_t = np.where(cmask_t, coarse_t - corr, np.nan)
                t_vals_list.append(_normalize_coarse(coarse_t, inv_t, cmask_t))
            t_masks_list.append(cmask_t)

            # --- salinity ---
            psal = ds['psal_mean'].isel(time=ti, depth=d_en4_idx).values
            n_prof_s = ds['psal_n_profiles'].isel(time=ti, depth=d_en4_idx).values
            mask_s_native = np.isfinite(psal) & (n_prof_s >= min_profiles)

            clim_s = psal_clim.sel(month=month).isel(depth=d_en4_idx).values
            anom_s = np.where(mask_s_native, psal - clim_s, np.nan)

            if pixel_normalize:
                norm_s = _pixel_normalize_field(anom_s, std_hpx_s, native_to_hpx)
                coarse_s = map_obs_to_coarse_cells(norm_s, lat, lon, mapper)
                cmask_s = np.isfinite(coarse_s)
                s_vals_list.append(np.where(cmask_s, coarse_s, np.nan).astype(np.float32))
            else:
                coarse_s = map_obs_to_coarse_cells(anom_s, lat, lon, mapper)
                cmask_s = np.isfinite(coarse_s)
                if obs_clim_s_coarse is not None:
                    corr = model_mean_s[month - 1] - obs_clim_s_coarse[month - 1]
                    coarse_s = np.where(cmask_s, coarse_s - corr, np.nan)
                s_vals_list.append(_normalize_coarse(coarse_s, inv_s, cmask_s))
            s_masks_list.append(cmask_s)

        thetao_per_depth.append((t_vals_list, t_masks_list))
        so_per_depth.append((s_vals_list, s_masks_list))

        pct_t = np.mean([m.sum() / mapper.n_cells * 100 for m in t_masks_list])
        pct_s = np.mean([m.sum() / mapper.n_cells * 100 for m in s_masks_list])
        mode = "pixel-norm" if pixel_normalize else (
            "bias-corr" if obs_clim_t_coarse is not None else "standard")
        logger.info(f"    depth {d_model_idx} ({en4_depths[d_en4_idx]:.0f}m): "
                     f"T {pct_t:.1f}%, S {pct_s:.1f}% observed ({mode})")

    ds.close()
    return thetao_per_depth, so_per_depth


# =============================================================================
# Real-observation Multi-Variable Observation Operator
# =============================================================================

class RealObsMultiVarObs:
    """
    Multi-variable observation operator using real observational data.

    Same interface as RealisticMultiVarObs from v4: builds y, A, mapper_groups,
    channel_obs — drop-in replacement for PatchedRealisticScore.
    """

    def __init__(
        self,
        n_depths: int,
        C: int,
        L: int,
        H: int,
        hadsst_path: str,
        crutem_path: str,
        icoads_path: str,
        en4_zarr_path: str,
        norm_stats: dict,
        model_depths: np.ndarray,
        start_dt: dt.datetime,
        noise_std: float = 0.15,
        device: str = 'cuda',
        min_profiles: int = 3,
        use_model_clim: bool = False,
        std_padxy: np.ndarray = None,
        matched_coarsening: bool = False,
        pixel_normalize: bool = False,
        per_cell_sigma_obs: dict = None,
        duacs_path: str = None,
        use_ssh_obs: bool = False,
        ssh_only: bool = False,
        ssh_coarsen_deg: float = 5.0,
        ssh_remove_global_mean: bool = True,
        duacs_var: str = "sla",
        ssh_exclude_med: bool = True,
    ):
        self.C, self.L, self.H = C, L, H
        self.device = device
        self.matched_coarsening = matched_coarsening
        self.use_ssh_obs = use_ssh_obs
        self.ssh_only = ssh_only

        # ---- channel indices ---------------------------------------------------
        self.thetao_chs = list(range(0, n_depths))
        self.so_chs     = list(range(n_depths, 2 * n_depths))
        base_2d = 2 * n_depths
        self.tos_ch  = base_2d + DEFAULT_2D_VARS.index("tos")
        self.psl_ch  = base_2d + DEFAULT_2D_VARS.index("psl")
        self.tas_ch  = base_2d + DEFAULT_2D_VARS.index("tas")
        sos_ch       = base_2d + DEFAULT_2D_VARS.index("sos")
        zos_ch       = base_2d + DEFAULT_2D_VARS.index("zos")
        siconc_ch    = base_2d + DEFAULT_2D_VARS.index("siconc")
        self.zos_ch  = zos_ch
        if use_ssh_obs:
            self.unobserved = {sos_ch, siconc_ch}
        else:
            self.unobserved = {sos_ch, zos_ch, siconc_ch}

        # ---- coarse-grid mappers -----------------------------------------------
        logger.info("Creating coarse-grid mappers ...")
        self.mapper_2deg = CoarseGridMapper(2.0, device)
        self.mapper_5deg = CoarseGridMapper(5.0, device)
        if use_ssh_obs:
            if abs(ssh_coarsen_deg - 5.0) < 1e-6:
                self.mapper_ssh = self.mapper_5deg
            elif abs(ssh_coarsen_deg - 2.0) < 1e-6:
                self.mapper_ssh = self.mapper_2deg
            else:
                self.mapper_ssh = CoarseGridMapper(ssh_coarsen_deg, device)

        # ---- load real observations (per-cell normalized) -----------------------
        mode_str = "pixel-norm" if pixel_normalize else (
            "model-clim" if use_model_clim else "standard")
        logger.info(f"Loading real observations (start={start_dt:%Y-%m}, L={L})"
                    f" — mode: {mode_str} ...")

        # ---- per-channel obs: ch -> (mapper, masks, values) --------------------
        self.channel_obs: dict = {}

        if not ssh_only:
            hadsst_vals, hadsst_masks = load_hadsst_obs(
                hadsst_path, self.mapper_5deg, start_dt, L,
                norm_stats['tos'], use_model_clim=use_model_clim,
                pixel_normalize=pixel_normalize)

            crutem_vals, crutem_masks = load_crutem_obs(
                crutem_path, self.mapper_5deg, start_dt, L,
                norm_stats['tas'], use_model_clim=use_model_clim,
                pixel_normalize=pixel_normalize)

            icoads_vals, icoads_masks = load_icoads_obs(
                icoads_path, self.mapper_2deg, start_dt, L,
                norm_stats['psl'], use_model_clim=use_model_clim,
                pixel_normalize=pixel_normalize)

            en4_thetao, en4_so = load_en4_zarr_obs(
                en4_zarr_path, self.mapper_5deg, start_dt, L,
                model_depths, norm_stats, min_profiles=min_profiles,
                use_model_clim=use_model_clim,
                pixel_normalize=pixel_normalize)

            for i, ch in enumerate(self.thetao_chs):
                if i < len(en4_thetao):
                    vals_i, masks_i = en4_thetao[i]
                    self.channel_obs[ch] = (self.mapper_5deg, masks_i, vals_i)
            for i, ch in enumerate(self.so_chs):
                if i < len(en4_so):
                    vals_i, masks_i = en4_so[i]
                    self.channel_obs[ch] = (self.mapper_5deg, masks_i, vals_i)
            self.channel_obs[self.tos_ch] = (self.mapper_5deg, hadsst_masks, hadsst_vals)
            self.channel_obs[self.psl_ch] = (self.mapper_2deg, icoads_masks, icoads_vals)
            self.channel_obs[self.tas_ch] = (self.mapper_5deg, crutem_masks, crutem_vals)

        if use_ssh_obs:
            assert duacs_path is not None, "duacs_path required for use_ssh_obs"
            assert 'zos' in norm_stats, "norm_stats['zos'] required (add 'zos' mapper)"
            zos_vals, zos_masks = load_duacs_obs(
                duacs_path, self.mapper_ssh, start_dt, L,
                norm_stats['zos'], use_model_clim=use_model_clim,
                pixel_normalize=pixel_normalize,
                remove_global_mean=ssh_remove_global_mean,
                duacs_var=duacs_var, exclude_med=ssh_exclude_med)
            self.channel_obs[self.zos_ch] = (self.mapper_ssh, zos_masks, zos_vals)

        # ---- matched coarsening: per-pixel std + per-cell inv_std ---------------
        if matched_coarsening:
            assert std_padxy is not None, \
                "std_padxy required when matched_coarsening=True"
            self.std_hpx = torch.from_numpy(
                np.maximum(std_padxy, 1e-6).astype(np.float32)
            ).to(device)                              # (C, H)
            self.ch_inv_std: dict = {}
            for key, entry in norm_stats.items():
                ch_idx = entry['ch_idx']
                self.ch_inv_std[ch_idx] = torch.from_numpy(
                    entry['inv_std'].copy()
                ).float().to(device)
            logger.info(f"  Matched coarsening ON: std_hpx {self.std_hpx.shape}, "
                        f"{len(self.ch_inv_std)} channel inv_std loaded")

        # ---- build y vector and mapper_groups -----------------------------------
        logger.info("Building observation vector ...")
        mapper_to_items: dict = {}

        for ch, (mapper, masks, values) in self.channel_obs.items():
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

        y_parts = []
        for mapper, items in self.mapper_groups:
            for ch, t, obs_idx in items:
                _, masks_ch, values_ch = self.channel_obs[ch]
                vals_t = torch.as_tensor(
                    values_ch[t], device=device, dtype=torch.float32)
                y_parts.append(vals_t[obs_idx])

        self.y_clean = torch.cat(y_parts).to(device) if y_parts else \
            torch.empty(0, device=device, dtype=torch.float32)
        self.noise_std = noise_std
        self.y = self.y_clean.clone()
        if noise_std > 0 and len(self.y) > 0:
            self.y = self.y + torch.randn_like(self.y) * noise_std

        # ---- per-cell sigma_obs (Option C) ----------------------------------------
        self.sigma_obs_vec = None
        if per_cell_sigma_obs is not None:
            ch_to_key = {}
            for i in self.thetao_chs:
                ch_to_key[i] = f"thetao_{i}"
            for i in self.so_chs:
                ch_to_key[i] = f"so_{i - len(self.thetao_chs)}"
            ch_to_key[self.tos_ch] = "tos"
            ch_to_key[self.psl_ch] = "psl"
            ch_to_key[self.tas_ch] = "tas"
            ch_to_key[self.zos_ch] = "zos"

            sigma_parts = []
            for mapper, items in self.mapper_groups:
                for ch, t, obs_idx in items:
                    key = ch_to_key.get(ch)
                    if key and key in per_cell_sigma_obs:
                        sigma_arr = per_cell_sigma_obs[key]['sigma_obs']
                        sigma_t = torch.as_tensor(
                            sigma_arr, device=device, dtype=torch.float32)
                        sigma_parts.append(sigma_t[obs_idx.cpu()])
                    else:
                        sigma_parts.append(torch.full(
                            (len(obs_idx),), noise_std, device=device))

            self.sigma_obs_vec = torch.cat(sigma_parts).to(device) if sigma_parts \
                else None
            if self.sigma_obs_vec is not None:
                logger.info(f"  Per-cell σ_obs loaded: "
                            f"[{self.sigma_obs_vec.min():.4f}, "
                            f"{self.sigma_obs_vec.median():.4f}, "
                            f"{self.sigma_obs_vec.max():.4f}]")

        self.n_obs = len(self.y)
        logger.info(f"RealObsMultiVarObs: {len(self.channel_obs)} observed channels, "
                     f"{self.n_obs:,} total observations")
        logger.info(f"  Unobserved channels: {sorted(self.unobserved)}")

        # ---- per-source normalization validation ---------------------------------
        self._validate_obs()

    def _validate_obs(self):
        """Log per-channel observation statistics to catch normalization issues."""
        ch_names_map = {}
        for i in self.thetao_chs:
            ch_names_map[i] = f"thetao_{i}"
        for i in self.so_chs:
            ch_names_map[i] = f"so_{i - len(self.thetao_chs)}"
        ch_names_map[self.tos_ch] = "tos"
        ch_names_map[self.psl_ch] = "psl"
        ch_names_map[self.tas_ch] = "tas"
        ch_names_map[self.zos_ch] = "zos"

        logger.info("  Observation normalization validation (per channel):")
        any_bad = False
        for ch, (mapper, masks, values) in self.channel_obs.items():
            all_vals = np.concatenate([v[m] for v, m in zip(values, masks)
                                       if m.any()])
            if len(all_vals) == 0:
                continue
            mn, sd = all_vals.mean(), all_vals.std()
            lo, hi = all_vals.min(), all_vals.max()
            n_clip = ((np.abs(all_vals) >= OBS_CLIP - 0.01).sum())
            name = ch_names_map.get(ch, f"ch{ch}")
            flag = " *** CHECK" if abs(mn) > 1.0 or sd > 2.0 else ""
            logger.info(f"    {name:>12s}: mean={mn:+.4f} std={sd:.4f} "
                        f"[{lo:.3f}, {hi:.3f}] clipped={n_clip}{flag}")
            if flag:
                any_bad = True
        if any_bad:
            logger.warning("  Some channels have large normalised values — "
                           "check data/climatology alignment")
        y_np = self.y_clean.cpu().numpy()
        logger.info(f"  Overall y_clean: mean={y_np.mean():.4f} std={y_np.std():.4f} "
                    f"[{y_np.min():.3f}, {y_np.max():.3f}]")

    def perturb_observations(self, seed: int):
        g = torch.Generator(device=self.y_clean.device).manual_seed(seed)
        self.y = self.y_clean + torch.randn(
            self.y_clean.shape, device=self.y_clean.device,
            dtype=self.y_clean.dtype, generator=g,
        ) * self.noise_std

    def _rebuild_y(self, device: str = 'cuda'):
        """Rebuild mapper_groups and y_clean/y after masks have been modified."""
        L = len(next(iter(self.channel_obs.values()))[1])
        mapper_to_items = {}
        for ch, (mapper, masks, values) in self.channel_obs.items():
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
        y_parts = []
        for mapper, items in self.mapper_groups:
            for ch, t, obs_idx in items:
                _, _, values_ch = self.channel_obs[ch]
                vals_t = torch.as_tensor(
                    values_ch[t], device=device, dtype=torch.float32)
                y_parts.append(vals_t[obs_idx])
        self.y_clean = torch.cat(y_parts).to(device) if y_parts else \
            torch.empty(0, device=device, dtype=torch.float32)
        self.y = self.y_clean.clone()
        if self.noise_std > 0 and len(self.y) > 0:
            self.y = self.y + torch.randn_like(self.y) * self.noise_std
        self.sigma_obs_vec = None
        self.n_obs = len(self.y)

    def A(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable observation operator.  x: (1, C, L, H) -> (n_obs,)"""
        parts = []
        if self.matched_coarsening:
            for mapper, items in self.mapper_groups:
                for ch, t, obs_idx in items:
                    raw_anom = (x[0, ch, t, :] * self.std_hpx[ch]
                                / SIGMA_DATA)
                    coarsened = (mapper.coarsen(raw_anom)
                                 * self.ch_inv_std[ch] * SIGMA_DATA)
                    parts.append(coarsened[obs_idx])
        else:
            for mapper, items in self.mapper_groups:
                fields = torch.stack([x[0, ch, t, :] for ch, t, _ in items])
                coarsened = mapper.coarsen_batch(fields)
                for i, (_, _, obs_idx) in enumerate(items):
                    parts.append(coarsened[i][obs_idx])
        return torch.cat(parts)


# =============================================================================
# Observation field builder (for saving to results.pt)
# =============================================================================

def build_obs_field(obs, C, L, H):
    """Build (C, L, H) observation field in PAD_XY order for saving.

    Each HEALPix pixel gets the coarse-cell value from real observations
    when observed, NaN otherwise.
    """
    obs_field = np.full((C, L, H), np.nan, dtype=np.float32)

    for ch, (mapper, masks, values) in obs.channel_obs.items():
        h2c = mapper.hpx_to_coarse.cpu().numpy()
        for t in range(L):
            if not masks[t].any():
                continue
            v = values[t]
            o_hpx = v[h2c]
            o_hpx[~masks[t][h2c]] = np.nan
            obs_field[ch, t] = o_hpx

    return torch.from_numpy(obs_field)




# =============================================================================
# Visualization  (no GT — just Observations + Generated)
# =============================================================================

def _prepare_ring_data(gen, ch_names, land_mask_nest):
    gen_np = gen[0].cpu().numpy()

    grid = Grid(level=6, pixel_order=PixelOrder.RING)
    lon = np.where(np.array(grid.lon) > 180,
                   np.array(grid.lon) - 360, np.array(grid.lon))
    lat = np.array(grid.lat)

    gen_ring = earth2grid.healpix.reorder(
        torch.from_numpy(gen_np), HEALPIX_PAD_XY, PixelOrder.RING).numpy()

    lm_ring = None
    if land_mask_nest is not None:
        lm_ring = earth2grid.healpix.reorder(
            torch.as_tensor(land_mask_nest, dtype=torch.float32),
            PixelOrder.NEST, PixelOrder.RING).numpy() > 0.5

    return gen_np, gen_ring, lm_ring, lon, lat


def _build_obs_ring(channels, obs, L, H):
    """Map real obs (coarse-cell values) back to RING for visualization."""
    obs_ring: dict = {}
    for ch in channels:
        if ch not in obs.channel_obs:
            continue
        mapper, masks, values = obs.channel_obs[ch]
        h2c = mapper.hpx_to_coarse.cpu().numpy()
        for t in range(L):
            if not masks[t].any():
                obs_ring[(ch, t)] = np.full(H, np.nan)
                continue
            v = values[t]
            o_hpx = v[h2c]
            o_hpx[~masks[t][h2c]] = np.nan
            obs_ring[(ch, t)] = earth2grid.healpix.reorder(
                torch.from_numpy(o_hpx.astype(np.float32)),
                HEALPIX_PAD_XY, PixelOrder.RING).numpy()
    return obs_ring


def _make_one_gif(channels, group_label, gen_ring, lm_ring,
                  lon, lat, obs, obs_ring, ch_names,
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
        vals = gen_ring[ch].ravel()
        if lm_ring is not None and is_ocean_var(nm):
            ocean_tile = ~np.tile(lm_ring[ch], L)
            if ocean_tile.shape[0] == vals.shape[0]:
                vals = vals[ocean_tile]
        v = vals[np.isfinite(vals)]
        vr[ch] = (float(np.percentile(v, 2)),
                  float(np.percentile(v, 98))) if len(v) > 0 else (-1, 1)

    from matplotlib import cm
    from matplotlib.colors import Normalize as mplNorm

    frames_imgs = []
    for t in tqdm(range(L), desc=f"GIF [{group_label}]"):
        kw = {'subplot_kw': {'projection': ccrs.Robinson()}} if has_carto else {}
        fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 7), **kw)
        if n_cols == 1:
            axes = axes[:, np.newaxis]

        for ci, ch in enumerate(channels):
            nm = ch_names[ch] if ch < len(ch_names) else f"ch{ch}"
            vmin, vmax = vr[ch]
            is_oc = is_ocean_var(nm)
            scatter_kw = dict(s=0.3, cmap="coolwarm", vmin=vmin, vmax=vmax)

            # Row 0: Observations
            ax = axes[0, ci]
            d = obs_ring.get((ch, t), np.full(H, np.nan))
            vi = np.isfinite(d)
            if has_carto:
                ax.set_global()
                ax.coastlines(linewidth=0.3)
                if vi.any():
                    ax.scatter(lon[vi], lat[vi], c=d[vi],
                               transform=ccrs.PlateCarree(), **scatter_kw)
            else:
                ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
                if vi.any():
                    ax.scatter(lon[vi], lat[vi], c=d[vi], **scatter_kw)
            n_obs_str = ""
            if ch in obs.channel_obs:
                _, ms, _ = obs.channel_obs[ch]
                n_obs_str = f" ({ms[t].sum()})"
            ax.set_title(f"Obs: {nm}{n_obs_str}", fontsize=6)

            # Row 1: Generated
            ax = axes[1, ci]
            d = gen_ring[ch, t].copy()
            if lm_ring is not None and is_oc:
                d[lm_ring[ch]] = np.nan
            vi = np.isfinite(d)
            sc = None
            if has_carto:
                ax.set_global()
                ax.coastlines(linewidth=0.3)
                if vi.any():
                    sc = ax.scatter(lon[vi], lat[vi], c=d[vi],
                                    transform=ccrs.PlateCarree(), **scatter_kw)
            else:
                ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
                if vi.any():
                    sc = ax.scatter(lon[vi], lat[vi], c=d[vi], **scatter_kw)
            ax.set_title(f"Gen: {nm}", fontsize=6)

            # Colorbar below the generated row
            norm = mplNorm(vmin=vmin, vmax=vmax)
            sm = cm.ScalarMappable(cmap="coolwarm", norm=norm)
            sm.set_array([])
            cb = fig.colorbar(sm, ax=[axes[0, ci], axes[1, ci]],
                              orientation='horizontal', fraction=0.06,
                              pad=0.04, aspect=25)
            cb.ax.tick_params(labelsize=5)

        fig.suptitle(f"{group_label}  t={t+1}/{L}", fontsize=10, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        frames_imgs.append(Image.open(buf).copy())
        buf.close()

    safe_label = group_label.lower().replace(" ", "_").replace("/", "_")
    gif_path = os.path.join(out_dir, f"real_obs_{safe_label}.gif")
    frames_imgs[0].save(gif_path, save_all=True,
                        append_images=frames_imgs[1:],
                        duration=int(1000 / fps), loop=0)
    logger.info(f"Saved {gif_path}  ({n_cols} channels)")


def create_all_gifs(gen, obs, ch_names, land_mask_nest,
                    out_dir, n_depths, fps=2, custom_channels=None):
    os.makedirs(out_dir, exist_ok=True)
    C = gen.shape[1]
    L = gen.shape[2]
    H = gen.shape[3]

    gen_np, gen_ring, lm_ring, lon, lat = \
        _prepare_ring_data(gen, ch_names, land_mask_nest)

    if custom_channels is not None:
        channels = [c for c in custom_channels if c < C]
        obs_ring = _build_obs_ring(channels, obs, L, H)
        _make_one_gif(channels, "Custom", gen_ring, lm_ring,
                      lon, lat, obs, obs_ring, ch_names,
                      L, H, out_dir, fps)
        return

    # Thetao depths
    thetao_chs = list(range(0, min(n_depths, C)))
    obs_ring_th = _build_obs_ring(thetao_chs, obs, L, H)
    _make_one_gif(thetao_chs, "Thetao (depth profiles)",
                  gen_ring, lm_ring,
                  lon, lat, obs, obs_ring_th, ch_names,
                  L, H, out_dir, fps)

    # So depths
    so_chs = list(range(n_depths, min(2 * n_depths, C)))
    obs_ring_so = _build_obs_ring(so_chs, obs, L, H)
    _make_one_gif(so_chs, "So (depth profiles)",
                  gen_ring, lm_ring,
                  lon, lat, obs, obs_ring_so, ch_names,
                  L, H, out_dir, fps)

    # 2D surface vars (excluding siconc)
    surf_chs_all = list(range(2 * n_depths, C))
    siconc_chs = [ch for ch in surf_chs_all
                  if ch < len(ch_names) and ch_names[ch] == "siconc"]
    surf_chs = [ch for ch in surf_chs_all if ch not in siconc_chs]
    obs_ring_2d = _build_obs_ring(surf_chs, obs, L, H)
    _make_one_gif(surf_chs, "Surface 2D vars",
                  gen_ring, lm_ring,
                  lon, lat, obs, obs_ring_2d, ch_names,
                  L, H, out_dir, fps)




def sample_sda_real_obs(
    net,
    C: int, L: int, H: int,
    months: torch.Tensor,
    obs,
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
    B = 1

    if parallel_sda:
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
                                  dtype=torch.float32)

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
        description="SDA reconstruction with real observational data"
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--climatology", required=True)

    # Data product paths (observations)
    p.add_argument("--hadsst_path", default=None)
    p.add_argument("--crutem_path", default=None)
    p.add_argument("--icoads_path", default=None)
    p.add_argument("--en4_zarr_path", default=None)
    p.add_argument("--duacs_path", default=None)

    # SSH (altimetry) assimilation
    p.add_argument("--use_ssh_obs", action="store_true", default=False,
                   help="Assimilate DUACS altimetry SSH into the zos channel.")
    p.add_argument("--ssh_only", action="store_true", default=False,
                   help="Assimilate ONLY SSH (skip T/S profiles, SST, SLP). "
                        "Implies --use_ssh_obs.")
    p.add_argument("--ssh_coarsen_deg", type=float, default=5.0,
                   help="Coarse-grid resolution (deg) for DUACS SSH; coarser "
                        "averages out mesoscale eddies so the reconstruction "
                        "stays closer to the diffusion prior.")
    p.add_argument("--duacs_var", type=str, default="sla",
                   choices=["sla", "adt", "ssh", "zos"],
                   help="DUACS variable to assimilate. 'sla' (sea-level anomaly) "
                        "is the clean anomaly; 'adt' includes mean dynamic "
                        "topography (removed via the per-pixel climatology).")
    p.add_argument("--ssh_keep_med", action="store_true", default=False,
                   help="Keep Mediterranean/Black Sea SSH obs (default: excluded).")
    p.add_argument("--ssh_keep_global_mean", action="store_true", default=False,
                   help="Keep the per-timestep global-mean SSH (global sea-level "
                        "rise). Default removes it (model is detrended).")

    # Time range
    p.add_argument("--start_date", type=str, default="1990-01",
                   help="Start date YYYY-MM")
    p.add_argument("--end_date", type=str, default="1991-12",
                   help="End date YYYY-MM (inclusive)")

    # Observation parameters
    p.add_argument("--noise_obs_std", type=float, default=0.15)
    p.add_argument("--gamma", type=float, default=0.001)
    p.add_argument("--min_profiles", type=int, default=3,
                   help="Min EN4 profile count per bin to accept observation")
    p.add_argument("--use_model_clim", action="store_true", default=False,
                   help="Use model's own monthly mean for obs deseasonaling "
                        "(eliminates climatology mismatch domain shift)")
    p.add_argument("--matched_coarsening", action="store_true", default=False,
                   help="Match A(x) pipeline to y: denormalize -> coarsen -> "
                        "re-normalize, fixing coarsening non-commutativity")
    p.add_argument("--pixel_normalize", action="store_true", default=False,
                   help="Normalize obs per-HEALPix-pixel BEFORE coarsening "
                        "(matches A(x) pipeline; eliminates commutativity bias)")
    p.add_argument("--per_cell_sigma_obs", type=str, default=None,
                   help="Path to precomputed per-cell σ_obs .pt file "
                        "(from precompute_per_cell_sigma_obs.py). "
                        "When provided, uses per-cell σ_obs in likelihood "
                        "instead of scalar sigma_obs.")

    # Sampling
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--corrections", type=int, default=0)
    p.add_argument("--tau", type=float, default=0.3)
    p.add_argument("--corrector_t_threshold", type=float, default=0.0)
    p.add_argument("--output_frames", type=int, default=5)
    p.add_argument("--chunk_windows", type=int, default=2)

    # Parallel SDA
    p.add_argument("--parallel_sda", action="store_true", default=True)
    p.add_argument("--no_parallel_sda", dest="parallel_sda",
                   action="store_false")
    p.add_argument("--no_checkpoint", dest="use_checkpoint",
                   action="store_false", default=True)

    # Ensemble
    p.add_argument("--n_ensemble", type=int, default=1)
    p.add_argument("--ensemble_id", type=int, default=None)
    p.add_argument("--withhold_obs_fraction", type=float, default=0.0,
                   help="Fraction of obs to randomly withhold (0.0-1.0). "
                        "Withheld obs are saved for validation.")
    p.add_argument("--mask_from_era", type=str, default=None,
                   help="Restrict obs to spatial coverage of another era, "
                        "e.g. '1970-01:1979-12'. Loads obs for that period, "
                        "takes the spatial union of masks, and intersects "
                        "with the main era's masks.")

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="out_real_obs_v1")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fps", type=int, default=2)
    p.add_argument("--use_bf16", action="store_true")
    p.add_argument("--no_gifs", action="store_true")
    p.add_argument("--channels_to_plot", type=int, nargs="+", default=None)
    args = p.parse_args()
    if not args.ssh_only:
        missing = [n for n in ("hadsst_path", "crutem_path", "icoads_path", "en4_zarr_path")
                   if getattr(args, n) is None]
        if missing:
            p.error("missing " + ", ".join("--" + n for n in missing)
                    + " (required unless --ssh_only)")
    if (args.use_ssh_obs or args.ssh_only) and args.duacs_path is None:
        p.error("--duacs_path is required with --use_ssh_obs or --ssh_only")

    # ---- multi-GPU setup -------------------------------------------------------
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        if rank != 0:
            logging.getLogger("profiles_real_obs_v1").setLevel(logging.WARNING)
        logger.info(f"Distributed: rank={rank}/{world_size}, device={device}")
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- load model ------------------------------------------------------------
    net = load_checkpoint(args.checkpoint, device)
    land_mask_nest, depths, ch_names_clim, spatial_std_nest, monthly_mean_nest = \
        load_climatology(args.climatology)
    n_depths = len(depths)

    T = int(net.time_length)
    C = int(net.img_channels)
    H = int(net.domain.numel())
    logger.info(f"Model: C={C}, T={T}, H={H}")

    # ---- compute time range ----------------------------------------------------
    start_dt = _parse_date(args.start_date)
    end_dt   = _parse_date(args.end_date)
    n_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month) + 1
    logger.info(f"Time range: {start_dt:%Y-%m} -> {end_dt:%Y-%m} ({n_months} months)")

    if n_months < T:
        raise ValueError(
            f"Sequence length {n_months} < model window T={T}. "
            f"Request a longer date range (>= {T} months)."
        )

    # Compute month indices for DOY encoding
    month_indices = []
    for i in range(n_months):
        y = start_dt.year + (start_dt.month - 1 + i) // 12
        m = (start_dt.month - 1 + i) % 12
        month_indices.append(m)
    months = torch.tensor(month_indices, device=device, dtype=torch.long)
    L = n_months

    # ---- adjust L for clean windowing ------------------------------------------
    if args.parallel_sda:
        L_eff, n_windows, L_output = adjust_L_for_windows(
            L, T, args.output_frames,
        )
        if L_eff < L:
            logger.info(
                f"Trimming sequence for windowing: L={L} -> L_eff={L_eff} "
                f"({n_windows} windows, {L_output} output frames)"
            )
            months = months[:L_eff]
            L = L_eff
            n_months = L_eff

    # ---- build per-pixel-consistent normalization stats --------------------------
    logger.info("Building per-pixel normalization stats for obs grids ...")
    mapper_5deg_cpu = CoarseGridMapper(5.0, 'cpu')
    mapper_4deg_cpu = CoarseGridMapper(4.0, 'cpu')
    mapper_2deg_cpu = CoarseGridMapper(2.0, 'cpu')
    norm_mappers = {
        'tos': mapper_5deg_cpu, 'tas': mapper_5deg_cpu,
        'psl': mapper_2deg_cpu,
        'thetao': mapper_5deg_cpu, 'so': mapper_5deg_cpu,
    }
    use_ssh = args.use_ssh_obs or args.ssh_only
    if use_ssh:
        if abs(args.ssh_coarsen_deg - 5.0) < 1e-6:
            norm_mappers['zos'] = mapper_5deg_cpu
        elif abs(args.ssh_coarsen_deg - 2.0) < 1e-6:
            norm_mappers['zos'] = mapper_2deg_cpu
        else:
            norm_mappers['zos'] = CoarseGridMapper(args.ssh_coarsen_deg, 'cpu')

    norm_stats, std_padxy = build_norm_stats(
        spatial_std_nest, monthly_mean_nest, ch_names_clim, n_depths,
        norm_mappers,
    )

    # ---- load per-cell sigma_obs if provided ------------------------------------
    per_cell_sigma_obs = None
    if args.per_cell_sigma_obs:
        logger.info(f"Loading per-cell σ_obs: {args.per_cell_sigma_obs}")
        sigma_data = torch.load(args.per_cell_sigma_obs, map_location='cpu',
                                weights_only=False)
        per_cell_sigma_obs = sigma_data['per_cell_sigma']
        logger.info(f"  σ_base={sigma_data.get('sigma_base')}, "
                    f"channels: {list(per_cell_sigma_obs.keys())}")

    # ---- build real observation operator ---------------------------------------
    obs = RealObsMultiVarObs(
        n_depths=n_depths,
        C=C, L=L, H=H,
        hadsst_path=args.hadsst_path,
        crutem_path=args.crutem_path,
        icoads_path=args.icoads_path,
        en4_zarr_path=args.en4_zarr_path,
        norm_stats=norm_stats,
        model_depths=depths,
        start_dt=start_dt,
        noise_std=args.noise_obs_std,
        device=str(device),
        min_profiles=args.min_profiles,
        use_model_clim=args.use_model_clim,
        std_padxy=std_padxy if args.matched_coarsening else None,
        matched_coarsening=args.matched_coarsening,
        pixel_normalize=args.pixel_normalize,
        per_cell_sigma_obs=per_cell_sigma_obs,
        duacs_path=args.duacs_path,
        use_ssh_obs=use_ssh,
        ssh_only=args.ssh_only,
        ssh_coarsen_deg=args.ssh_coarsen_deg,
        ssh_remove_global_mean=not args.ssh_keep_global_mean,
        duacs_var=args.duacs_var,
        ssh_exclude_med=not args.ssh_keep_med,
    )

    if args.use_bf16:
        logger.info("BF16 autocast ENABLED")

    # ---- optionally restrict to spatial coverage of another era ----------------
    obs_field_full_era = None
    if args.mask_from_era:
        mask_start_str, mask_end_str = args.mask_from_era.split(':')
        mask_start_dt = dt.datetime.strptime(mask_start_str, '%Y-%m')
        mask_end_dt = dt.datetime.strptime(mask_end_str, '%Y-%m')
        mask_L = (mask_end_dt.year - mask_start_dt.year) * 12 + \
                 (mask_end_dt.month - mask_start_dt.month) + 1
        logger.info(f"Loading obs coverage for mask era: "
                    f"{mask_start_str} to {mask_end_str} ({mask_L} months) ...")

        mask_obs = RealObsMultiVarObs(
            n_depths=n_depths, C=C, L=mask_L, H=H,
            hadsst_path=args.hadsst_path,
            crutem_path=args.crutem_path,
            icoads_path=args.icoads_path,
            en4_zarr_path=args.en4_zarr_path,
            norm_stats=norm_stats,
            model_depths=depths,
            start_dt=mask_start_dt,
            noise_std=0.0,
            device=str(device),
            min_profiles=args.min_profiles,
            use_model_clim=args.use_model_clim,
            pixel_normalize=args.pixel_normalize,
        )

        obs_field_full_era = build_obs_field(obs, C, L, H)
        n_before = 0
        n_after = 0
        for ch, (mapper, masks, values) in obs.channel_obs.items():
            if ch in mask_obs.channel_obs:
                _, mask_era_masks, _ = mask_obs.channel_obs[ch]
                spatial_union = np.zeros(masks[0].shape, dtype=bool)
                for t_m in range(mask_L):
                    spatial_union |= mask_era_masks[t_m]
                for t in range(L):
                    n_before += masks[t].sum()
                    masks[t] &= spatial_union
                    n_after += masks[t].sum()
                logger.info(f"  ch {ch}: mask-era spatial coverage = "
                            f"{spatial_union.sum()} / {len(spatial_union)} cells")
            else:
                for t in range(L):
                    n_before += masks[t].sum()
                    masks[t][:] = False
                    n_after += 0
                logger.info(f"  ch {ch}: no data in mask era -> all masked out")

        logger.info(f"  Mask-from-era: {n_before:,} -> {n_after:,} obs "
                    f"({n_after / max(n_before, 1) * 100:.1f}% retained)")
        del mask_obs
        obs._rebuild_y(device=str(device))

    # ---- optionally withhold observations for validation -----------------------
    obs_field_full = None
    if args.withhold_obs_fraction > 0:
        obs_field_full = build_obs_field(obs, C, L, H)
        logger.info(f"Withholding {args.withhold_obs_fraction*100:.0f}% of observations ...")
        wh_rng = np.random.RandomState(args.seed + 999)
        n_withheld_total = 0
        n_kept_total = 0
        for ch, (mapper, masks, values) in obs.channel_obs.items():
            for t in range(L):
                if not masks[t].any():
                    continue
                obs_idx = np.where(masks[t])[0]
                n_obs_t = len(obs_idx)
                n_drop = int(round(n_obs_t * args.withhold_obs_fraction))
                if n_drop == 0:
                    n_kept_total += n_obs_t
                    continue
                drop_idx = wh_rng.choice(obs_idx, size=n_drop, replace=False)
                masks[t][drop_idx] = False
                n_withheld_total += n_drop
                n_kept_total += (n_obs_t - n_drop)
        logger.info(f"  Withheld {n_withheld_total:,} cell-months, "
                    f"kept {n_kept_total:,}")
        obs._rebuild_y(device=str(device))

    # ---- determine ensemble members to generate --------------------------------
    if args.ensemble_id is not None:
        member_ids = [args.ensemble_id]
    else:
        member_ids = list(range(args.n_ensemble))

    logger.info(f"Ensemble members to generate: {member_ids}")

    # ---- build observation field for saving ------------------------------------
    obs_field = build_obs_field(obs, C, L, H)
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

        x = sample_sda_real_obs(
            net=net, C=C, L=L, H=H,
            months=months, obs=obs,
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

    x_ensemble = torch.cat(generated_members, dim=0)

    # ---- save + GIF (rank 0 only) ------------------------------------------------
    if rank == 0:
        result_dict = {
            "generated": x_ensemble,
            "observations": obs_field,
            "observations_y": obs.y.cpu(),
            "months": months.cpu(),
            "norm_stats": {
                k: float(np.mean(v['inv_std']))
                for k, v in norm_stats.items()
            },
            "ensemble_member_ids": member_ids,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "args": vars(args),
        }
        if obs_field_full is not None:
            result_dict["observations_full"] = obs_field_full
            logger.info(f"Saving full (pre-withhold) obs field for validation")
        if args.mask_from_era and obs_field_full_era is not None:
            result_dict["observations_full"] = obs_field_full_era
            logger.info(f"Saving full (pre-era-mask) obs field for validation")

        if args.ensemble_id is not None:
            fname = f"results_e{args.ensemble_id:04d}.pt"
        else:
            fname = "results.pt"

        torch.save(result_dict, os.path.join(args.output_dir, fname))

        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

        logger.info(f"Results saved to {args.output_dir}/{fname}")

        if not args.no_gifs:
            ch_names = ch_names_clim if ch_names_clim else get_channel_names(depths)
            x_for_viz = x_ensemble[0:1].to(device)
            create_all_gifs(
                x_for_viz, obs, ch_names, land_mask_nest,
                args.output_dir, n_depths=n_depths, fps=args.fps,
                custom_channels=args.channels_to_plot,
            )

        logger.info(f"Done! Results in {args.output_dir}/")

    # ---- cleanup distributed ---------------------------------------------------
    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
