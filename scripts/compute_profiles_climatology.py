#!/usr/bin/env python3
"""
Monthly mean/std clim pickle from a profiles zarr (for --climatology).
"""

from __future__ import annotations

import argparse
import datetime
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import zarr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("[WARN] cartopy not available - Robinson projection plots disabled")

from earth2grid import healpix
from earth2grid.healpix import PixelOrder


VARIABLES_3D = ["thetao", "so"]
VARIABLES_2D = ["tos", "sos", "psl", "zos", "tas", "siconc"]
HPX_LEVEL = 6
NPIX = 12 * 4**HPX_LEVEL


def _minutes_to_datetime(minutes: int) -> datetime.datetime:
    base = datetime.datetime(1900, 1, 1)
    return base + datetime.timedelta(minutes=int(minutes))


def _build_channel_names(depths: np.ndarray) -> List[str]:
    """Build channel names: 3D vars x depths, then 2D vars."""
    names: List[str] = []
    for var in VARIABLES_3D:
        for depth in depths:
            names.append(f"{var}{int(round(depth))}m")
    for var in VARIABLES_2D:
        names.append(var)
    return names


def _get_channel_info(
    channel_names: List[str], depths: np.ndarray
) -> Tuple[Dict[str, List[int]], Dict[str, int]]:
    """
    Returns mappings from variable name to channel indices.
    
    For 3D vars: var_name -> list of depth channel indices
    For 2D vars: var_name -> single channel index
    """
    var_3d_channels: Dict[str, List[int]] = {var: [] for var in VARIABLES_3D}
    var_2d_channels: Dict[str, int] = {}
    
    for ch_idx, ch_name in enumerate(channel_names):
        # Check if it's a 3D variable (has depth suffix like "thetao25m")
        matched = False
        for var in VARIABLES_3D:
            if ch_name.startswith(var) and ch_name[len(var):].rstrip("m").isdigit():
                var_3d_channels[var].append(ch_idx)
                matched = True
                break
        if not matched:
            # It's a 2D variable
            for var in VARIABLES_2D:
                if ch_name == var:
                    var_2d_channels[var] = ch_idx
                    break
    
    return var_3d_channels, var_2d_channels


def compute_climatology(
    store: zarr.Group,
    experiment: str,
    n_members: Optional[int],
    start_year: int,
    end_year: int,
    siconc_eps: float,
) -> Dict[str, np.ndarray]:
    """
    Compute monthly climatology for all channels.
    
    For EDM (Karras 2022), we need:
    - Monthly mean per pixel (for deseasoning)
    - Per-pixel spatial std per channel (map for each channel)
    
    After normalization: (data - mean) / spatial_std
    Then in dataloader: * SIGMA_DATA (0.5) → sigma_data = 0.5
    
    Note on spatial_std:
    - Most variables: per-pixel std computed over time and members
    - siconc (sea ice): single scalar (computed over ice-prone regions only),
      then broadcast to map shape for consistency
    - A minimum threshold is applied to prevent division by near-zero values
    
    Returns:
        Dictionary with:
        - mean: (12, n_channels, n_space) monthly mean per pixel
        - spatial_std: (n_channels, n_space) per-pixel std (map for each channel)
        - land_mask: (n_channels, n_space) per-channel land mask (True = land)
        - per_pixel_std: True (flag indicating per-pixel std format)
        - ... metadata
    """
    experiments = list(store["experiment"][:])
    members = list(store["member"][:])
    if experiment not in experiments:
        raise ValueError(f"Experiment {experiment} not found in zarr.")
    exp_idx = experiments.index(experiment)

    depths = np.asarray(store["depth"][:], dtype=np.float32)
    channel_names = _build_channel_names(depths)
    n_channels = len(channel_names)
    n_space = store[VARIABLES_2D[0]].shape[-1]

    # Determine which members to use
    if n_members is None:
        members_to_use = members  # Use all members
    else:
        members_to_use = members[:n_members]
    
    print(f"Computing climatology for {experiment} with {len(members_to_use)} members (all={n_members is None})")
    print(f"  Years: {start_year}-{end_year}")
    print(f"  Channels: {n_channels} ({len(VARIABLES_3D) * len(depths)} 3D + {len(VARIABLES_2D)} 2D)")
    print(f"  Spatial points: {n_space}")
    print(f"  Using members: {members_to_use}")

    # Accumulators for mean
    sum_arr = np.zeros((12, n_channels, n_space), dtype=np.float64)
    count_arr = np.zeros((12, n_channels, n_space), dtype=np.int64)
    
    # Track which pixels are always NaN (land mask per channel)
    ever_valid = np.zeros((n_channels, n_space), dtype=bool)
    
    # First pass: compute mean
    for mem in members_to_use:
        mem_idx = members.index(mem)
        valid_len = int(store["time_valid_length"][exp_idx, mem_idx])
        if valid_len <= 0:
            continue
        times = store["time"][exp_idx, mem_idx, :valid_len]
        
        print(f"  Processing member {mem} ({valid_len} time steps)...")
        
        for t_idx in range(valid_len):
            minutes = int(times[t_idx])
            if minutes < 0:
                continue
            dt = _minutes_to_datetime(minutes)
            if dt.year < start_year or dt.year > end_year:
                continue
            month_idx = dt.month - 1

            # Build channel data
            channels: List[np.ndarray] = []
            for var in VARIABLES_3D:
                var_data = store[var][exp_idx, mem_idx, t_idx, :, :]  # (depth, space)
                for d_idx in range(var_data.shape[0]):
                    channels.append(var_data[d_idx])
            for var in VARIABLES_2D:
                channels.append(store[var][exp_idx, mem_idx, t_idx, :])

            stacked = np.stack(channels, axis=0).astype(np.float64)  # (n_channels, n_space)
            valid = np.isfinite(stacked)
            
            sum_arr[month_idx] += np.where(valid, stacked, 0.0)
            count_arr[month_idx] += valid.astype(np.int64)
            ever_valid |= valid

    # Compute monthly mean per pixel
    mean = np.divide(
        sum_arr, 
        np.maximum(count_arr, 1), 
        out=np.zeros_like(sum_arr), 
        where=count_arr > 0
    )
    
    # Build land mask: True where pixel was never valid
    land_mask = ~ever_valid  # (n_channels, n_space)
    
    print(f"  Land mask stats per channel:")
    for ch_idx, ch_name in enumerate(channel_names):
        pct_land = land_mask[ch_idx].sum() / n_space * 100
        print(f"    {ch_name}: {pct_land:.1f}% land")

    # Second pass: compute PER-PIXEL spatial std per channel
    # This allows different std at each location (better for heterogeneous fields)
    # Exception: siconc uses single scalar (broadcast to map) for ice-prone regions
    pixel_sumsq = np.zeros((n_channels, n_space), dtype=np.float64)
    pixel_count = np.zeros((n_channels, n_space), dtype=np.int64)
    
    # Special accumulators for siconc (single global scalar for ice-prone pixels)
    siconc_idx = channel_names.index("siconc")
    siconc_sumsq = 0.0
    siconc_count = 0
    
    # Pre-compute ice-prone mask for siconc
    annual_mean_siconc = mean[:, siconc_idx, :].mean(axis=0)  # (n_space,)
    ice_prone_mask = annual_mean_siconc > siconc_eps

    for mem in members_to_use:
        mem_idx = members.index(mem)
        valid_len = int(store["time_valid_length"][exp_idx, mem_idx])
        if valid_len <= 0:
            continue
        times = store["time"][exp_idx, mem_idx, :valid_len]
        
        for t_idx in range(valid_len):
            minutes = int(times[t_idx])
            if minutes < 0:
                continue
            dt = _minutes_to_datetime(minutes)
            if dt.year < start_year or dt.year > end_year:
                continue
            month_idx = dt.month - 1

            channels = []
            for var in VARIABLES_3D:
                var_data = store[var][exp_idx, mem_idx, t_idx, :, :]
                for d_idx in range(var_data.shape[0]):
                    channels.append(var_data[d_idx])
            for var in VARIABLES_2D:
                channels.append(store[var][exp_idx, mem_idx, t_idx, :])

            stacked = np.stack(channels, axis=0).astype(np.float64)
            valid = np.isfinite(stacked)
            anom = stacked - mean[month_idx]  # Deseasoned anomaly

            # Accumulate sum of squares PER PIXEL (except siconc)
            for ch_idx in range(n_channels):
                if ch_idx == siconc_idx:
                    continue  # Handle siconc separately
                ch_valid = valid[ch_idx]
                ch_anom = np.where(ch_valid, anom[ch_idx], 0.0)
                pixel_sumsq[ch_idx] += ch_anom * ch_anom
                pixel_count[ch_idx] += ch_valid.astype(np.int64)

            # Special handling for siconc: single GLOBAL scalar for ice-prone pixels
            siconc_valid = valid[siconc_idx] & ice_prone_mask
            if np.any(siconc_valid):
                siconc_anom = anom[siconc_idx, siconc_valid]
                siconc_sumsq += np.sum(siconc_anom * siconc_anom)
                siconc_count += siconc_valid.sum()

    # Compute per-pixel spatial std
    spatial_std = np.sqrt(
        np.divide(
            pixel_sumsq,
            np.maximum(pixel_count, 1),
            out=np.ones((n_channels, n_space), dtype=np.float64),
            where=pixel_count > 0,
        )
    )
    
    # For siconc: use single scalar broadcast to full map
    if siconc_count > 0:
        siconc_std_scalar = np.sqrt(siconc_sumsq / siconc_count)
    else:
        siconc_std_scalar = 1.0
    spatial_std[siconc_idx, :] = siconc_std_scalar  # Broadcast scalar to all pixels
    
    # Apply minimum std threshold per channel to prevent division by tiny values
    # Use 10% of the channel's mean std as the floor (but at least 0.01)
    for ch_idx in range(n_channels):
        if ch_idx == siconc_idx:
            continue  # siconc already handled
        ch_std = spatial_std[ch_idx]
        valid_mask = ~land_mask[ch_idx]
        if np.any(valid_mask):
            mean_std = np.mean(ch_std[valid_mask])
            min_std = max(0.1 * mean_std, 0.01)  # 10% of mean or 0.01, whichever is larger
            spatial_std[ch_idx] = np.maximum(ch_std, min_std)
    
    # Prevent division by zero (safety)
    spatial_std = np.where(spatial_std == 0, 1.0, spatial_std)
    
    # Print std stats (show mean and range per channel)
    print(f"\n  Per-pixel spatial std per channel:")
    for ch_idx, ch_name in enumerate(channel_names):
        ch_std = spatial_std[ch_idx]
        valid_std = ch_std[~land_mask[ch_idx]] if np.any(~land_mask[ch_idx]) else ch_std
        if ch_idx == siconc_idx:
            print(f"    {ch_name}: std = {siconc_std_scalar:.4f} (single scalar for ice-prone regions)")
        else:
            print(f"    {ch_name}: std mean={valid_std.mean():.4f}, min={valid_std.min():.4f}, max={valid_std.max():.4f}")
    
    # Set land pixels in mean to 0 (will be masked anyway)
    for ch_idx in range(n_channels):
        for m in range(12):
            mean[m, ch_idx, land_mask[ch_idx]] = 0.0

    return {
        "mean": mean.astype(np.float32),  # (12, n_channels, n_space) - per-pixel monthly mean
        "spatial_std": spatial_std.astype(np.float32),  # (n_channels, n_space) - per-pixel std!
        "land_mask": land_mask,  # (n_channels, n_space)
        "channel_names": channel_names,
        "depths": depths,
        "variables_2d": VARIABLES_2D,
        "variables_3d": VARIABLES_3D,
        "experiment": experiment,
        "members_used": members_to_use,
        "start_year": start_year,
        "end_year": end_year,
        "monthly": True,
        "siconc_eps": siconc_eps,
        "per_pixel_std": True,  # Flag to indicate new per-pixel std format
    }


def _plot_map_scatter(
    lon: np.ndarray,
    lat: np.ndarray,
    values: np.ndarray,
    title: str,
    out_path: Path,
    cmap: str = "coolwarm",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """Plot a map using scatter on Robinson projection."""
    if HAS_CARTOPY:
        fig = plt.figure(figsize=(10, 5))
        ax = plt.axes(projection=ccrs.Robinson())
        ax.coastlines(linewidth=0.5)
        sc = ax.scatter(
            lon, lat, c=values, s=1, transform=ccrs.PlateCarree(),
            cmap=cmap, vmin=vmin, vmax=vmax,
        )
        plt.colorbar(sc, ax=ax, shrink=0.7, orientation="horizontal", pad=0.05)
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        sc = ax.scatter(lon, lat, c=values, s=1, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(sc, ax=ax, shrink=0.7)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_zonal_mean(
    lat: np.ndarray,
    values_3d: np.ndarray,  # (depth, space)
    depths: np.ndarray,
    title: str,
    out_path: Path,
    lat_bin_size: float = 2.0,
    cmap: str = "coolwarm",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """Plot zonal mean depth-latitude."""
    lat_bins = np.arange(-90.0, 90.0 + lat_bin_size, lat_bin_size)
    lat_centers = 0.5 * (lat_bins[:-1] + lat_bins[1:])
    bin_idx = np.digitize(lat, lat_bins) - 1

    zonal = np.full((len(depths), len(lat_centers)), np.nan, dtype=np.float64)
    for d_idx in range(values_3d.shape[0]):
        for b in range(len(lat_centers)):
            mask = bin_idx == b
            if np.any(mask):
                valid_vals = values_3d[d_idx, mask]
                valid_vals = valid_vals[np.isfinite(valid_vals)]
                if len(valid_vals) > 0:
                    zonal[d_idx, b] = np.mean(valid_vals)

    fig, ax = plt.subplots(figsize=(8, 5))
    mesh = ax.pcolormesh(
        lat_centers, depths, zonal, cmap=cmap, shading="auto",
        vmin=vmin, vmax=vmax,
    )
    fig.colorbar(mesh, ax=ax, shrink=0.8)
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Depth (m)")
    ax.invert_yaxis()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_climatology_diagnostics(
    store: zarr.Group,
    climatology: Dict[str, np.ndarray],
    output_dir: Path,
    sample_month: int = 0,  # 0 = January
) -> None:
    """
    Plot before/after normalization maps for all variables.
    
    Creates:
    - Before normalization: raw data maps
    - After normalization: (data - mean) / spatial_std maps (EDM style)
    - Mean maps
    - Std bar chart (since std is now global per channel)
    - Zonal mean plots for 3D variables
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    experiments = list(store["experiment"][:])
    members = list(store["member"][:])
    exp = climatology["experiment"]
    exp_idx = experiments.index(exp)
    mem_idx = 0  # Use first member for visualization
    
    depths = climatology["depths"]
    channel_names = climatology["channel_names"]
    mean = climatology["mean"]  # (12, n_channels, n_space) - monthly mean per pixel
    spatial_std = climatology["spatial_std"]  # (n_channels,) - global std per channel!
    land_mask = climatology["land_mask"]  # (n_channels, n_space)
    
    n_channels = len(channel_names)
    n_space = mean.shape[-1]
    
    # Get grid coordinates
    grid = healpix.Grid(HPX_LEVEL, pixel_order=PixelOrder.NEST)
    lat = grid.lat
    lon = grid.lon
    
    # Find a valid time index in the target month
    valid_len = int(store["time_valid_length"][exp_idx, mem_idx])
    times = store["time"][exp_idx, mem_idx, :valid_len]
    
    sample_t_idx = None
    for t_idx in range(valid_len):
        dt = _minutes_to_datetime(int(times[t_idx]))
        if dt.month == sample_month + 1:
            sample_t_idx = t_idx
            break
    
    if sample_t_idx is None:
        print(f"[WARN] No time step found for month {sample_month + 1}, using first available")
        sample_t_idx = 0
    
    sample_dt = _minutes_to_datetime(int(times[sample_t_idx]))
    month_idx = sample_dt.month - 1
    
    print(f"Plotting diagnostics for {sample_dt.strftime('%Y-%m-%d')} (month {month_idx + 1})")
    
    # Load sample data
    raw_channels: List[np.ndarray] = []
    for var in VARIABLES_3D:
        var_data = store[var][exp_idx, mem_idx, sample_t_idx, :, :]
        for d_idx in range(var_data.shape[0]):
            raw_channels.append(var_data[d_idx])
    for var in VARIABLES_2D:
        raw_channels.append(store[var][exp_idx, mem_idx, sample_t_idx, :])
    
    raw_data = np.stack(raw_channels, axis=0).astype(np.float32)  # (n_channels, n_space)
    
    # Compute normalized data using EDM formula:
    # (data - monthly_mean) / spatial_std
    month_mean = mean[month_idx]  # (n_channels, n_space)
    # spatial_std is now (n_channels, n_space) - same shape as data, no broadcasting needed
    normalized_data = (raw_data - month_mean) / spatial_std
    
    # Apply land mask (set to NaN for visualization)
    raw_data_masked = np.where(land_mask, np.nan, raw_data)
    normalized_data_masked = np.where(land_mask, np.nan, normalized_data)
    month_mean_masked = np.where(land_mask, np.nan, month_mean)
    
    # Print normalized data statistics to verify EDM scaling
    print(f"\n  Normalized data statistics (should have std ≈ 1):")
    for ch_idx, ch_name in enumerate(channel_names):
        valid_data = normalized_data_masked[ch_idx][np.isfinite(normalized_data_masked[ch_idx])]
        if len(valid_data) > 0:
            actual_std = np.std(valid_data)
            print(f"    {ch_name}: std = {actual_std:.4f}, mean = {np.mean(valid_data):.4f}")
    
    # Get channel mappings
    var_3d_channels, var_2d_channels = _get_channel_info(channel_names, depths)
    
    # -------------------------------------------------------------------------
    # Plot mean spatial std bar chart (spatial_std is now per-pixel)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(n_channels)
    # For per-pixel std, compute mean over valid (non-land) pixels per channel
    if spatial_std.ndim == 2:
        mean_std_per_channel = np.array([
            np.mean(spatial_std[ch, ~land_mask[ch]]) if np.any(~land_mask[ch]) else np.mean(spatial_std[ch])
            for ch in range(n_channels)
        ])
    else:
        mean_std_per_channel = spatial_std
    bars = ax.bar(x, mean_std_per_channel, color='steelblue')
    ax.set_xlabel("Channel")
    ax.set_ylabel("Mean Spatial Std (over valid pixels)")
    ax.set_title("Per-Pixel Spatial Std: Mean per Channel")
    ax.set_xticks(x[::max(1, n_channels // 20)])  # Show every Nth label
    ax.set_xticklabels([channel_names[i] for i in x[::max(1, n_channels // 20)]], rotation=45, ha='right')
    ax.axhline(y=1.0, color='red', linestyle='--', label='std=1 (ideal)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "spatial_std_per_channel.png", dpi=150)
    plt.close(fig)
    
    # -------------------------------------------------------------------------
    # Plot 2D variables
    # -------------------------------------------------------------------------
    for var in VARIABLES_2D:
        if var not in var_2d_channels:
            continue
        ch_idx = var_2d_channels[var]
        
        # Raw data
        _plot_map_scatter(
            lon, lat, raw_data_masked[ch_idx],
            f"{var} - Raw ({sample_dt.strftime('%Y-%m')})",
            output_dir / f"{var}_raw.png",
        )
        
        # Normalized data (EDM: should be ~N(0,1) before SIGMA_DATA scaling)
        # For per-pixel std, show mean std in title
        ch_std = spatial_std[ch_idx]
        mean_ch_std = np.mean(ch_std[~land_mask[ch_idx]]) if np.any(~land_mask[ch_idx]) else np.mean(ch_std)
        _plot_map_scatter(
            lon, lat, normalized_data_masked[ch_idx],
            f"{var} - Normalized (mean std={mean_ch_std:.2f})",
            output_dir / f"{var}_normalized.png",
            vmin=-3, vmax=3,
        )
        
        # Mean
        _plot_map_scatter(
            lon, lat, month_mean_masked[ch_idx],
            f"{var} - Monthly Mean (month {month_idx + 1})",
            output_dir / f"{var}_mean.png",
        )
    
    # -------------------------------------------------------------------------
    # Plot 3D variables
    # -------------------------------------------------------------------------
    for var in VARIABLES_3D:
        if var not in var_3d_channels or not var_3d_channels[var]:
            continue
        
        ch_indices = var_3d_channels[var]
        n_depths = len(ch_indices)
        
        # Collect data for all depths
        raw_3d = np.stack([raw_data_masked[i] for i in ch_indices], axis=0)  # (n_depths, n_space)
        norm_3d = np.stack([normalized_data_masked[i] for i in ch_indices], axis=0)
        mean_3d = np.stack([month_mean_masked[i] for i in ch_indices], axis=0)
        
        # Plot maps for all depth levels
        for d_idx in range(n_depths):
            depth_val = int(round(depths[d_idx]))
            ch_idx = ch_indices[d_idx]
            ch_std = spatial_std[ch_idx]
            # For per-pixel std, compute mean over valid pixels
            mean_ch_std = np.mean(ch_std[~land_mask[ch_idx]]) if np.any(~land_mask[ch_idx]) else np.mean(ch_std)
            
            _plot_map_scatter(
                lon, lat, raw_3d[d_idx],
                f"{var} {depth_val}m - Raw ({sample_dt.strftime('%Y-%m')})",
                output_dir / f"{var}_{depth_val}m_raw.png",
            )
            
            _plot_map_scatter(
                lon, lat, norm_3d[d_idx],
                f"{var} {depth_val}m - Normalized (mean std={mean_ch_std:.2f})",
                output_dir / f"{var}_{depth_val}m_normalized.png",
                vmin=-3, vmax=3,
            )
        
        # Zonal mean plots
        _plot_zonal_mean(
            lat, raw_3d, depths,
            f"{var} - Raw Zonal Mean ({sample_dt.strftime('%Y-%m')})",
            output_dir / f"{var}_zonal_raw.png",
        )
        
        _plot_zonal_mean(
            lat, norm_3d, depths,
            f"{var} - Normalized Zonal Mean",
            output_dir / f"{var}_zonal_normalized.png",
            vmin=-3, vmax=3,
        )
        
        _plot_zonal_mean(
            lat, mean_3d, depths,
            f"{var} - Mean Zonal Mean (month {month_idx + 1})",
            output_dir / f"{var}_zonal_mean.png",
        )
        
        # Plot mean std vs depth for this 3D variable
        var_stds = []
        for ch_idx in ch_indices:
            ch_std = spatial_std[ch_idx]
            mean_ch_std = np.mean(ch_std[~land_mask[ch_idx]]) if np.any(~land_mask[ch_idx]) else np.mean(ch_std)
            var_stds.append(mean_ch_std)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(var_stds, depths, 'o-', color='steelblue', linewidth=2)
        ax.set_xlabel("Mean Spatial Std (over valid pixels)")
        ax.set_ylabel("Depth (m)")
        ax.invert_yaxis()
        ax.set_title(f"{var} - Mean Std vs Depth")
        ax.axvline(x=1.0, color='red', linestyle='--', label='std=1')
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{var}_std_vs_depth.png", dpi=150)
        plt.close(fig)
    
    # -------------------------------------------------------------------------
    # Plot land mask
    # -------------------------------------------------------------------------
    # Plot land mask for a few channels
    for var in VARIABLES_2D[:2]:  # Just first two 2D vars
        if var in var_2d_channels:
            ch_idx = var_2d_channels[var]
            _plot_map_scatter(
                lon, lat, land_mask[ch_idx].astype(float),
                f"{var} - Land Mask",
                output_dir / f"{var}_land_mask.png",
                cmap="binary",
            )
    
    # Plot land mask for 3D vars at different depths
    for var in VARIABLES_3D:
        if var not in var_3d_channels:
            continue
        ch_indices = var_3d_channels[var]
        for d_idx in range(len(ch_indices)):
            ch_idx = ch_indices[d_idx]
            depth_val = int(round(depths[d_idx]))
            _plot_map_scatter(
                lon, lat, land_mask[ch_idx].astype(float),
                f"{var} {depth_val}m - Land Mask",
                output_dir / f"{var}_{depth_val}m_land_mask.png",
                cmap="binary",
            )
    
    print(f"Saved diagnostic plots to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser("Compute MIROC6 profiles climatology")
    parser.add_argument("--zarr_path", required=True)
    parser.add_argument("--output_path", required=True, help="Path for pickle output")
    parser.add_argument("--experiment", default="historical")
    parser.add_argument("--n_members", type=int, default=None, help="Number of members to use (default: all)")
    parser.add_argument("--start_year", type=int, default=1990)
    parser.add_argument("--end_year", type=int, default=2014)
    parser.add_argument("--siconc_eps", type=float, default=0.05)
    parser.add_argument("--plot_dir", default=None, help="Directory for diagnostic plots")
    parser.add_argument("--plot_month", type=int, default=1, help="Month for sample plots (1-12)")
    args = parser.parse_args()

    store = zarr.open(args.zarr_path, mode="r")
    
    climatology = compute_climatology(
        store=store,
        experiment=args.experiment,
        n_members=args.n_members,
        start_year=args.start_year,
        end_year=args.end_year,
        siconc_eps=args.siconc_eps,
    )

    # Save climatology
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(climatology, f)
    print(f"Saved climatology to {output_path}")
    
    # Generate diagnostic plots if requested
    if args.plot_dir is not None:
        plot_dir = Path(args.plot_dir)
        plot_climatology_diagnostics(
            store=store,
            climatology=climatology,
            output_dir=plot_dir,
            sample_month=args.plot_month - 1,  # Convert to 0-indexed
        )


if __name__ == "__main__":
    main()
