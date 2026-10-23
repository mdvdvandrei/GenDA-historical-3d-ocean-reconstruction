#!/usr/bin/env python3
"""
Monthly CMIP NetCDFs → HEALPix-6 profiles zarr (MIROC6 / CESM2).

  python scripts/prepare_profiles_zarr.py \\
    --input_dir /path/to/MIROC6 \\
    --output_zarr miroc6_profiles.zarr \\
    --experiments historical piControl

  python scripts/prepare_profiles_zarr.py \\
    --input_dir /path/to/CESM2 \\
    --output_zarr cesm2_profiles.zarr \\
    --experiments historical --resume
"""

import argparse
import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import datetime

import numpy as np
import pandas as pd
import xarray as xr
import zarr
from numcodecs import Blosc
from earth2grid import healpix
import cftime

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# HEALPix level 6 configuration
HPX_LEVEL = 6
NPIX = 12 * 4**HPX_LEVEL  # 49,152 pixels

# Variables to process
VARIABLES_3D = ["thetao", "so"]
VARIABLES_2D = ["tos", "sos", "psl", "zos", "tas", "siconc"]
ALL_VARIABLES = VARIABLES_2D + VARIABLES_3D

# Time chunk size for zarr
TIME_CHUNK = 12  # 12 months per chunk

# Depth configuration
TARGET_DEPTHS = np.array(
    [25, 75, 150, 300, 500, 700, 1000, 1500, 2000], dtype=np.float32
)
DEEP_LEVEL_VALUE = 9999.0  # sentinel for >2000m mean
N_DEPTH = len(TARGET_DEPTHS) + 1

DEPTH_DIM_CANDIDATES = [
    "depth",
    "lev",
    "olevel",
    "z",
    "deptht",
    "deptho",
]


@dataclass
class ExperimentInfo:
    """Information about a CMIP6 experiment."""

    name: str
    description: str
    start_year: Optional[int]
    end_year: Optional[int]
    forcing: str


EXPERIMENT_METADATA = {
    "historical": ExperimentInfo(
        name="historical",
        description="Historical simulation (1850-2014)",
        start_year=1850,
        end_year=2014,
        forcing="Historical forcing (GHG, aerosols, land use, etc.)",
    ),
    "piControl": ExperimentInfo(
        name="piControl",
        description="Pre-industrial control (constant 1850 forcing)",
        start_year=None,
        end_year=None,
        forcing="Constant pre-industrial (1850) forcing",
    ),
    "ssp119": ExperimentInfo(
        name="ssp119",
        description="SSP1-1.9: Sustainability, very low emissions",
        start_year=2015,
        end_year=2100,
        forcing="SSP1 socioeconomics + RCP1.9 forcing (~1.5°C)",
    ),
    "ssp126": ExperimentInfo(
        name="ssp126",
        description="SSP1-2.6: Sustainability, low emissions",
        start_year=2015,
        end_year=2100,
        forcing="SSP1 socioeconomics + RCP2.6 forcing (~2°C)",
    ),
    "ssp245": ExperimentInfo(
        name="ssp245",
        description="SSP2-4.5: Middle of the road",
        start_year=2015,
        end_year=2100,
        forcing="SSP2 socioeconomics + RCP4.5 forcing (~2.7°C)",
    ),
    "ssp370": ExperimentInfo(
        name="ssp370",
        description="SSP3-7.0: Regional rivalry, high emissions",
        start_year=2015,
        end_year=2100,
        forcing="SSP3 socioeconomics + RCP7.0 forcing (~3.6°C)",
    ),
    "ssp585": ExperimentInfo(
        name="ssp585",
        description="SSP5-8.5: Fossil-fueled development, very high emissions",
        start_year=2015,
        end_year=2100,
        forcing="SSP5 socioeconomics + RCP8.5 forcing (~4.4°C)",
    ),
}


def check_cdo() -> bool:
    """Check if CDO is available."""
    try:
        subprocess.run(["cdo", "-V"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("CDO (Climate Data Operators) is not available.")
        return False


def create_grid_file(level: int, path: str) -> None:
    """Create a CDO grid description file for HEALPix using earth2grid."""
    logger.info(f"Generating grid file for level {level} at {path}")
    grid = healpix.Grid(level, pixel_order=healpix.PixelOrder.NEST)

    lats = grid.lat
    lons = grid.lon
    size = len(lats)

    with open(path, "w") as f:
        f.write("gridtype = unstructured\n")
        f.write(f"gridsize = {size}\n")
        f.write("xname = lon\n")
        f.write("xunits = degrees_east\n")
        f.write("yname = lat\n")
        f.write("yunits = degrees_north\n")
        f.write("xvals = ")
        np.savetxt(f, lons, fmt="%.6f", newline=" ")
        f.write("\n")
        f.write("yvals = ")
        np.savetxt(f, lats, fmt="%.6f", newline=" ")
        f.write("\n")


def regrid_to_hpx6(
    input_file: str, output_file: str, variable: str, grid_file: str
) -> str:
    """Regrid a NetCDF file to HEALPix level 6 using CDO."""
    cmd = [
        "cdo",
        "-L",
        "-s",
        f"-remapbil,{grid_file}",
        f"-selname,{variable}",
        input_file,
        output_file,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"CDO failed: {result.stderr}")
        raise RuntimeError(f"CDO failed for {variable} in {input_file}")

    return output_file


def _depth_units_to_meters(values: np.ndarray, units: Optional[str]) -> np.ndarray:
    if units is None:
        return values.astype(np.float32)
    unit = units.lower().strip()
    if unit in ("cm", "centimeter", "centimeters"):
        return (values / 100.0).astype(np.float32)
    if unit in ("km", "kilometer", "kilometers"):
        return (values * 1000.0).astype(np.float32)
    return values.astype(np.float32)


def _find_depth_dim(da: xr.DataArray) -> Optional[str]:
    for dim in da.dims:
        if dim in DEPTH_DIM_CANDIDATES:
            return dim
    for dim in da.dims:
        coord = da.coords.get(dim)
        if coord is None:
            continue
        units = str(coord.attrs.get("units", "")).lower()
        if "meter" in units or units.endswith("m"):
            return dim
    return None


def _interpolate_depth(da: xr.DataArray) -> xr.DataArray:
    """Bin-average source depth levels to TARGET_DEPTHS + one deep level.

    Uses pure numpy to avoid xarray apply_ufunc issues on unstructured
    grids (e.g. HEALPix ncells dimension).
    """
    depth_dim = _find_depth_dim(da)
    if depth_dim is None:
        raise ValueError("Depth dimension not found for 3D variable.")

    depth_coord = da[depth_dim]
    depth_m = _depth_units_to_meters(
        depth_coord.values, depth_coord.attrs.get("units"),
    )

    depth_axis = list(da.dims).index(depth_dim)
    vals = da.values  # eager load — typically (time, lev, ncells)

    if len(depth_m) > 1 and np.all(np.diff(depth_m) < 0):
        order = np.argsort(depth_m)
        depth_m = depth_m[order]
        vals = np.take(vals, order, axis=depth_axis)

    # Bin edges: midpoints between target depths, 0 at top, 2000 at bottom
    edges = np.zeros(len(TARGET_DEPTHS) + 1, dtype=np.float64)
    edges[0] = 0.0
    edges[-1] = 2000.0
    for i in range(1, len(TARGET_DEPTHS)):
        edges[i] = 0.5 * (TARGET_DEPTHS[i - 1] + TARGET_DEPTHS[i])

    n_target = len(TARGET_DEPTHS)
    # Move depth axis to position 0 for easier indexing
    vals = np.moveaxis(vals, depth_axis, 0)  # (lev, time, ncells) or (lev, ...)
    rest_shape = vals.shape[1:]

    out = np.full((n_target + 1,) + rest_shape, np.nan, dtype=np.float32)

    for i in range(n_target):
        if i == 0:
            mask = (depth_m >= edges[i]) & (depth_m <= edges[i + 1])
        else:
            mask = (depth_m > edges[i]) & (depth_m <= edges[i + 1])
        idx = np.where(mask)[0]
        if len(idx) > 0:
            out[i] = np.nanmean(vals[idx], axis=0)

    # Deep level: mean of everything > 2000 m
    deep_mask = depth_m > 2000.0
    deep_idx = np.where(deep_mask)[0]
    if len(deep_idx) > 0:
        out[n_target] = np.nanmean(vals[deep_idx], axis=0)
    else:
        out[n_target] = vals[-1]  # fallback: deepest available level

    # Move depth axis back to original position
    out = np.moveaxis(out, 0, depth_axis)

    all_depths = np.concatenate([TARGET_DEPTHS, [DEEP_LEVEL_VALUE]])
    new_coords = {k: v for k, v in da.coords.items()
                  if k != depth_dim and k in da.dims}
    new_coords[depth_dim] = all_depths
    result = xr.DataArray(
        out,
        dims=da.dims,
        coords=new_coords,
    )
    return result


def _transpose_time_depth_space(da: xr.DataArray) -> xr.DataArray:
    time_dim = "time" if "time" in da.dims else da.dims[0]
    depth_dim = _find_depth_dim(da)
    other_dims = [d for d in da.dims if d != time_dim and d != depth_dim]
    if len(other_dims) != 1:
        raise ValueError(f"Expected 1 spatial dim, found {other_dims}")
    space_dim = other_dims[0]
    order = [time_dim]
    if depth_dim is not None:
        order.append(depth_dim)
    order.append(space_dim)
    return da.transpose(*order)


def discover_members(input_dir: Path, experiment: str) -> List[str]:
    """Discover all available ensemble members for an experiment."""
    exp_dir = input_dir / experiment
    if not exp_dir.exists():
        logger.warning(f"Experiment directory {exp_dir} does not exist")
        return []

    members = []
    for member_dir in sorted(exp_dir.iterdir()):
        if not member_dir.is_dir() or not member_dir.name.startswith("r"):
            continue
        for var in ALL_VARIABLES:
            var_dir = member_dir / var
            if var_dir.exists() and list(var_dir.glob("*.nc")):
                members.append(member_dir.name)
                break

    return sorted(members, key=lambda x: int(x.split("i")[0][1:]))


def _time_key(t) -> Tuple[int, int]:
    """Monthly-resolution key so atmos/ocean timestamps for the same month merge."""
    return (t.year, t.month)


def _time_to_minutes(t) -> int:
    if isinstance(t, (np.datetime64, pd.Timestamp, datetime.datetime)):
        return int((pd.Timestamp(t) - pd.Timestamp("1900-01-01")).total_seconds() / 60.0)
    if isinstance(t, cftime.datetime):
        ref = type(t)(1900, 1, 1)  # preserves calendar (noleap, 360day, etc.)
        delta = t - ref
        return int(delta.total_seconds() / 60.0)
    return int((pd.Timestamp(str(t)) - pd.Timestamp("1900-01-01")).total_seconds() / 60.0)


def get_member_time_range(
    input_dir: Path, experiment: str, member: str
) -> Tuple[pd.DatetimeIndex, int]:
    """Get time range for a specific member."""
    all_times: Dict[Tuple[int, int], object] = {}
    for var in ALL_VARIABLES:
        var_dir = input_dir / experiment / member / var
        if not var_dir.exists():
            continue
        for nc_file in sorted(var_dir.glob("*.nc")):
            try:
                with xr.open_dataset(str(nc_file), use_cftime=True) as ds:
                    for t in ds.time.values:
                        all_times[_time_key(t)] = t
            except Exception as e:
                logger.warning(f"Could not read {nc_file}: {e}")

    if not all_times:
        return pd.DatetimeIndex([]), 0

    sorted_times = [all_times[k] for k in sorted(all_times)]
    safe_times = pd.to_datetime([str(t) for t in sorted_times], errors="coerce")
    return pd.DatetimeIndex(safe_times), len(sorted_times)


@dataclass
class ZarrScanResult:
    """Result of scanning an existing zarr for missing data."""
    
    exists: bool
    experiments: List[str]
    members: List[str]
    max_time: int
    processed: Dict[Tuple[str, str], bool]  # (exp, member) -> has_data
    
    def is_processed(self, exp: str, member: str) -> bool:
        """Check if experiment/member combination has already been processed."""
        return self.processed.get((exp, member), False)


def scan_existing_zarr(output_path: Path) -> Optional[ZarrScanResult]:
    """
    Scan an existing zarr store to determine what data is already present.
    
    Returns None if the zarr doesn't exist or is invalid.
    """
    if not output_path.exists():
        logger.info(f"Zarr store does not exist: {output_path}")
        return None
    
    try:
        store = zarr.open(str(output_path), mode="r")
    except Exception as e:
        logger.warning(f"Could not open existing zarr: {e}")
        return None
    
    # Check required arrays exist
    required = ["experiment", "member", "time", "time_valid_length"]
    for arr_name in required:
        if arr_name not in store:
            logger.warning(f"Existing zarr missing required array: {arr_name}")
            return None
    
    experiments = list(store["experiment"][:])
    members = list(store["member"][:])
    time_shape = store["time"].shape
    max_time = time_shape[2] if len(time_shape) >= 3 else 0
    
    # Scan which experiment/member combinations have data
    processed: Dict[Tuple[str, str], bool] = {}
    valid_lengths = store["time_valid_length"][:]
    
    for exp_idx, exp in enumerate(experiments):
        for mem_idx, member in enumerate(members):
            valid_len = int(valid_lengths[exp_idx, mem_idx])
            has_data = valid_len > 0
            processed[(exp, member)] = has_data
            if has_data:
                logger.debug(f"  Found existing data: {exp}/{member} ({valid_len} timesteps)")
    
    n_processed = sum(1 for v in processed.values() if v)
    logger.info(f"Scanned existing zarr: {len(experiments)} experiments, "
                f"{len(members)} members, {n_processed}/{len(processed)} have data")
    
    return ZarrScanResult(
        exists=True,
        experiments=experiments,
        members=members,
        max_time=max_time,
        processed=processed,
    )


def expand_zarr_if_needed(
    store: zarr.Group,
    new_experiments: List[str],
    new_members: List[str],
    new_max_time: int,
    compressor=None,
) -> Tuple[zarr.Group, List[str], List[str], Dict[str, int], Dict[str, int]]:
    """
    Expand an existing zarr store if new experiments/members are requested.
    
    Returns the updated store, full experiment list, full member list, and 
    mappings from experiment/member names to indices.
    """
    if compressor is None:
        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)
    
    existing_exp = list(store["experiment"][:])
    existing_mem = list(store["member"][:])
    existing_max_time = store["time"].shape[2]
    
    # Merge lists preserving order
    all_experiments = list(existing_exp)
    for exp in new_experiments:
        if exp not in all_experiments:
            all_experiments.append(exp)
    
    all_members = list(existing_mem)
    for mem in new_members:
        if mem not in all_members:
            all_members.append(mem)
    
    # Sort members by ensemble number
    all_members = sorted(all_members, key=lambda x: int(x.split("i")[0][1:]))
    
    final_max_time = max(existing_max_time, new_max_time)
    
    needs_resize = (
        len(all_experiments) > len(existing_exp) or
        len(all_members) > len(existing_mem) or
        final_max_time > existing_max_time
    )
    
    if needs_resize:
        logger.info("Expanding zarr dimensions:")
        logger.info(f"  Experiments: {len(existing_exp)} -> {len(all_experiments)}")
        logger.info(f"  Members: {len(existing_mem)} -> {len(all_members)}")
        logger.info(f"  Max time: {existing_max_time} -> {final_max_time}")
        
        # This is complex - for now, we only support adding to existing dimensions
        # not expanding them. User should pre-allocate enough space.
        if len(all_experiments) > len(existing_exp) or len(all_members) > len(existing_mem):
            raise ValueError(
                "Expanding experiments/members in existing zarr is not yet supported. "
                "Please create a new zarr with all desired experiments/members, "
                "or use existing experiments/members only."
            )
        
        if final_max_time > existing_max_time:
            logger.warning(
                f"New data has more timesteps ({final_max_time}) than existing zarr "
                f"({existing_max_time}). Extra timesteps will be truncated."
            )
            final_max_time = existing_max_time
    
    exp_to_idx = {exp: i for i, exp in enumerate(all_experiments)}
    mem_to_idx = {mem: i for i, mem in enumerate(all_members)}
    
    return store, all_experiments, all_members, exp_to_idx, mem_to_idx


def initialize_structured_zarr(
    output_path: Path,
    experiments: List[str],
    members: List[str],
    max_time: int,
    compressor=None,
) -> zarr.Group:
    """Create the structured zarr store with proper dimensions."""
    if compressor is None:
        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)

    logger.info(f"Initializing structured zarr at {output_path}")
    logger.info(f"  Experiments: {experiments}")
    logger.info(f"  Members: {len(members)}")
    logger.info(f"  Max time steps: {max_time}")
    logger.info(f"  Spatial pixels: {NPIX}")
    logger.info(f"  Depth levels: {N_DEPTH}")

    store = zarr.open(str(output_path), mode="w", zarr_version=2)

    n_exp = len(experiments)
    n_mem = len(members)

    exp_data = np.array(experiments, dtype="U32")
    exp_arr = store.create_array("experiment", data=exp_data)
    exp_arr.attrs["description"] = "CMIP6 experiment names"

    mem_data = np.array(members, dtype="U16")
    mem_arr = store.create_array("member", data=mem_data)
    mem_arr.attrs["description"] = "Ensemble member IDs (rXiYpZfW format)"

    depth_vals = np.concatenate([TARGET_DEPTHS, [DEEP_LEVEL_VALUE]]).astype(np.float32)
    depth_arr = store.create_array("depth", data=depth_vals)
    depth_arr.attrs["units"] = "m"
    depth_arr.attrs["description"] = (
        "Depth bin centers in meters. 9999 = mean of depths >2000m."
    )

    time_arr = store.create_array(
        "time",
        shape=(n_exp, n_mem, max_time),
        chunks=(1, 1, max_time),
        dtype="int64",
        compressor=compressor,
        fill_value=-1,
    )
    time_arr.attrs["description"] = "Time in minutes since 1900-01-01"
    time_arr.attrs["units"] = "minutes since 1900-1-1 0:0:0"
    time_arr.attrs["calendar"] = "standard"

    valid_len_arr = store.create_array(
        "time_valid_length",
        shape=(n_exp, n_mem),
        dtype="int32",
        fill_value=0,
    )
    valid_len_arr.attrs["description"] = (
        "Number of valid timesteps per experiment/member"
    )

    for var in VARIABLES_2D:
        var_arr = store.create_array(
            var,
            shape=(n_exp, n_mem, max_time, NPIX),
            chunks=(1, 1, TIME_CHUNK, NPIX),
            dtype="float32",
            compressor=compressor,
            fill_value=np.nan,
        )
        var_arr.attrs["description"] = f"{var} data"
        var_arr.attrs["dimensions"] = ["experiment", "member", "time", "space"]

    for var in VARIABLES_3D:
        var_arr = store.create_array(
            var,
            shape=(n_exp, n_mem, max_time, N_DEPTH, NPIX),
            chunks=(1, 1, TIME_CHUNK, N_DEPTH, NPIX),
            dtype="float32",
            compressor=compressor,
            fill_value=np.nan,
        )
        var_arr.attrs["description"] = f"{var} data"
        var_arr.attrs["dimensions"] = ["experiment", "member", "time", "depth", "space"]

    exp_metadata = {}
    for exp in experiments:
        if exp in EXPERIMENT_METADATA:
            info = EXPERIMENT_METADATA[exp]
            exp_metadata[exp] = {
                "description": info.description,
                "forcing": info.forcing,
                "start_year": info.start_year,
                "end_year": info.end_year,
            }
        else:
            exp_metadata[exp] = {"description": f"Unknown experiment: {exp}"}

    store.attrs["experiment_metadata"] = json.dumps(exp_metadata)
    store.attrs["structure"] = "MIROC6 climate model with experiment/member dimensions"
    store.attrs["spatial_grid"] = "HEALPix level 6 (NEST ordering)"
    store.attrs["spatial_npix"] = NPIX
    store.attrs["hpx_level"] = HPX_LEVEL
    store.attrs["depth_levels"] = depth_vals.tolist()
    store.attrs["depth_deep_mean_threshold_m"] = 2000.0

    return store


def _store_member_data(
    store: zarr.Group,
    var: str,
    exp_idx: int,
    mem_idx: int,
    sorted_keys: List[Tuple[int, int]],
    data_by_time: Dict[Tuple[int, int], np.ndarray],
) -> None:
    n_times = len(sorted_keys)
    if var in VARIABLES_2D:
        var_array = np.full((n_times, NPIX), np.nan, dtype=np.float32)
    else:
        var_array = np.full((n_times, N_DEPTH, NPIX), np.nan, dtype=np.float32)

    for i, key in enumerate(sorted_keys):
        if key in data_by_time:
            var_array[i] = data_by_time[key]

    if var in VARIABLES_2D:
        store[var][exp_idx, mem_idx, :n_times, :] = var_array
    else:
        store[var][exp_idx, mem_idx, :n_times, :, :] = var_array


def _build_channel_names(depths: np.ndarray) -> List[str]:
    names: List[str] = []
    for var in VARIABLES_3D:
        for depth in depths:
            names.append(f"{var}{int(round(depth))}m")
    for var in VARIABLES_2D:
        names.append(var)
    return names


def _minutes_to_datetime(minutes: int) -> datetime.datetime:
    base = datetime.datetime(1900, 1, 1)
    return base + datetime.timedelta(minutes=int(minutes))


def compute_and_store_climatology(
    store: zarr.Group,
    exp_name: str,
    n_members: int,
    start_year: int,
    end_year: int,
) -> None:
    if exp_name not in store["experiment"][:]:
        logger.warning(f"Experiment {exp_name} not found; skipping climatology.")
        return

    exp_idx = list(store["experiment"][:]).index(exp_name)
    all_members = list(store["member"][:])
    members_to_use = all_members[:n_members]

    depths = np.asarray(store["depth"][:], dtype=np.float32)
    channel_names = _build_channel_names(depths)
    n_channels = len(channel_names)
    n_space = store[VARIABLES_2D[0]].shape[-1]

    sum_arr = np.zeros((12, n_channels, n_space), dtype=np.float64)
    sumsq_arr = np.zeros((12, n_channels, n_space), dtype=np.float64)
    count_arr = np.zeros((12, n_channels, n_space), dtype=np.int64)

    for mem in members_to_use:
        mem_idx = all_members.index(mem)
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

            channels: List[np.ndarray] = []
            for var in VARIABLES_3D:
                var_data = store[var][exp_idx, mem_idx, t_idx, :, :]  # (depth, space)
                for d_idx in range(var_data.shape[0]):
                    channels.append(var_data[d_idx])
            for var in VARIABLES_2D:
                channels.append(store[var][exp_idx, mem_idx, t_idx, :])

            stacked = np.stack(channels, axis=0).astype(np.float64)
            valid = np.isfinite(stacked)
            sum_arr[month_idx] += np.where(valid, stacked, 0.0)
            sumsq_arr[month_idx] += np.where(valid, stacked * stacked, 0.0)
            count_arr[month_idx] += valid.astype(np.int64)

    mean = np.divide(
        sum_arr, np.maximum(count_arr, 1), out=np.zeros_like(sum_arr), where=count_arr > 0
    )
    var = np.divide(
        sumsq_arr, np.maximum(count_arr, 1), out=np.zeros_like(sumsq_arr), where=count_arr > 0
    ) - mean * mean
    std = np.sqrt(np.maximum(var, 0.0))

    if "climatology" in store:
        del store["climatology"]
    clim_group = store.create_group("climatology")
    clim_group.create_array("mean", data=mean.astype(np.float32))
    clim_group.create_array("std", data=std.astype(np.float32))
    clim_group.attrs["channel_names"] = channel_names
    clim_group.attrs["depths"] = depths.tolist()
    clim_group.attrs["variables_2d"] = VARIABLES_2D
    clim_group.attrs["variables_3d"] = VARIABLES_3D
    clim_group.attrs["experiment"] = exp_name
    clim_group.attrs["members_used"] = members_to_use
    clim_group.attrs["start_year"] = start_year
    clim_group.attrs["end_year"] = end_year
    clim_group.attrs["monthly"] = True

    logger.info(
        f"Stored climatology for {exp_name} using {len(members_to_use)} members "
        f"({start_year}-{end_year}) in group 'climatology'."
    )


def process_member(
    input_dir: Path,
    experiment: str,
    member: str,
    exp_idx: int,
    mem_idx: int,
    store: zarr.Group,
    grid_file: str,
    regrid_dir: Path,
) -> int:
    """Process a single member and write to zarr. Returns number of timesteps."""
    logger.info(f"Processing {experiment}/{member} (exp_idx={exp_idx}, mem_idx={mem_idx})")

    member_dir = input_dir / experiment / member
    all_data: Dict[str, Dict[Tuple[int, int], np.ndarray]] = {
        var: {} for var in ALL_VARIABLES
    }
    all_times: Dict[Tuple[int, int], object] = {}

    for var in ALL_VARIABLES:
        var_dir = member_dir / var
        if not var_dir.exists():
            logger.warning(f"Variable directory {var_dir} does not exist")
            continue

        nc_files = sorted(var_dir.glob("*.nc"))
        for nc_file in nc_files:
            regrid_file = regrid_dir / f"{experiment}_{member}_{var}_{nc_file.name}"
            if not regrid_file.exists():
                try:
                    regrid_to_hpx6(str(nc_file), str(regrid_file), var, grid_file)
                except Exception as e:
                    logger.error(f"Failed to regrid {nc_file}: {e}")
                    continue

            try:
                with xr.open_dataset(str(regrid_file), use_cftime=True) as ds:
                    file_times = ds.time.values
                    da = ds[var]
                    if var in VARIABLES_3D:
                        da = _interpolate_depth(da)
                    da = _transpose_time_depth_space(da)
                    var_data = da.values

                    for i, t in enumerate(file_times):
                        key = _time_key(t)
                        all_data[var][key] = var_data[i].astype(np.float32)
                        all_times[key] = t
            except Exception as e:
                logger.error(f"Failed to read {regrid_file}: {e}")
                continue

    if not all_times:
        logger.warning(f"No valid data for {experiment}/{member}")
        return 0

    sorted_keys = sorted(all_times)
    sorted_times = [all_times[k] for k in sorted_keys]
    n_times = len(sorted_times)
    logger.info(f"  Found {n_times} timesteps: {sorted_times[0]} to {sorted_times[-1]}")

    time_minutes = np.array([_time_to_minutes(t) for t in sorted_times], dtype=np.int64)
    store["time"][exp_idx, mem_idx, :n_times] = time_minutes
    store["time_valid_length"][exp_idx, mem_idx] = n_times

    for var in ALL_VARIABLES:
        if all_data[var]:
            _store_member_data(store, var, exp_idx, mem_idx, sorted_keys, all_data[var])
            logger.info(f"  Written {var}: {n_times} timesteps")

    return n_times


def create_miroc6_profiles_structured(
    input_dir: str,
    output_zarr: str,
    experiments: List[str],
    n_members: Optional[int] = None,
    specific_members: Optional[List[str]] = None,
    regrid_dir: Optional[str] = None,
    clim_start_year: int = 1990,
    clim_end_year: int = 2014,
    clim_n_members: int = 5,
    compute_climatology: bool = False,
    resume: bool = False,
) -> None:
    """
    Main function to create structured MIROC6 dataset with profile variables.
    
    Args:
        resume: If True, scan existing zarr and only process missing data.
                If False, create zarr from scratch (will fail if it exists).
    """
    input_path = Path(input_dir)
    output_path = Path(output_zarr)

    if not check_cdo():
        raise RuntimeError("CDO is required but not available")

    if regrid_dir is None:
        temp_ctx = tempfile.TemporaryDirectory()
        regrid_dir = Path(temp_ctx.name)
    else:
        temp_ctx = None
        regrid_dir = Path(regrid_dir)
        regrid_dir.mkdir(parents=True, exist_ok=True)

    grid_file = str(regrid_dir / "hp6.grid")
    if not os.path.exists(grid_file):
        create_grid_file(HPX_LEVEL, grid_file)

    # Scan existing zarr if resuming
    scan_result: Optional[ZarrScanResult] = None
    if resume:
        scan_result = scan_existing_zarr(output_path)
        if scan_result is None:
            logger.warning("Resume requested but no valid existing zarr found. Creating new.")
            resume = False

    all_available_members = set()
    for exp in experiments:
        members = discover_members(input_path, exp)
        all_available_members.update(members)
        logger.info(f"Experiment {exp}: found {len(members)} members")

    if specific_members:
        members_to_use = specific_members
    else:
        members_to_use = sorted(
            all_available_members, key=lambda x: int(x.split("i")[0][1:])
        )
        if n_members is not None:
            members_to_use = members_to_use[:n_members]

    logger.info(f"Will process {len(members_to_use)} members: {members_to_use[:5]}...")

    # Handle resume vs create-new logic
    if resume and scan_result is not None:
        # Use existing zarr structure
        logger.info("Resuming from existing zarr store...")
        store = zarr.open(str(output_path), mode="r+")
        
        # Verify requested experiments/members are in the zarr
        existing_experiments = list(store["experiment"][:])
        existing_members = list(store["member"][:])
        
        # Filter to only experiments/members that exist in the zarr
        experiments_to_process = [e for e in experiments if e in existing_experiments]
        members_to_process = [m for m in members_to_use if m in existing_members]
        
        if not experiments_to_process:
            raise ValueError(
                f"None of requested experiments {experiments} exist in zarr. "
                f"Existing experiments: {existing_experiments}"
            )
        if not members_to_process:
            raise ValueError(
                f"None of requested members exist in zarr. "
                f"Existing members: {existing_members}"
            )
        
        missing_exp = set(experiments) - set(experiments_to_process)
        missing_mem = set(members_to_use) - set(members_to_process)
        if missing_exp:
            logger.warning(f"Experiments not in existing zarr (skipped): {missing_exp}")
        if missing_mem:
            logger.warning(f"Members not in existing zarr (skipped): {missing_mem}")
        
        experiments = experiments_to_process
        members_to_use = members_to_process
        max_time = scan_result.max_time
        
        # Build index mappings
        exp_to_idx = {exp: i for i, exp in enumerate(existing_experiments)}
        mem_to_idx = {mem: i for i, mem in enumerate(existing_members)}
        
        # Count how many need processing
        to_process = []
        for exp in experiments:
            for member in members_to_use:
                if not scan_result.is_processed(exp, member):
                    to_process.append((exp, member))
        
        logger.info(f"Found {len(to_process)} experiment/member combinations to process")
        if not to_process:
            logger.info("All requested data already exists in zarr. Nothing to do.")
            if temp_ctx is not None:
                temp_ctx.cleanup()
            return
        
    else:
        # Create new zarr from scratch
        if output_path.exists() and not resume:
            raise FileExistsError(
                f"Output zarr already exists: {output_path}. "
                f"Use --resume to add missing data, or delete the existing zarr."
            )

    logger.info("Scanning all members for time ranges...")
    max_time = 0
    for exp in experiments:
        for member in members_to_use:
            _, n_times = get_member_time_range(input_path, exp, member)
            max_time = max(max_time, n_times)

    logger.info(f"Maximum time steps across all members: {max_time}")

    store = initialize_structured_zarr(
        output_path, experiments, members_to_use, max_time
    )

    # All combinations need processing
    exp_to_idx = {exp: i for i, exp in enumerate(experiments)}
    mem_to_idx = {mem: i for i, mem in enumerate(members_to_use)}
    to_process = [(exp, member) for exp in experiments for member in members_to_use]
    scan_result = None

    total_processed = 0
    total_skipped = 0
    
    for exp_idx, exp in enumerate(experiments):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing experiment: {exp} ({exp_idx + 1}/{len(experiments)})")
        logger.info(f"{'='*60}")

        for member in members_to_use:
            # Check if already processed (resume mode)
            if scan_result is not None and scan_result.is_processed(exp, member):
                total_skipped += 1
                logger.info(f"  Skipping {member}: already has data")
                continue
            
            exp_store_idx = exp_to_idx[exp]
            mem_store_idx = mem_to_idx[member]
            
            try:
                n_times = process_member(
                    input_path,
                    exp,
                    member,
                    exp_store_idx,
                    mem_store_idx,
                    store,
                    grid_file,
                    regrid_dir,
                )
                if n_times > 0:
                    total_processed += 1
                    logger.info(
                        f"  Completed {member}: {n_times} timesteps "
                        f"(processed: {total_processed}, skipped: {total_skipped})"
                    )
            except Exception as e:
                logger.error(f"Failed to process {exp}/{member}: {e}")
                continue

    if compute_climatology:
        compute_and_store_climatology(
            store=store,
            exp_name="historical",
            n_members=clim_n_members,
            start_year=clim_start_year,
            end_year=clim_end_year,
        )

    logger.info("\nConsolidating zarr metadata...")
    zarr.consolidate_metadata(store.store)
    if temp_ctx is not None:
        temp_ctx.cleanup()

    logger.info("\n" + "=" * 60)
    logger.info("DATASET PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output: {output_zarr}")
    logger.info(f"Experiments: {experiments}")
    logger.info(f"Members: {len(members_to_use)}")
    logger.info(f"Max time steps: {max_time}")
    logger.info(f"Spatial pixels: {NPIX}")
    logger.info(f"Depth levels: {N_DEPTH}")
    logger.info(f"Total member series processed: {total_processed}")
    logger.info(f"Total member series skipped (already existed): {total_skipped}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CMIP monthly NetCDF → HEALPix profiles zarr"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Root dir with monthly NetCDFs (MIROC6 / CESM2)",
    )
    parser.add_argument(
        "--output_zarr",
        type=str,
        required=True,
        help="Path to output zarr store",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["historical"],
        help="Experiments to process (default: historical)",
    )
    parser.add_argument(
        "--n_members",
        type=int,
        default=None,
        help="Number of members to process (default: all)",
    )
    parser.add_argument(
        "--members",
        nargs="+",
        default=None,
        help="Specific member IDs to process (overrides --n_members)",
    )
    parser.add_argument(
        "--regrid_dir",
        type=str,
        default=None,
        help="Directory for regridded files (default: auto)",
    )
    parser.add_argument(
        "--clim_start_year",
        type=int,
        default=1990,
        help="Climatology start year (default: 1990)",
    )
    parser.add_argument(
        "--clim_end_year",
        type=int,
        default=2014,
        help="Climatology end year (default: 2014)",
    )
    parser.add_argument(
        "--clim_n_members",
        type=int,
        default=5,
        help="Number of historical members for climatology (default: 5)",
    )
    parser.add_argument(
        "--compute_climatology",
        action="store_true",
        help="Compute and store climatology (default: skip)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing zarr: scan for missing data and only process those",
    )

    args = parser.parse_args()

    create_miroc6_profiles_structured(
        input_dir=args.input_dir,
        output_zarr=args.output_zarr,
        experiments=args.experiments,
        n_members=args.n_members,
        specific_members=args.members,
        regrid_dir=args.regrid_dir,
        clim_start_year=args.clim_start_year,
        clim_end_year=args.clim_end_year,
        clim_n_members=args.clim_n_members,
        compute_climatology=args.compute_climatology,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
