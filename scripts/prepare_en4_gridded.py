#!/usr/bin/env python3
"""
EN4 → 5° monthly T/S zarr.

  climatology  — monthly clim from analysis NC (default 1993–2016)
  grid         — bin raw profiles on anomalies, add clim back
  all          — both

Empty bins stay NaN (no vertical fill).

  python scripts/prepare_en4_gridded.py climatology \\
    --analysis_nc en4_temp_sal_1993_onwards.nc --clim_out en4_climatology.nc

  python scripts/prepare_en4_gridded.py grid \\
    --profiles_dir en4_profiles --clim_file en4_climatology.nc \\
    --output_zarr en4_gridded_5deg_monthly.zarr
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
import zarr
from scipy.interpolate import RegularGridInterpolator

DEFAULT_DEPTH_EDGES = np.array(
    [0, 50, 112.5, 225, 400, 600, 850, 1250, 1750, 2000, 10000.0],
    dtype=float,
)
DEFAULT_LAT_RES = 5.0
DEFAULT_LON_RES = 5.0
DEFAULT_BASE_START = "1993-01-01"
DEFAULT_BASE_END = "2016-12-31"
EN4_FILL = 99999.0
POTM_ERROR_FLOOR = 0.02
PSAL_ERROR_FLOOR = 0.01
Z_FINE_STEP = 5.0
MAX_GAP = 400.0
MIN_COVERAGE = 0.001

TEMP_NAMES = [
    "temperature", "temp", "POTM", "potm", "thetao", "temperature_observation",
]
SALT_NAMES = [
    "salinity", "sal", "PSAL", "psal", "so", "salinity_observation",
]
DEPTH_NAMES = ["depth", "deptht", "lev", "z", "level"]


# ===========================================================================
# Shared helpers
# ===========================================================================
def pick_name(ds, candidates, kind):
    for c in candidates:
        if c in ds.variables or c in ds.dims or c in ds.coords:
            return c
    raise KeyError(
        f"Could not find {kind}. Tried: {candidates}. Available: {list(ds.variables)}"
    )


def wrap_lon(lon):
    lon = np.asarray(lon, dtype=np.float64)
    return ((lon + 180.0) % 360.0) - 180.0


def sanity_check_temperature(T_da):
    tmax = float(T_da.max())
    tmin = float(T_da.min())
    note = ""
    if tmax > 100.0:
        T_da = T_da - 273.15
        note = f"K→°C conversion applied (original max={tmax:.2f})"
        print(f"  [T] Kelvin detected → subtracting 273.15 (was {tmin:.2f}..{tmax:.2f} K)")
    elif tmin < -10.0:
        note = f"WARNING: suspicious low T (min={tmin:.2f} °C)"
        print(f"  [T] {note}")
    else:
        note = f"°C range {tmin:.2f}..{tmax:.2f} looks fine"
        print(f"  [T] {note}")
    return T_da, note


def sanity_check_salinity(S_da):
    smax = float(S_da.max())
    smin = float(S_da.min())
    if smax > 100.0:
        note = f"WARNING: salinity max={smax:.2f} — possibly density (kg/m³). Not converted."
        print(f"  [S] {note}")
    elif smin < 0 or smax > 50:
        note = f"WARNING: salinity outside [0, 50] PSU ({smin:.2f}..{smax:.2f})"
        print(f"  [S] {note}")
    else:
        note = f"PSU range {smin:.2f}..{smax:.2f} looks fine"
        print(f"  [S] {note}")
    return S_da, note


# ===========================================================================
# Stage 1 — climatology
# ===========================================================================
def build_climatology(
    analysis_nc: str,
    clim_out: str,
    base_start: str = DEFAULT_BASE_START,
    base_end: str = DEFAULT_BASE_END,
) -> Path:
    """Write en4_climatology.nc from analysis file."""
    print(f"Opening {analysis_nc}")
    ds = xr.open_dataset(analysis_nc, chunks="auto")

    tname = pick_name(ds, TEMP_NAMES, "temperature")
    sname = pick_name(ds, SALT_NAMES, "salinity")
    zname = pick_name(ds, DEPTH_NAMES, "depth")

    lat_name, lon_name, t_name = "lat", "lon", "time"
    for alt in ("lat", "latitude", "LATITUDE"):
        if alt in ds.coords or alt in ds.dims:
            lat_name = alt
            break
    for alt in ("lon", "longitude", "LONGITUDE"):
        if alt in ds.coords or alt in ds.dims:
            lon_name = alt
            break

    print(f"  temperature : {tname}")
    print(f"  salinity    : {sname}")
    print(f"  depth       : {zname}  ({ds.sizes[zname]} levels)")
    print(f"  lat/lon     : {lat_name} / {lon_name}")
    print(f"  time span   : {ds[t_name].values[0]} → {ds[t_name].values[-1]}")

    t0 = np.datetime64(base_start)
    t1 = np.datetime64(base_end)
    sub = ds.sel({t_name: slice(t0, t1)})
    n_months = sub.sizes[t_name]
    print(
        f"Baseline: {base_start} … {base_end}  "
        f"({n_months} monthly fields, {n_months / 12:.1f} yr)"
    )

    print("Computing monthly climatology on native levels…")
    T_clim = sub[tname].groupby(f"{t_name}.month").mean(t_name, skipna=True)
    S_clim = sub[sname].groupby(f"{t_name}.month").mean(t_name, skipna=True)
    T_std = sub[tname].groupby(f"{t_name}.month").std(t_name, skipna=True)
    S_std = sub[sname].groupby(f"{t_name}.month").std(t_name, skipna=True)

    rename_map = {}
    if lat_name != "lat":
        rename_map[lat_name] = "lat"
    if lon_name != "lon":
        rename_map[lon_name] = "lon"
    if zname != "depth":
        rename_map[zname] = "depth"
    if rename_map:
        T_clim = T_clim.rename(rename_map)
        S_clim = S_clim.rename(rename_map)
        T_std = T_std.rename(rename_map)
        S_std = S_std.rename(rename_map)

    lon_vals = T_clim["lon"].values
    if lon_vals.max() > 180.0:
        new_lon = wrap_lon(lon_vals)
        order = np.argsort(new_lon)
        T_clim = T_clim.assign_coords(lon=new_lon).isel(lon=order)
        S_clim = S_clim.assign_coords(lon=new_lon).isel(lon=order)
        T_std = T_std.assign_coords(lon=new_lon).isel(lon=order)
        S_std = S_std.assign_coords(lon=new_lon).isel(lon=order)
        print("  wrapped longitudes to [-180, 180)")

    print("\nRunning unit sanity checks…")
    T_clim, T_note = sanity_check_temperature(T_clim)
    S_clim, S_note = sanity_check_salinity(S_clim)

    out = xr.Dataset(
        data_vars=dict(
            temperature_clim=T_clim.astype("float32"),
            salinity_clim=S_clim.astype("float32"),
            temperature_std=T_std.astype("float32"),
            salinity_std=S_std.astype("float32"),
        ),
        attrs=dict(
            title="EN4 analysis monthly climatology (native levels)",
            source=f"EN4 Met Office gap-filled analysis: {analysis_nc}",
            baseline=f"{base_start} .. {base_end}",
            n_months=int(n_months),
            n_years=float(n_months / 12.0),
            T_units_note=T_note,
            S_units_note=S_note,
            T_units="degC",
            S_units="PSU",
            purpose=(
                "Used to compute level-wise anomalies from raw EN4 "
                "profiles before depth-binning."
            ),
        ),
    )
    out["temperature_clim"].attrs = dict(
        units="degC", long_name="Monthly climatological potential temperature"
    )
    out["salinity_clim"].attrs = dict(
        units="PSU", long_name="Monthly climatological practical salinity"
    )
    out["temperature_std"].attrs = dict(
        units="degC", long_name="Interannual std of monthly T"
    )
    out["salinity_std"].attrs = dict(
        units="PSU", long_name="Interannual std of monthly S"
    )

    enc = {
        v: {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)}
        for v in out.data_vars
    }
    clim_path = Path(clim_out)
    clim_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting {clim_path}")
    out.to_netcdf(clim_path, encoding=enc)

    print("\n" + "=" * 60)
    print("CLIMATOLOGY SUMMARY")
    print("=" * 60)
    print(f"  dims  : {dict(out.sizes)}")
    print(
        f"  depth : {out['depth'].values[0]:.1f} … "
        f"{out['depth'].values[-1]:.1f} m  ({out.sizes['depth']} levels)"
    )
    print(
        f"  T range: {float(out['temperature_clim'].min()):.2f} … "
        f"{float(out['temperature_clim'].max()):.2f} °C"
    )
    print(
        f"  S range: {float(out['salinity_clim'].min()):.2f} … "
        f"{float(out['salinity_clim'].max()):.2f} PSU"
    )
    print("=" * 60)
    return clim_path


# ===========================================================================
# Stage 2 — profile gridding helpers + main
# ===========================================================================
def safe_digitize(values, edges):
    idx = np.digitize(values, edges) - 1
    ok = (idx >= 0) & (idx < (len(edges) - 1))
    return idx, ok


def month_id_from_juld(juld):
    base = np.datetime64("1950-01-01", "D")
    dt = base + juld.astype("timedelta64[D]")
    y = dt.astype("datetime64[Y]").astype(int) + 1970
    m = (dt.astype("datetime64[M]") - dt.astype("datetime64[Y]")).astype(int) + 1
    return y * 12 + (m - 1)


def month_of_year_from_juld(juld):
    base = np.datetime64("1950-01-01", "D")
    dt = base + juld.astype("timedelta64[D]")
    m = (dt.astype("datetime64[M]") - dt.astype("datetime64[Y]")).astype(int) + 1
    return m.astype(np.int8)


def to_str_array(x):
    a = np.asarray(x)
    if a.dtype.kind == "S":
        if a.ndim == 2:
            return np.array([b"".join(row).decode("ascii", "ignore") for row in a])
        return np.array([s.decode("ascii", "ignore") for s in a])
    if a.dtype.kind == "U":
        if a.ndim == 2:
            return np.array(["".join(row) for row in a])
        return a.astype(str)
    return a.astype(str)


def qc_keep_profile(qc_strings_1d):
    out = np.ones(len(qc_strings_1d), dtype=bool)
    for i, s in enumerate(qc_strings_1d):
        s = s.strip()
        if len(s) == 0:
            continue
        for c in s:
            if c in ("3", "4", "9"):
                out[i] = False
                break
    return out


def qc_keep_levels(qc_strings_1d, n_lev):
    n_prof = len(qc_strings_1d)
    out = np.zeros((n_prof, n_lev), dtype=bool)
    for i, s in enumerate(qc_strings_1d):
        L = min(len(s), n_lev)
        for j in range(L):
            out[i, j] = s[j] in ("0", "1", "2")
    return out


def mask_fill(values, fill=EN4_FILL, atol=1.0):
    return np.isfinite(values) & (np.abs(values) < fill - atol)


def grid_en4_profiles(
    profiles_dir: str,
    clim_file: str,
    accum_zarr: str,
    output_zarr: str,
    pattern: str = "*.nc",
    lat_res: float = DEFAULT_LAT_RES,
    lon_res: float = DEFAULT_LON_RES,
    depth_edges: Optional[np.ndarray] = None,
    min_coverage: float = MIN_COVERAGE,
    max_gap: float = MAX_GAP,
    z_fine_step: float = Z_FINE_STEP,
) -> Path:
    """Bin EN4 profiles → 5° zarr."""
    depth_edges = np.asarray(
        depth_edges if depth_edges is not None else DEFAULT_DEPTH_EDGES, dtype=float
    )

    files = sorted(glob.glob(os.path.join(profiles_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No input files in {profiles_dir!r} matching {pattern!r}")

    lat_edges = np.arange(-90.0, 90.0 + lat_res, lat_res)
    lon_edges = np.arange(-180.0, 180.0 + lon_res, lon_res)
    lat = ((lat_edges[:-1] + lat_edges[1:]) * 0.5).astype(np.float32)
    lon = ((lon_edges[:-1] + lon_edges[1:]) * 0.5).astype(np.float32)
    depth = ((depth_edges[:-1] + depth_edges[1:]) * 0.5).astype(np.float32)

    NY, NX, NZ = len(lat), len(lon), len(depth)
    STRIDE_Y = NX * NZ
    STRIDE_X = NZ
    FLAT_LEN = NY * NX * NZ

    print(f"n_files: {len(files)}  grid: ({NY}, {NX}, {NZ})")
    print(f"depth bins: {depth_edges.tolist()}")
    print(f"MIN_COVERAGE={min_coverage}  MAX_GAP={max_gap}m  (no vertical extrapolation)")

    print(f"Loading climatology: {clim_file}")
    cl = xr.open_dataset(clim_file).sortby("lat").sortby("depth").sortby("lon")

    T_max = float(cl["temperature_clim"].max())
    T_min = float(cl["temperature_clim"].min())
    if T_max > 100.0:
        print(f"  [!] temperature_clim in KELVIN (max={T_max:.2f}). Converting to °C.")
        cl["temperature_clim"] = cl["temperature_clim"] - 273.15
    elif T_min < -10.0:
        print(f"  [!] WARNING: temperature_clim min={T_min:.2f} °C looks suspicious")
    else:
        print(f"  [OK] temperature_clim range: {T_min:.2f} … {T_max:.2f} °C")

    S_max = float(cl["salinity_clim"].max())
    S_min = float(cl["salinity_clim"].min())
    if S_max > 100.0:
        raise RuntimeError(
            f"salinity_clim max={S_max:.2f} — looks like density (kg/m³). Aborting."
        )
    elif S_min < 0 or S_max > 50:
        print(f"  [!] WARNING: salinity_clim range {S_min:.2f}..{S_max:.2f} PSU")
    else:
        print(f"  [OK] salinity_clim range: {S_min:.2f} … {S_max:.2f} PSU")

    CLIM_MONTH = cl["month"].values.astype(int)
    CLIM_DEPTH = cl["depth"].values.astype(float)
    CLIM_LAT = cl["lat"].values.astype(float)
    CLIM_LON = cl["lon"].values.astype(float)
    T_CLIM = (
        cl["temperature_clim"].transpose("month", "depth", "lat", "lon").values.astype(np.float32)
    )
    S_CLIM = (
        cl["salinity_clim"].transpose("month", "depth", "lat", "lon").values.astype(np.float32)
    )

    def build_level_interp(cube_4d):
        interps = {}
        for mi, m in enumerate(CLIM_MONTH):
            interps[int(m)] = RegularGridInterpolator(
                (CLIM_DEPTH, CLIM_LAT, CLIM_LON),
                cube_4d[mi],
                method="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
        lon_min, lon_max = CLIM_LON[0], CLIM_LON[-1]

        def interp(month_int, z, la, lo):
            lo = np.asarray(lo, dtype=np.float64).copy()
            lo = np.where(lo < lon_min, lo + 360.0, lo)
            lo = np.where(lo > lon_max, lo - 360.0, lo)
            la = np.clip(np.asarray(la, dtype=np.float64), CLIM_LAT[0], CLIM_LAT[-1])
            z = np.clip(np.asarray(z, dtype=np.float64), CLIM_DEPTH[0], CLIM_DEPTH[-1])
            pts = np.column_stack([z.ravel(), la.ravel(), lo.ravel()])
            return interps[int(month_int)](pts).reshape(z.shape)

        return interp

    T_interp = build_level_interp(T_CLIM)
    S_interp = build_level_interp(S_CLIM)

    def clim_for_profile_levels(var_interp, depths_2d, lat_1d, lon_1d, month_1d):
        out = np.full(depths_2d.shape, np.nan, dtype=np.float32)
        for m in np.unique(month_1d):
            sel = np.where(month_1d == m)[0]
            if sel.size == 0:
                continue
            d = depths_2d[sel, :]
            la = np.broadcast_to(lat_1d[sel][:, None], d.shape)
            lo = np.broadcast_to(lon_1d[sel][:, None], d.shape)
            out[sel, :] = var_interp(int(m), d, la, lo).astype(np.float32)
        return out

    print("Integrating climatology from native levels → target bins…")

    def bin_integrate_clim(var_interp):
        z_min, z_max = depth_edges[0], depth_edges[-1]
        z_top_cap = min(z_max, float(CLIM_DEPTH.max()))
        z_fine = np.arange(
            z_min + z_fine_step / 2.0, z_top_cap, z_fine_step, dtype=np.float64
        )
        fine_to_bin = np.digitize(z_fine, depth_edges) - 1
        keep = (fine_to_bin >= 0) & (fine_to_bin < NZ)
        z_fine, fine_to_bin = z_fine[keep], fine_to_bin[keep]

        out = np.full((12, NY, NX, NZ), np.nan, dtype=np.float32)
        LA2, LO2 = np.meshgrid(lat.astype(np.float64), lon.astype(np.float64), indexing="ij")
        LA3 = np.broadcast_to(LA2[:, :, None], (NY, NX, z_fine.size))
        LO3 = np.broadcast_to(LO2[:, :, None], (NY, NX, z_fine.size))
        Z3 = np.broadcast_to(z_fine[None, None, :], (NY, NX, z_fine.size))

        for mi in range(12):
            m = mi + 1
            vals_fine = var_interp(m, Z3, LA3, LO3)
            for k in range(NZ):
                sel = fine_to_bin == k
                if not np.any(sel):
                    continue
                with np.errstate(invalid="ignore"):
                    out[mi, :, :, k] = np.nanmean(vals_fine[:, :, sel], axis=2).astype(np.float32)
        return out

    T_CLIM_BIN = bin_integrate_clim(T_interp)
    S_CLIM_BIN = bin_integrate_clim(S_interp)
    print(
        f"  T_CLIM_BIN shape: {T_CLIM_BIN.shape}  "
        f"T range: {np.nanmin(T_CLIM_BIN):.2f}..{np.nanmax(T_CLIM_BIN):.2f} °C"
    )
    print(
        f"  S_CLIM_BIN shape: {S_CLIM_BIN.shape}  "
        f"S range: {np.nanmin(S_CLIM_BIN):.2f}..{np.nanmax(S_CLIM_BIN):.2f} PSU"
    )

    def profile_bin_averages_depth_weighted(
        anomalies, errors, depths, ok_mask, depth_edges_local
    ):
        n_prof, n_lev = anomalies.shape
        NZ_local = len(depth_edges_local) - 1
        prof_bin_val = np.full((n_prof, NZ_local), np.nan, dtype=np.float64)
        prof_bin_err = np.full((n_prof, NZ_local), np.nan, dtype=np.float64)
        prof_bin_ok = np.zeros((n_prof, NZ_local), dtype=bool)
        prof_bin_cov = np.zeros((n_prof, NZ_local), dtype=np.float32)

        z_min, z_max = depth_edges_local[0], depth_edges_local[-1]
        z_fine = np.arange(
            z_min + z_fine_step / 2.0, min(z_max, 6000.0), z_fine_step, dtype=np.float64
        )
        fine_to_bin = np.digitize(z_fine, depth_edges_local) - 1
        fine_valid_bin = (fine_to_bin >= 0) & (fine_to_bin < NZ_local)
        bin_widths = np.diff(depth_edges_local).astype(np.float64)

        for p in range(n_prof):
            m = ok_mask[p]
            if not np.any(m):
                continue
            zp = depths[p, m].astype(np.float64)
            vp = anomalies[p, m].astype(np.float64)
            order = np.argsort(zp, kind="stable")
            zp, vp = zp[order], vp[order]
            zp_u, uidx = np.unique(zp, return_index=True)
            vp_u = vp[uidx]

            if errors is not None:
                ep = errors[p, m].astype(np.float64)[order][uidx]
                ep_ok = np.isfinite(ep) & (np.abs(ep) < EN4_FILL - 1)
            else:
                ep, ep_ok = None, None

            if len(zp_u) < 2:
                k = int(np.digitize([zp_u[0]], depth_edges_local)[0] - 1)
                if 0 <= k < NZ_local:
                    prof_bin_val[p, k] = vp_u[0]
                    prof_bin_ok[p, k] = True
                    prof_bin_cov[p, k] = 0.0
                    if ep is not None and ep_ok[0]:
                        prof_bin_err[p, k] = ep[0]
                continue

            in_range = (z_fine >= zp_u[0]) & (z_fine <= zp_u[-1]) & fine_valid_bin
            if not np.any(in_range):
                continue
            seg = np.searchsorted(zp_u, z_fine[in_range], side="right") - 1
            seg = np.clip(seg, 0, len(zp_u) - 2)
            seg_gap = zp_u[seg + 1] - zp_u[seg]
            keep_seg = seg_gap <= max_gap
            covered = np.zeros_like(in_range)
            idx_in = np.where(in_range)[0]
            covered[idx_in[keep_seg]] = True
            if not np.any(covered):
                continue

            zf = z_fine[covered]
            vf = np.interp(zf, zp_u, vp_u)
            bf = fine_to_bin[covered]
            s = np.bincount(bf, weights=vf, minlength=NZ_local)
            c = np.bincount(bf, minlength=NZ_local)
            has = c > 0
            if np.any(has):
                prof_bin_val[p, has] = s[has] / c[has]
                prof_bin_ok[p, has] = True
                prof_bin_cov[p, has] = (c[has] * z_fine_step) / bin_widths[has]

            if ep is not None and np.any(ep_ok):
                zp_e, ep_e = zp_u[ep_ok], ep[ep_ok]
                if len(zp_e) >= 2:
                    ef_sq = np.interp(zf, zp_e, ep_e ** 2, left=np.nan, right=np.nan)
                else:
                    ef_sq = np.full_like(vf, ep_e[0] ** 2)
                finite = np.isfinite(ef_sq)
                if np.any(finite):
                    se = np.bincount(bf[finite], weights=ef_sq[finite], minlength=NZ_local)
                    ce = np.bincount(bf[finite], minlength=NZ_local)
                    ok_e = ce > 0
                    with np.errstate(invalid="ignore"):
                        prof_bin_err[p, ok_e] = np.sqrt(se[ok_e] / ce[ok_e])

        return prof_bin_val, prof_bin_err, prof_bin_ok, prof_bin_cov

    def accumulate_profiles_to_grid_fast(prof_vals, prof_errs, prof_ok, y_prof, x_prof, ok_geo):
        n_prof = len(y_prof)
        ok = prof_ok & ok_geo[:, None]
        if not np.any(ok):
            z = np.zeros((NY, NX, NZ), dtype=np.float64)
            return z, z.copy(), np.zeros((NY, NX, NZ), dtype=np.int64), z.copy()

        yi_2d = y_prof[:, None] * np.ones(NZ, dtype=int)[None, :]
        xi_2d = x_prof[:, None] * np.ones(NZ, dtype=int)[None, :]
        zi_2d = np.arange(NZ)[None, :] * np.ones(n_prof, dtype=int)[:, None]
        flat = (yi_2d * STRIDE_Y + xi_2d * STRIDE_X + zi_2d).astype(np.int64)

        fl_ok = ok.ravel()
        idx = flat.ravel()[fl_ok]
        vv = prof_vals.ravel()[fl_ok]
        s = np.bincount(idx, weights=vv, minlength=FLAT_LEN).reshape(NY, NX, NZ)
        ss = np.bincount(idx, weights=vv * vv, minlength=FLAT_LEN).reshape(NY, NX, NZ)
        c = np.bincount(idx, minlength=FLAT_LEN).reshape(NY, NX, NZ)
        ee = prof_errs.ravel()[fl_ok]
        e_finite = np.isfinite(ee)
        ee_sq = np.where(e_finite, ee ** 2, 0.0)
        ess = np.bincount(idx, weights=ee_sq, minlength=FLAT_LEN).reshape(NY, NX, NZ)
        return s, ss, c, ess

    print("Scanning files for time axis…")
    all_months = []
    for fn in files:
        with xr.open_dataset(fn, decode_cf=False, mask_and_scale=False, engine="netcdf4") as ds:
            all_months.extend(month_id_from_juld(ds["JULD"].values).astype(np.int64).tolist())
    time_ids = np.unique(np.array(all_months, dtype=np.int64))
    time_ids.sort()
    NT = len(time_ids)
    id2t = {int(mid): i for i, mid in enumerate(time_ids)}
    times = np.array(
        [np.datetime64(f"{t // 12:04d}-{(t % 12) + 1:02d}-01") for t in time_ids],
        dtype="datetime64[ns]",
    )
    times_moy = np.array([int(t % 12) + 1 for t in time_ids], dtype=np.int8)
    print(f"months: {NT}  from {times[0]} to {times[-1]}")

    def init_store(zarr_path):
        ds0 = xr.Dataset(
            coords=dict(
                time=times,
                lat=lat,
                lon=lon,
                depth=depth,
                depth_upper=(("depth",), depth_edges[:-1].astype(np.float32)),
                depth_lower=(("depth",), depth_edges[1:].astype(np.float32)),
                file=np.arange(len(files), dtype=np.int32),
            ),
            data_vars=dict(
                file_done=("file", np.zeros(len(files), dtype=bool)),
                potm_sum=(("time", "lat", "lon", "depth"), np.zeros((NT, NY, NX, NZ), dtype=np.float32)),
                potm_sumsq=(("time", "lat", "lon", "depth"), np.zeros((NT, NY, NX, NZ), dtype=np.float32)),
                potm_count=(("time", "lat", "lon", "depth"), np.zeros((NT, NY, NX, NZ), dtype=np.int32)),
                potm_err_sumsq=(("time", "lat", "lon", "depth"), np.zeros((NT, NY, NX, NZ), dtype=np.float32)),
                psal_sum=(("time", "lat", "lon", "depth"), np.zeros((NT, NY, NX, NZ), dtype=np.float32)),
                psal_sumsq=(("time", "lat", "lon", "depth"), np.zeros((NT, NY, NX, NZ), dtype=np.float32)),
                psal_count=(("time", "lat", "lon", "depth"), np.zeros((NT, NY, NX, NZ), dtype=np.int32)),
                psal_err_sumsq=(("time", "lat", "lon", "depth"), np.zeros((NT, NY, NX, NZ), dtype=np.float32)),
            ),
            attrs=dict(
                title="EN4 gridded (raw, for downstream reconstruction)",
                depth_edges_m=",".join(map(str, depth_edges.tolist())),
                min_coverage=float(min_coverage),
                max_gap_m=float(max_gap),
                note="No vertical extrapolation — empty bins stay empty.",
            ),
        )
        enc = {
            v: {"chunks": (1, NY, NX, NZ)}
            for v in ds0.data_vars
            if v.endswith(("sum", "sumsq", "count", "err_sumsq"))
        }
        enc["file_done"] = {"chunks": (len(files),)}
        Path(zarr_path).parent.mkdir(parents=True, exist_ok=True)
        ds0.to_zarr(zarr_path, mode="w", encoding=enc)
        return zarr.open_group(zarr_path, mode="a")

    zg = init_store(accum_zarr) if not os.path.exists(accum_zarr) else zarr.open_group(accum_zarr, mode="a")
    print(f"Zarr store: {accum_zarr}")
    print(f"file_done: {int(zg['file_done'][:].sum())} / {len(files)}")

    A = {
        k: zg[k]
        for k in [
            "potm_sum", "potm_sumsq", "potm_count", "potm_err_sumsq",
            "psal_sum", "psal_sumsq", "psal_count", "psal_err_sumsq",
            "file_done",
        ]
    }
    file_done = A["file_done"][:]
    todo_files = np.where(~file_done)[0]
    print(f"Files remaining: {len(todo_files)}")

    def read_var_safe(ds, name):
        return ds[name].values if name in ds else None

    def write_back(key, t, slab_add, dtype):
        slab = A[key][t].astype(np.float64 if dtype is np.float32 else np.int64)
        slab += slab_add
        A[key][t] = slab.astype(dtype)

    for ii in todo_files:
        fn = files[ii]
        with xr.open_dataset(fn, decode_cf=False, mask_and_scale=False, engine="netcdf4") as ds:
            juld = ds["JULD"].values
            mids = month_id_from_juld(juld).astype(np.int64)
            moy_all = month_of_year_from_juld(juld)
            uniq_mids = np.unique(mids)

            lat_prof = ds["LATITUDE"].values
            lon_prof = ((ds["LONGITUDE"].values + 180) % 360) - 180

            depth_all = read_var_safe(ds, "DEPH_CORRECTED")
            if depth_all is None:
                depth_all = read_var_safe(ds, "DEPTH")
            if depth_all is None:
                A["file_done"][ii] = True
                continue

            n_prof_total, n_lev = depth_all.shape
            keep_pos = (
                qc_keep_profile(to_str_array(ds["POSITION_QC"].values))
                if "POSITION_QC" in ds
                else np.ones(n_prof_total, dtype=bool)
            )
            y_prof_all, oky = safe_digitize(lat_prof, lat_edges)
            x_prof_all, okx = safe_digitize(lon_prof, lon_edges)
            ok_geo_all = oky & okx & keep_pos

            prof_potm_keep = (
                qc_keep_profile(to_str_array(ds["PROFILE_POTM_QC"].values))
                if "PROFILE_POTM_QC" in ds
                else np.ones(n_prof_total, dtype=bool)
            )
            prof_psal_keep = (
                qc_keep_profile(to_str_array(ds["PROFILE_PSAL_QC"].values))
                if "PROFILE_PSAL_QC" in ds
                else np.ones(n_prof_total, dtype=bool)
            )
            lev_potm_keep = (
                qc_keep_levels(to_str_array(ds["POTM_CORRECTED_QC"].values), n_lev)
                if "POTM_CORRECTED_QC" in ds
                else np.ones((n_prof_total, n_lev), dtype=bool)
            )
            lev_psal_keep = (
                qc_keep_levels(to_str_array(ds["PSAL_CORRECTED_QC"].values), n_lev)
                if "PSAL_CORRECTED_QC" in ds
                else np.ones((n_prof_total, n_lev), dtype=bool)
            )

            potm_all = read_var_safe(ds, "POTM_CORRECTED")
            psal_all = read_var_safe(ds, "PSAL_CORRECTED")
            potm_err_all = read_var_safe(ds, "POTM_CORRECTED_ERROR")
            psal_err_all = read_var_safe(ds, "PSAL_CORRECTED_ERROR")

            if potm_all is not None:
                T_clim_levels = clim_for_profile_levels(
                    T_interp, depth_all, lat_prof, lon_prof, moy_all
                )
            if psal_all is not None:
                S_clim_levels = clim_for_profile_levels(
                    S_interp, depth_all, lat_prof, lon_prof, moy_all
                )

            for mid in uniq_mids:
                t = id2t.get(int(mid))
                if t is None:
                    continue
                sel = mids == mid
                if not np.any(sel):
                    continue

                ok_geo = ok_geo_all[sel]
                y_prof = y_prof_all[sel]
                x_prof = x_prof_all[sel]
                d_sel = depth_all[sel, :]

                if potm_all is not None:
                    v = potm_all[sel, :]
                    e = potm_err_all[sel, :] if potm_err_all is not None else None
                    clim = T_clim_levels[sel, :]
                    ok_lev = (
                        lev_potm_keep[sel, :]
                        & prof_potm_keep[sel][:, None]
                        & mask_fill(v)
                        & mask_fill(d_sel)
                        & np.isfinite(clim)
                    )
                    anom = v.astype(np.float32) - clim
                    pbv, pbe, pbok, pbcov = profile_bin_averages_depth_weighted(
                        anom, e, d_sel, ok_lev, depth_edges
                    )
                    pbok &= (pbcov >= min_coverage) | (pbcov == 0.0)
                    s, ss, c, ess = accumulate_profiles_to_grid_fast(
                        pbv, pbe, pbok, y_prof, x_prof, ok_geo
                    )
                    if c.sum() > 0:
                        write_back("potm_sum", t, s, np.float32)
                        write_back("potm_sumsq", t, ss, np.float32)
                        write_back("potm_count", t, c, np.int32)
                        write_back("potm_err_sumsq", t, ess, np.float32)

                if psal_all is not None:
                    v = psal_all[sel, :]
                    e = psal_err_all[sel, :] if psal_err_all is not None else None
                    clim = S_clim_levels[sel, :]
                    ok_lev = (
                        lev_psal_keep[sel, :]
                        & prof_psal_keep[sel][:, None]
                        & mask_fill(v)
                        & mask_fill(d_sel)
                        & np.isfinite(clim)
                    )
                    anom = v.astype(np.float32) - clim
                    pbv, pbe, pbok, pbcov = profile_bin_averages_depth_weighted(
                        anom, e, d_sel, ok_lev, depth_edges
                    )
                    pbok &= (pbcov >= min_coverage) | (pbcov == 0.0)
                    s, ss, c, ess = accumulate_profiles_to_grid_fast(
                        pbv, pbe, pbok, y_prof, x_prof, ok_geo
                    )
                    if c.sum() > 0:
                        write_back("psal_sum", t, s, np.float32)
                        write_back("psal_sumsq", t, ss, np.float32)
                        write_back("psal_count", t, c, np.int32)
                        write_back("psal_err_sumsq", t, ess, np.float32)

        A["file_done"][ii] = True
        if (int(ii) % 25 == 0) or (ii == todo_files[-1]):
            print(
                f"[file {ii + 1}/{len(files)}] {os.path.basename(fn)}  "
                f"done: {int(A['file_done'][:].sum())}"
            )

    print("Exporting final product…")
    st = xr.open_zarr(accum_zarr).chunk({"time": 1, "depth": -1})

    def compute_fields(sum_, sumsq_, count_, err_sumsq_, error_floor=0.0):
        n = count_.where(count_ > 0).astype("float64")
        mean = (sum_ / n).astype("float32")
        var = ((sumsq_ / n) - (sum_ / n) ** 2).clip(min=0)
        sampling_std = np.sqrt(var).astype("float32")
        raw_obs_err = np.sqrt((err_sumsq_ / n).clip(min=0))
        obs_error = np.maximum(raw_obs_err, error_floor).astype("float32")
        total_error = np.sqrt(
            (var / n + obs_error.astype("float64") ** 2 / n).clip(min=0)
        ).astype("float32")
        return mean, sampling_std, obs_error, total_error, count_.astype("int32")

    potm_anom, potm_sstd, potm_oerr, potm_terr, potm_n = compute_fields(
        st["potm_sum"], st["potm_sumsq"], st["potm_count"], st["potm_err_sumsq"],
        error_floor=POTM_ERROR_FLOOR,
    )
    psal_anom, psal_sstd, psal_oerr, psal_terr, psal_n = compute_fields(
        st["psal_sum"], st["psal_sumsq"], st["psal_count"], st["psal_err_sumsq"],
        error_floor=PSAL_ERROR_FLOOR,
    )

    clim_T_tyxz = xr.DataArray(
        T_CLIM_BIN[times_moy - 1],
        dims=("time", "lat", "lon", "depth"),
        coords=dict(time=times, lat=lat, lon=lon, depth=depth),
    ).astype("float32")
    clim_S_tyxz = xr.DataArray(
        S_CLIM_BIN[times_moy - 1],
        dims=("time", "lat", "lon", "depth"),
        coords=dict(time=times, lat=lat, lon=lon, depth=depth),
    ).astype("float32")

    potm_mean = (potm_anom + clim_T_tyxz).astype("float32")
    psal_mean = (psal_anom + clim_S_tyxz).astype("float32")

    out = xr.Dataset(
        coords=dict(
            time=st["time"], lat=st["lat"], lon=st["lon"], depth=st["depth"],
            depth_upper=st["depth_upper"], depth_lower=st["depth_lower"],
        ),
        data_vars=dict(
            potm_mean=potm_mean,
            psal_mean=psal_mean,
            potm_sampling_std=potm_sstd,
            potm_obs_error=potm_oerr,
            potm_total_error=potm_terr,
            potm_n_profiles=potm_n,
            psal_sampling_std=psal_sstd,
            psal_obs_error=psal_oerr,
            psal_total_error=psal_terr,
            psal_n_profiles=psal_n,
            potm_anomaly_mean=potm_anom,
            psal_anomaly_mean=psal_anom,
            potm_clim_bin=clim_T_tyxz,
            psal_clim_bin=clim_S_tyxz,
        ),
        attrs=dict(
            title="EN4 monthly 5° × 5° gridded T/S (raw, for downstream reconstruction)",
            grid=f"{lat_res}° × {lon_res}°",
            depth_edges=str(depth_edges.tolist()),
            baseline_climatology="1993-01 .. 2016-12",
            min_coverage=float(min_coverage),
            max_gap_m=float(max_gap),
            T_units="degC",
            S_units="PSU",
            methodology=(
                "Raw EN4 profile levels -> anomaly (level minus climatology). "
                "Depth-weighted trapezoidal averaging within each (profile, bin). "
                "NO vertical extrapolation: empty bins remain NaN."
            ),
        ),
    ).chunk({"time": 1, "depth": -1})

    Path(output_zarr).parent.mkdir(parents=True, exist_ok=True)
    out.to_zarr(output_zarr, mode="w")
    print(f"Wrote: {output_zarr}")
    return Path(output_zarr)


# ===========================================================================
# CLI
# ===========================================================================
def _add_clim_args(p):
    p.add_argument("--analysis_nc", required=True)
    p.add_argument("--clim_out", default="en4_climatology.nc")
    p.add_argument("--base_start", default=DEFAULT_BASE_START)
    p.add_argument("--base_end", default=DEFAULT_BASE_END)


def _add_grid_args(p, clim_required=True):
    p.add_argument("--profiles_dir", required=True)
    p.add_argument("--clim_file", required=clim_required, default=None)
    p.add_argument("--accum_zarr", default="en4_binned_anom.zarr")
    p.add_argument("--output_zarr", required=True)
    p.add_argument("--pattern", default="*.nc")
    p.add_argument("--lat_res", type=float, default=DEFAULT_LAT_RES)
    p.add_argument("--lon_res", type=float, default=DEFAULT_LON_RES)
    p.add_argument("--min_coverage", type=float, default=MIN_COVERAGE)
    p.add_argument("--max_gap", type=float, default=MAX_GAP)


def main(argv=None):
    parser = argparse.ArgumentParser(description="EN4 clim + 5° gridded T/S")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_clim = sub.add_parser("climatology", help="clim from analysis NC")
    _add_clim_args(p_clim)

    p_grid = sub.add_parser("grid", help="grid profiles")
    _add_grid_args(p_grid, clim_required=True)

    p_all = sub.add_parser("all", help="clim then grid")
    _add_clim_args(p_all)
    _add_grid_args(p_all, clim_required=False)
    p_all.set_defaults(clim_file=None)

    args = parser.parse_args(argv)

    if args.cmd == "climatology":
        build_climatology(args.analysis_nc, args.clim_out, args.base_start, args.base_end)
        return

    if args.cmd == "all":
        clim_path = build_climatology(
            args.analysis_nc, args.clim_out, args.base_start, args.base_end
        )
        clim_file = args.clim_file or str(clim_path)
    else:
        clim_file = args.clim_file

    grid_en4_profiles(
        profiles_dir=args.profiles_dir,
        clim_file=clim_file,
        accum_zarr=args.accum_zarr,
        output_zarr=args.output_zarr,
        pattern=args.pattern,
        lat_res=args.lat_res,
        lon_res=args.lon_res,
        min_coverage=args.min_coverage,
        max_gap=args.max_gap,
    )


if __name__ == "__main__":
    main()
