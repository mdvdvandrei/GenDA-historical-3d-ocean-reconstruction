#!/usr/bin/env python3
"""
Shared reconstruct helpers used by reconstruct_osse.py and reconstruct_real_obs.py.

Includes: channel helpers, CoarseGridMapper, EDM wrappers, LongDenoiser,
PatchedRealisticScore (windowed SDA), windowing utils.

GIF helpers differ between OSSE (has GT) and real-obs — kept in each script.
Obs loaders / main() also stay per-script.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt_util
from tqdm import tqdm

from earth2grid.healpix import HEALPIX_PAD_XY, Grid

logger = logging.getLogger("reconstruct_common")

SIGMA_DATA = 0.5

DEFAULT_3D_VARS = ["thetao", "so"]
DEFAULT_2D_VARS = ["tos", "sos", "psl", "zos", "tas", "siconc"]
OCEAN_VARS = {"thetao", "so", "tos", "sos", "zos", "siconc"}

def get_channel_names(depths) -> list:
    names = []
    for var in DEFAULT_3D_VARS:
        for d in depths:
            depth_str = f"{int(d)}m" if d < 9999 else "deep"
            names.append(f"{var}_{depth_str}")
    for var in DEFAULT_2D_VARS:
        names.append(var)
    return names


def is_ocean_var(name: str) -> bool:
    """Check if channel name is an ocean variable (handles depth suffixes)."""
    base = name.split("_")[0]
    for ov in OCEAN_VARS:
        if base == ov or base.startswith(ov):
            return True
    return False

def load_checkpoint(path: str, device):
    logger.info(f"Loading checkpoint: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    net = data["ema"] if isinstance(data, dict) and "ema" in data else data
    net = net.to(device).eval()
    logger.info(f"  img_channels={net.img_channels}, time_length={net.time_length}")
    return net

def _parse_date(s: str) -> dt.datetime:
    """Parse 'YYYY-MM' or 'YYYY-MM-DD' into a datetime."""
    for fmt in ("%Y-%m", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date '{s}'. Use YYYY-MM or YYYY-MM-DD.")

def _find_time_index(ds, target_dt: dt.datetime, time_coord: str = 'time') -> int:
    """Find dataset time index for *target_dt*, preferring exact year-month match."""
    times = pd.DatetimeIndex(ds[time_coord].values)
    target = pd.Timestamp(target_dt)

    ym_match = (times.year == target.year) & (times.month == target.month)
    if ym_match.any():
        idx = int(np.where(ym_match)[0][0])
    else:
        idx = int(np.argmin(np.abs(times - target)))

    logger.info(f"    {target_dt:%Y-%m} -> time index {idx} ({times[idx]})")
    return idx

class CoarseGridMapper:
    """
    Differentiable mapping from HEALPix level-6 (PAD_XY, 49152 pixels)
    to a coarse regular lat-lon grid via scatter-add averaging.
    """

    def __init__(self, resolution_deg: float, device: str = 'cuda'):
        self.resolution = resolution_deg
        self.device = device

        hpx_grid = Grid(level=6, pixel_order=HEALPIX_PAD_XY)
        lat_padxy = np.array(hpx_grid.lat)
        lon_padxy = np.array(hpx_grid.lon)
        lon_padxy = np.where(lon_padxy > 180, lon_padxy - 360, lon_padxy)
        self.npix = len(lat_padxy)

        self.lat_edges = np.arange(-90, 90 + resolution_deg, resolution_deg)
        self.lon_edges = np.arange(-180, 180 + resolution_deg, resolution_deg)
        self.n_lat = len(self.lat_edges) - 1
        self.n_lon = len(self.lon_edges) - 1
        self.n_cells = self.n_lat * self.n_lon

        lat_idx = np.clip(np.digitize(lat_padxy, self.lat_edges) - 1, 0, self.n_lat - 1)
        lon_idx = np.clip(np.digitize(lon_padxy, self.lon_edges) - 1, 0, self.n_lon - 1)
        coarse_idx = lat_idx * self.n_lon + lon_idx

        self.hpx_to_coarse = torch.tensor(coarse_idx, dtype=torch.long, device=device)

        counts = torch.zeros(self.n_cells, dtype=torch.float32, device=device)
        counts.scatter_add_(0, self.hpx_to_coarse,
                            torch.ones(self.npix, device=device))
        self.counts = counts.clamp(min=1.0)

        logger.info(f"  CoarseGridMapper {resolution_deg}° -> "
                     f"{self.n_lat}x{self.n_lon} = {self.n_cells} cells")

    def coarsen(self, x_hpx: torch.Tensor) -> torch.Tensor:
        """(npix,) -> (n_cells,)  differentiable coarse-grid average."""
        dev = x_hpx.device
        x_c = torch.zeros(self.n_cells, device=dev, dtype=x_hpx.dtype)
        x_c.scatter_add_(0, self.hpx_to_coarse.to(dev), x_hpx)
        return x_c / self.counts.to(dev, dtype=x_hpx.dtype)

    def coarsen_batch(self, x_batch: torch.Tensor) -> torch.Tensor:
        """(B, npix) -> (B, n_cells)  batched differentiable coarse-grid average."""
        B = x_batch.shape[0]
        dev = x_batch.device
        idx = self.hpx_to_coarse.to(dev).unsqueeze(0).expand(B, -1)
        x_c = torch.zeros(B, self.n_cells, device=dev, dtype=x_batch.dtype)
        x_c.scatter_add_(1, idx, x_batch)
        return x_c / self.counts.to(dev, dtype=x_batch.dtype).unsqueeze(0)

    def regrid_mask(self, data_lat, data_lon, data_mask_2d):
        """Convert data-product mask to coarse-cell boolean mask (n_cells,)."""
        lon_mesh, lat_mesh = np.meshgrid(data_lon, data_lat)
        lat_flat = lat_mesh.ravel()
        lon_flat = lon_mesh.ravel()
        lon_flat = np.where(lon_flat > 180, lon_flat - 360, lon_flat)
        mask_flat = data_mask_2d.ravel()

        obs_idx = np.where(mask_flat)[0]
        if len(obs_idx) == 0:
            return np.zeros(self.n_cells, dtype=bool)

        lat_obs = lat_flat[obs_idx]
        lon_obs = lon_flat[obs_idx]
        lat_bin = np.clip(np.digitize(lat_obs, self.lat_edges) - 1, 0, self.n_lat - 1)
        lon_bin = np.clip(np.digitize(lon_obs, self.lon_edges) - 1, 0, self.n_lon - 1)
        flat = lat_bin * self.n_lon + lon_bin

        coarse_mask = np.zeros(self.n_cells, dtype=bool)
        coarse_mask[flat] = True
        return coarse_mask

def make_centered_window_indices_reflect(L: int, T: int, device):
    assert T >= 2
    r = T // 2
    center_pos = r - 1 if (T % 2 == 0) else r
    base = torch.arange(L, device=device)[:, None]
    offsets = torch.arange(T, device=device)[None, :] - center_pos
    idx = base + offsets
    idx = torch.where(idx < 0, -idx, idx)
    idx = torch.where(idx >= L, 2 * (L - 1) - idx, idx)
    idx = idx.clamp(0, L - 1)
    return idx, center_pos


class EDMWrapperProfiles(nn.Module):
    """Wrap profiles net (unconditional) for SDA. Supports BF16 autocast."""
    def __init__(self, net, use_bf16: bool = False):
        super().__init__()
        self.net = net
        self.use_bf16 = use_bf16

    def forward(self, x, sigma, condition, second_of_day, day_of_year):
        B = x.shape[0]
        if isinstance(sigma, (float, int)):
            sigma = torch.full((B,), float(sigma), device=x.device, dtype=torch.float32)
        elif sigma.ndim == 0:
            sigma = sigma.float().expand(B)
        elif sigma.ndim == 1 and sigma.shape[0] == 1 and B > 1:
            sigma = sigma.float().expand(B)
        else:
            sigma = sigma.float()

        if self.use_bf16 and x.device.type == 'cuda':
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = self.net(
                    x.float(), sigma,
                    class_labels=None, condition=None,
                    second_of_day=second_of_day.float(),
                    day_of_year=day_of_year.float(),
                )
                result = out.out if hasattr(out, "out") else out
            return result.float()
        else:
            out = self.net(
                x.float(), sigma,
                class_labels=None, condition=None,
                second_of_day=second_of_day.float(),
                day_of_year=day_of_year.float(),
            )
            return out.out if hasattr(out, "out") else out


class LongDenoiserProfiles(nn.Module):
    """Denoiser for sequences longer than T using tiled windows."""

    def __init__(self, net, months, C, H, device,
                 chunk_windows=4, output_frames=3, use_bf16=False):
        super().__init__()
        self.net = net
        self.T = int(net.time_length)
        self.L = int(months.numel())
        self.device = device
        self.chunk_windows = chunk_windows
        self.output_frames = output_frames

        self.edm = EDMWrapperProfiles(net, use_bf16=use_bf16).to(device)

        day_of_year_full = (months.float() / 11.0).clamp(0, 1)
        self.register_buffer("day_of_year_full", day_of_year_full)
        self.register_buffer("second_of_day_full", torch.zeros_like(day_of_year_full))

        idx, center_pos = make_centered_window_indices_reflect(
            L=self.L, T=self.T, device=device)
        self.center_pos = int(center_pos)

        half = output_frames // 2
        self.extract_start = center_pos - half + 1
        self.extract_end = self.extract_start + output_frames
        assert self.extract_start >= 0
        assert self.extract_end <= self.T

        self.n_windows = (self.L + output_frames - 1) // output_frames
        first_center = half - 1
        centers = torch.arange(self.n_windows, device=device) * output_frames + first_center
        centers = centers.clamp(0, self.L - 1)
        self.register_buffer("window_centers", centers)

        window_idx = idx[centers]
        self.register_buffer("window_idx", window_idx)

        self.C = C
        self.H = H

        self.register_buffer(
            "condition_buffer",
            torch.zeros(chunk_windows, 0, self.T, H, dtype=torch.float32),
        )

        doy_windows = self.day_of_year_full[window_idx]
        sod_windows = self.second_of_day_full[window_idx]
        self.register_buffer("doy_windows", doy_windows)
        self.register_buffer("sod_windows", sod_windows)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor):
        B, C, L, H = x.shape
        assert B == 1 and C == self.C and H == self.H and L == self.L

        chunk_outputs = []
        for chunk_start in range(0, self.n_windows, self.chunk_windows):
            chunk_end = min(self.n_windows, chunk_start + self.chunk_windows)
            chunk_n = chunk_end - chunk_start

            chunk_idx = self.window_idx[chunk_start:chunk_end]
            x_chunk = x[0, :, chunk_idx, :].permute(1, 0, 2, 3)

            doy_chunk = self.doy_windows[chunk_start:chunk_end]
            sod_chunk = self.sod_windows[chunk_start:chunk_end]
            condition = self.condition_buffer[:chunk_n]

            x0_pred = self.edm(x_chunk, sigma, condition=condition,
                               second_of_day=sod_chunk, day_of_year=doy_chunk)
            chunk_outputs.append(x0_pred[:, :, self.extract_start:self.extract_end, :])

        x0_all = torch.cat(chunk_outputs, dim=0)
        n_w, C_out, of, H_out = x0_all.shape
        x0_flat = x0_all.permute(0, 2, 1, 3).reshape(n_w * of, C_out, H_out)
        x0_flat = x0_flat[:L]
        return x0_flat.permute(1, 0, 2).unsqueeze(0)


# =============================================================================
# L adjustment for clean windowing (no reflection padding)
# =============================================================================

def adjust_L_for_windows(L_original: int, T: int, output_frames: int):
    """Compute effective L so windows of size T with stride output_frames tile
    cleanly without reflection padding.

    Returns:
        L_eff:     effective input length (<= L_original)
        n_windows: number of windows
        L_output:  number of output frames (= n_windows * output_frames)
    """
    if L_original < T:
        return L_original, 1, min(L_original, output_frames)
    n_windows = (L_original - T) // output_frames + 1
    L_eff = (n_windows - 1) * output_frames + T
    L_output = n_windows * output_frames
    return L_eff, n_windows, L_output

class PatchedRealisticScore(nn.Module):
    def __init__(
        self,
        edm,
        obs,
        months,
        T: int,
        C: int,
        H: int,
        L: int,
        output_frames: int = 3,
        chunk_windows: int = 4,
        gamma: float = 0.1,
        sigma_obs: float = 0.15,
        alpha: str = "cos",
        eta: float = 1e-3,
        world_size: int = 1,
        rank: int = 0,
        use_checkpoint: bool = True,
    ):
        super().__init__()

        self.edm = edm
        self.T = T
        self.C = C
        self.H = H
        self.L = L
        self.output_frames = output_frames
        self.chunk_windows = chunk_windows
        self.eta = eta
        self.world_size = world_size
        self.rank = rank
        self.use_checkpoint = use_checkpoint
        self.corrector_mode = False

        if alpha == "cos":
            self.alpha_fn = (
                lambda t: torch.cos(math.acos(math.sqrt(eta)) * t) ** 2
            )
        elif alpha == "lin":
            self.alpha_fn = lambda t: 1 - (1 - eta) * t
        elif alpha == "exp":
            self.alpha_fn = lambda t: torch.exp(math.log(eta) * t ** 2)
        else:
            raise ValueError(f"Unknown alpha schedule: {alpha}")

        self.register_buffer(
            "sigma_obs_t",
            torch.as_tensor(sigma_obs, dtype=torch.float32),
        )
        self.register_buffer(
            "gamma_t",
            torch.as_tensor(gamma, dtype=torch.float32),
        )

        stride = output_frames
        self.stride = stride
        n_windows = max(1, (L - T) // stride + 1)
        self.n_windows = n_windows

        self.extract_start = (T - output_frames) // 2
        self.extract_end = self.extract_start + output_frames

        self.L_output = n_windows * output_frames

        device = months.device

        window_starts = torch.arange(n_windows, device=device) * stride
        self.register_buffer("window_starts", window_starts)

        window_idx = (
            window_starts.unsqueeze(1)
            + torch.arange(T, device=device).unsqueeze(0)
        )
        self.register_buffer("window_idx", window_idx)

        win_count = torch.zeros(L, device=device, dtype=torch.float32)
        win_count.scatter_add_(
            0, window_idx.reshape(-1),
            torch.ones(n_windows * T, device=device, dtype=torch.float32),
        )
        self.register_buffer("win_count", win_count.clamp(min=1.0))

        day_of_year_full = (months.float() / 11.0).clamp(0, 1)
        second_of_day_full = torch.zeros_like(day_of_year_full)

        self.register_buffer("doy_windows", day_of_year_full[window_idx])
        self.register_buffer("sod_windows", second_of_day_full[window_idx])

        self.register_buffer(
            "condition_buffer",
            torch.zeros(chunk_windows, 0, T, H, dtype=torch.float32,
                        device=device),
        )

        self.matched_coarsening = getattr(obs, 'matched_coarsening', False)
        if self.matched_coarsening:
            self.register_buffer("std_hpx", obs.std_hpx.clone())
            self._ch_inv_std = {
                k: v.clone() for k, v in obs.ch_inv_std.items()
            }

        self._build_per_window_obs(obs, device)

        logger.info(
            f"PatchedRealisticScore: {n_windows} windows, stride={stride}, "
            f"T={T}, of={output_frames}, extract=[{self.extract_start}:{self.extract_end}], "
            f"L={L}, world_size={world_size}, rank={rank}, "
            f"checkpoint={use_checkpoint}, matched_coarsen={self.matched_coarsening}"
        )

    def _build_per_window_obs(self, obs, device):
        logger.info("Building per-window local observations ...")

        es = self.extract_start
        ee = self.extract_end
        of = self.output_frames
        has_sigma = obs.sigma_obs_vec is not None

        all_items = []
        y_off = 0
        for mapper, items in obs.mapper_groups:
            for ch, t_global, obs_idx in items:
                n = len(obs_idx)
                y_sl = obs.y[y_off:y_off + n]
                s_sl = obs.sigma_obs_vec[y_off:y_off + n] if has_sigma else None
                all_items.append((ch, t_global, mapper, obs_idx, y_sl, s_sl))
                y_off += n

        overall_main_end = min(es + self.n_windows * of, self.L)

        def _owning_window(t_g):
            if t_g < es:
                return 0
            if t_g >= overall_main_end:
                return self.n_windows - 1
            return (t_g - es) // of

        self.per_window_obs = []
        self.per_window_sigma = [] if has_sigma else None
        total_local = 0
        windows_with = 0

        buckets = [[] for _ in range(self.n_windows)]
        for item in all_items:
            ch, t_g, mapper, obs_idx, y_sl, s_sl = item
            w = _owning_window(t_g)
            if w < 0 or w >= self.n_windows:
                continue
            g_start = int(self.window_starts[w])
            if not (g_start <= t_g < g_start + self.T):
                continue
            buckets[w].append((ch, t_g - g_start, mapper, obs_idx, y_sl, s_sl))

        for w in range(self.n_windows):
            entries = []
            y_parts = []
            s_parts = []
            for ch, t_local, mapper, obs_idx, y_sl, s_sl in buckets[w]:
                entries.append((ch, t_local, mapper, obs_idx))
                y_parts.append(y_sl)
                if s_sl is not None:
                    s_parts.append(s_sl)

            if y_parts:
                y_w = torch.cat(y_parts).to(device)
                windows_with += 1
            else:
                y_w = torch.empty(0, device=device, dtype=torch.float32)

            total_local += len(y_w)
            self.per_window_obs.append((y_w, entries))
            if has_sigma and s_parts:
                self.per_window_sigma.append(
                    torch.cat(s_parts).to(device))
            elif has_sigma:
                self.per_window_sigma.append(
                    torch.empty(0, device=device, dtype=torch.float32))

        logger.info(
            f"  {self.n_windows} windows, {windows_with} with obs, "
            f"{total_local:,} total local obs entries "
            f"(global: {obs.n_obs:,}, no double-counting)"
            f"{', per-cell σ_obs' if has_sigma else ''}"
        )

    def mu(self, t):
        return self.alpha_fn(t).clamp_min(1e-5)

    def sigma_sched(self, t):
        return (1 - self.alpha_fn(t) ** 2 + self.eta ** 2).sqrt()

    def get_rank_range(self, total):
        n_local = math.ceil(total / self.world_size)
        start = self.rank * n_local
        end = min(start + n_local, total)
        return start, end

    def _edm_chunk(self, x_input, mu_val, sigma_d, sod, doy):
        chunk_n = x_input.shape[0]
        return self.edm(
            x_input / mu_val, sigma_d,
            condition=self.condition_buffer[:chunk_n],
            second_of_day=sod, day_of_year=doy,
        )

    def _fold_prior_eps(self, eps_out, eps_all, start, end):
        es = self.extract_start
        ee = self.extract_end
        of = self.output_frames
        L  = self.L
        n_local = end - start

        if start == 0 and es > 0:
            eps_out[0, :, :es, :] = eps_all[0, :, :es, :]

        if n_local > 0:
            eps_central = eps_all[:, :, es:ee, :]
            eps_flat = eps_central.permute(0, 2, 1, 3).reshape(
                -1, self.C, self.H
            )
            local_main_start = start * self.stride + es
            local_main_end = min(
                (end - 1) * self.stride + es + of, L
            )
            n_local_main = local_main_end - local_main_start
            eps_out[0, :, local_main_start:local_main_end, :] = (
                eps_flat[:n_local_main].permute(1, 0, 2)
            )

        overall_main_end = min(es + self.n_windows * of, L)
        if end == self.n_windows and overall_main_end < L:
            tail = L - overall_main_end
            eps_out[0, :, overall_main_end:, :] = (
                eps_all[-1, :, ee:ee + tail, :]
            )

    def _forward_prior_only(self, x, mu_val, sigma_d, start, end,
                            n_local, eps_out):
        with torch.no_grad():
            x_win = x[0, :, self.window_idx[start:end], :]
            x_win = x_win.permute(1, 0, 2, 3)
            x0_chunks = []
            for i in range(0, n_local, self.chunk_windows):
                j = min(i + self.chunk_windows, n_local)
                x0_chunks.append(self._edm_chunk(
                    x_win[i:j], mu_val, sigma_d,
                    self.sod_windows[start + i:start + j],
                    self.doy_windows[start + i:start + j],
                ))
            x0_all = torch.cat(x0_chunks, dim=0)
            eps_all = (x_win / mu_val - x0_all) / sigma_d

        self._fold_prior_eps(eps_out, eps_all, start, end)
        if self.world_size > 1:
            torch.distributed.all_reduce(
                eps_out, op=torch.distributed.ReduceOp.SUM,
            )
        return eps_out

    def forward(self, x, t_diff, c=None):
        B, C, L, H = x.shape
        mu_val  = self.mu(t_diff)
        sigma_val = self.sigma_sched(t_diff)
        sigma_d = sigma_val / mu_val

        start, end = self.get_rank_range(self.n_windows)
        n_local = end - start

        eps_out = torch.zeros(1, C, L, H, device=x.device, dtype=x.dtype)

        if n_local == 0:
            if self.world_size > 1:
                torch.distributed.all_reduce(
                    eps_out, op=torch.distributed.ReduceOp.SUM,
                )
            return eps_out

        if self.corrector_mode:
            return self._forward_prior_only(
                x, mu_val, sigma_d, start, end, n_local, eps_out,
            )

        has_any_obs = any(
            len(self.per_window_obs[wi][0]) > 0
            for wi in range(start, end)
        )

        if not has_any_obs:
            return self._forward_prior_only(
                x, mu_val, sigma_d, start, end, n_local, eps_out,
            )

        return self._forward_shared(
            x, mu_val, sigma_val, sigma_d, start, end, n_local, eps_out,
        )

    def _forward_shared(self, x, mu_val, sigma_val, sigma_d,
                        start, end, n_local, eps_out):
        C, L, H = self.C, self.L, self.H

        with torch.enable_grad():
            x_flat = x[0].detach().requires_grad_(True)

            local_idx = self.window_idx[start:end]
            x_win = x_flat[:, local_idx, :]
            x_win = x_win.permute(1, 0, 2, 3)

            x0_chunks = []
            for i in range(0, n_local, self.chunk_windows):
                j = min(i + self.chunk_windows, n_local)
                if self.use_checkpoint:
                    x0_chunk = ckpt_util.checkpoint(
                        self._edm_chunk,
                        x_win[i:j], mu_val, sigma_d,
                        self.sod_windows[start + i:start + j],
                        self.doy_windows[start + i:start + j],
                        use_reentrant=False,
                    )
                else:
                    x0_chunk = self._edm_chunk(
                        x_win[i:j], mu_val, sigma_d,
                        self.sod_windows[start + i:start + j],
                        self.doy_windows[start + i:start + j],
                    )
                x0_chunks.append(x0_chunk)
            x0_all = torch.cat(x0_chunks, dim=0)

            eps_all = (x_win / mu_val - x0_all) / sigma_d

            log_p = x_flat.new_zeros(1)

            for local_i, w_idx in enumerate(range(start, end)):
                y_w, entries = self.per_window_obs[w_idx]
                if len(y_w) == 0:
                    continue
                y_pred_parts = []
                for ch, t_local, mapper, obs_idx in entries:
                    if self.matched_coarsening:
                        raw_anom = (x0_all[local_i, ch, t_local, :]
                                    * self.std_hpx[ch] / SIGMA_DATA)
                        coarsened = (mapper.coarsen(raw_anom)
                                     * self._ch_inv_std[ch] * SIGMA_DATA)
                    else:
                        coarsened = mapper.coarsen(
                            x0_all[local_i, ch, t_local, :]
                        )
                    y_pred_parts.append(coarsened[obs_idx])
                y_pred = torch.cat(y_pred_parts)
                err = y_w - y_pred

                if self.per_window_sigma is not None:
                    sigma_w = self.per_window_sigma[w_idx]
                    var = sigma_w ** 2 + self.gamma_t * sigma_d ** 2
                else:
                    var = self.sigma_obs_t ** 2 + self.gamma_t * sigma_d ** 2

                log_p = log_p - (err ** 2 / var).sum() * 0.5

            (s,) = torch.autograd.grad(log_p, x_flat)

        eps_all = eps_all.detach()
        s = s.detach()

        self._fold_prior_eps(eps_out, eps_all, start, end)
        eps_out[0] -= sigma_val * s

        if self.world_size > 1:
            torch.distributed.all_reduce(
                eps_out, op=torch.distributed.ReduceOp.SUM,
            )

        return eps_out

def _sample_with_threshold(
    sde, score, B, C, L, H, *,
    steps, corrections, tau, corrector_t_threshold,
):
    """Custom sampling loop that restricts corrector to high-noise levels."""
    device = next(sde.parameters()).device
    x = torch.randn(B, C, L, H, device=device)
    time = torch.linspace(1, 0, steps + 1, device=device)
    dt = 1.0 / steps
    dims = tuple(range(1, x.ndim))

    with torch.no_grad():
        for t in tqdm(time[:-1], ncols=88):
            # Predictor (always full posterior)
            score.corrector_mode = False
            r = sde.mu(t - dt) / sde.mu(t)
            eps_pred = score(x, t)
            x = r * x + (sde.sigma(t - dt) - r * sde.sigma(t)) * eps_pred

            # Corrector — only at high noise (t above threshold)
            if corrections > 0 and float(t - dt) < corrector_t_threshold:
                continue
            if corrections > 0:
                score.corrector_mode = True
            for _ in range(corrections):
                z = torch.randn_like(x)
                eps = score(x, t - dt)
                delta = tau / eps.square().mean(dim=dims, keepdim=True)
                x = x - (delta * eps + torch.sqrt(2 * delta) * z) * sde.sigma(t - dt)

    return x

