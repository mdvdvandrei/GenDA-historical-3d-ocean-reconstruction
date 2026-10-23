# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import xarray as xr
import numpy as np
import pandas as pd
from cbottle.config import environment as config
import cbottle.storage
from cbottle.datasets.zarr_loader import NO_LEVEL
import torch
import earth2grid.healpix
import logging

PUBLIC_URL = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/netcdf/IBTrACS.since1980.v04r01.nc"


class IBTracs:
    """Handle IBTrACS (International Best Track Archive for Climate Stewardship) data.

    Loads, interpolates, and allows to time-select the data.

    Attributes:
        level: The HPX level to return the classification levels on. Defaults to 3.
        tracks_interp (xarray.Dataset): Interpolated track data containing lat/lon coordinates
    """

    KEY = ("ibtracs_labels", NO_LEVEL)

    def __init__(self, level: int = 3):
        logging.info("Loading IBTrACS data...")
        cbottle.storage.ensure_downloaded(PUBLIC_URL, config.IBTRACS_DATA_PATH)
        tracks = xr.open_dataset(config.IBTRACS_DATA_PATH, engine="netcdf4").load()
        # Make sure `date_time` is a coordinate
        if "date_time" not in tracks.coords:
            tracks = tracks.assign_coords(
                date_time=np.arange(tracks.sizes["date_time"])
            )

        # Step 1: Reset time as a data variable if it's still a coordinate
        if "time" in tracks.coords:
            tracks = tracks.reset_coords("time")

        # Step 2: Create hourly date_time
        new_date_time = np.arange(0, tracks.sizes["date_time"] - 1 + 1e-6, 1 / 3)

        # Step 3: Interpolate lat/lon only
        tracks_interp = tracks[["lat", "lon"]].interp(date_time=new_date_time)
        tracks_interp = tracks_interp.interpolate_na(
            dim="date_time", method="linear", use_coordinate=False, limit_area="inside"
        )

        # Step 4: Interpolate `time` manually, inserting NaT outside the interpolation range
        original_time = tracks["time"].values  # shape: (storm, date_time)
        num_storms = original_time.shape[0]

        interp_times_list = []

        for i in range(num_storms):
            t_vals = original_time[i]
            # Mask valid indices
            valid = ~pd.isnull(t_vals)
            if valid.sum() < 2:
                # Append NaT for this storm if not enough valid data
                interp_times_list.append(
                    np.full_like(new_date_time, np.datetime64("NaT"))
                )
                continue

            # Find the valid x (indices) and corresponding y (timestamps)
            x = np.arange(len(t_vals))[valid]
            y = pd.to_datetime(t_vals[valid]).astype("int64")  # Convert to nanoseconds

            # Perform interpolation
            y_interp = np.interp(new_date_time, x, y)

            # Create a mask of the interpolated values that should be NaT
            valid_range_mask = (new_date_time >= x.min()) & (new_date_time <= x.max())

            # Insert NaT outside the valid range
            y_interp[~valid_range_mask] = np.datetime64("NaT")

            # Append the interpolated times (with NaT outside the valid range)
            interp_times_list.append(pd.to_datetime(y_interp.astype("int64")))

        interp_times = np.array(interp_times_list, dtype="datetime64[m]")

        # Create the DataArray for time
        time_da = xr.DataArray(
            interp_times,
            dims=("storm", "date_time"),
            coords={"storm": tracks.storm.values, "date_time": new_date_time},
        )

        self.tracks_interp = tracks_interp.assign_coords(time=time_da)
        logging.info("IBTrACS data loaded.")
        self._grid = earth2grid.healpix.Grid(
            level=level, pixel_order=earth2grid.healpix.PixelOrder.NEST
        )

    async def sel_time(self, timestamps) -> dict[tuple[str, int], torch.Tensor]:
        labels = self._get_labels(timestamps)
        return {self.KEY: labels}

    def _get_labels(self, timestamps) -> torch.Tensor:
        # Initialize labels tensor for all timestamps
        n_timestamps = len(timestamps)
        all_labels = torch.zeros(
            (n_timestamps, 1, 1, *self._grid.shape), dtype=torch.float32
        )

        # Flatten (storm, date_time) to a 1D index
        stacked = self.tracks_interp.stack(obs=("storm", "date_time"))

        # Process each timestamp
        for i, timestamp in enumerate(timestamps):
            # Find the entries where time matches this timestamp
            matches = stacked.time == timestamp

            # Extract lat/lon for matching time and convert directly to torch tensors
            matched_lats = torch.from_numpy(
                stacked["lat"].where(matches, drop=True).values
            )
            matched_lons = torch.from_numpy(
                stacked["lon"].where(matches, drop=True).values
            )

            # Convert to healpix indices
            indices_where_tc = self._grid.ang2pix(matched_lons, matched_lats)

            # Set labels for this timestamp
            if len(indices_where_tc) > 0:
                all_labels[i, 0, 0, indices_where_tc] = 1.0

        return all_labels
