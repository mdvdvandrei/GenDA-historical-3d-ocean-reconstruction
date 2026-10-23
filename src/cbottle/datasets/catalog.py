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
import cbottle.storage
import zarr
from typing import Literal
from dataclasses import dataclass
import xarray
from cbottle.config import environment
import zarr.storage
import os


@dataclass
class _Zarr:
    path: str
    profile: str

    @property
    def storage_options(self):
        return cbottle.storage.get_storage_options(self.profile)

    def to_zarr(self) -> zarr.Group:
        so = cbottle.storage.get_storage_options(self.profile)
        return zarr.open_consolidated(self.path, storage_options=so)

    def to_xarray(self, **kwargs) -> xarray.Dataset:
        return xarray.open_zarr(self.to_zarr().store, **kwargs)

    def consolidate_metadata(self):
        store = zarr.storage.FsspecStore.from_url(
            self.path, storage_options=cbottle.storage.get_storage_options(self.profile)
        )
        zarr.consolidate_metadata(store)


def icon(level: int, freq: Literal["PT3H", "P1D", "PT30M"]):
    path = environment.RAW_DATA_URL.rstrip("/")
    dir = os.path.dirname(path)
    return _Zarr(f"{dir}/ngc3028_{freq}_{level}.zarr/", environment.RAW_DATA_PROFILE)


def icon_land(level: int, freq: Literal["P1D", "PT3H"]):
    path = environment.LAND_DATA_URL_6.rstrip("/")
    dir = os.path.dirname(path)
    return _Zarr(f"{dir}/ngc3028_{freq}_{level}.zarr/", environment.LAND_DATA_PROFILE)


def icon_plevel():
    return _Zarr(environment.V6_ICON_ZARR, environment.V6_ICON_ZARR_PROFILE)


def icon_sst_monmean():
    return _Zarr(
        environment.SST_MONMEAN_DATA_URL_6, environment.SST_MONMEAN_DATA_PROFILE
    )


def era5_hpx6():
    return _Zarr(environment.V6_ERA5_ZARR, environment.V6_ERA5_ZARR_PROFILE)
