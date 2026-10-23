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
from typing import Protocol, Any
from enum import Enum
import numpy as np
from cbottle.domain import Domain
import torch
import dataclasses
from datetime import timedelta


class TimeUnit(Enum):
    """Time units supported by the dataset.
    Values are the pandas frequency strings (offset aliases)"""

    HOUR = "h"
    DAY = "D"
    MINUTE = "min"
    SECOND = "s"

    def to_timedelta(self, steps: float) -> timedelta:
        return {
            TimeUnit.HOUR: timedelta(hours=steps),
            TimeUnit.DAY: timedelta(days=steps),
            TimeUnit.MINUTE: timedelta(minutes=steps),
            TimeUnit.SECOND: timedelta(seconds=steps),
        }[self]


@dataclasses.dataclass
class BatchInfo:
    channels: list[str]
    time_step: int = 1  # Time (in units `time_unit`) between consecutive frames
    time_unit: TimeUnit = TimeUnit.HOUR
    scales: Any | None = None
    center: Any | None = None

    def sel_channels(self, channels: list[str]):
        channels = list(channels)
        index = np.array([self.channels.index(ch) for ch in channels])
        scales = None
        if self.scales is not None:
            scales = np.asarray(self.scales)[index]

        center = None
        if self.center is not None:
            center = np.asarray(self.center)[index]

        return BatchInfo(
            time_step=self.time_step,
            time_unit=self.time_unit,
            channels=channels,
            scales=scales,
            center=center,
        )

    def denormalize(self, x):
        scales = torch.as_tensor(self.scales).to(x)
        scales = scales.view(-1, 1, 1)

        center = torch.as_tensor(self.center).to(x)
        center = center.view(-1, 1, 1)
        return x * scales + center

    def get_time_delta(self, t: int) -> timedelta:
        """Gets time offset of the t-th frame in a frame sequence."""
        total_steps = t * self.time_step
        return self.time_unit.to_timedelta(total_steps)


class SpatioTemporalDataset(Protocol):
    @property
    def domain(self) -> Domain:
        pass

    def __len__(self) -> int:
        pass

    @property
    def num_channels(self) -> int:
        pass

    @property
    def condition_channels(self) -> int:
        pass

    @property
    def augment_channels(self) -> int:
        return 0

    @property
    def label_dim(self) -> int:
        return 0

    @property
    def time_length(self) -> int:
        pass

    @property
    def batch_info(self) -> BatchInfo:
        return BatchInfo(
            channels=[str(i) for i in range(self.num_channels)],
        )

    def metadata(self) -> Any:
        """Unstructured metadata about the dataset and the values it yields

        Can be used to save normalization constants, timestamps, channel names,
        config values, etc. The training code will avoid looking into this, but
        could be useful for inference.

        """
        return {}

    def __getitem__(self, idx) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """

        Returns:
            image: shaped (num_channels, time_length, x)
            labels: shaped (label_dim,)
            condition: shaped (condition_channels, time_length, x)
        """
        pass
