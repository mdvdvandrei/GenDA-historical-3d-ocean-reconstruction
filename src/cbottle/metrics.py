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
import torch
from torchmetrics import Metric


class BinnedAverage(Metric):
    def __init__(self, bin_edges: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer("bin_edges", bin_edges.float())
        self.num_bins = len(bin_edges) - 1

        self.add_state(
            "bin_sum", default=torch.zeros(self.num_bins), dist_reduce_fx="sum"
        )
        self.add_state(
            "bin_count", default=torch.zeros(self.num_bins), dist_reduce_fx="sum"
        )

    def update(self, bin_by: torch.Tensor, metric: torch.Tensor):
        # Broadcast and flatten inputs
        bin_by, metric = torch.broadcast_tensors(bin_by.float(), metric.float())
        bin_by = bin_by.flatten()
        metric = metric.flatten()

        # Compute bin indices
        bin_idx = torch.bucketize(bin_by, self.bin_edges, right=False) - 1

        # Filter out-of-range values
        valid = (bin_idx >= 0) & (bin_idx < self.num_bins)
        bin_idx = bin_idx[valid]
        metric = metric[valid]

        # Accumulate using bincount
        bin_sum = torch.bincount(bin_idx, weights=metric, minlength=self.num_bins)
        bin_count = torch.bincount(bin_idx, minlength=self.num_bins)

        self.bin_sum += bin_sum
        self.bin_count += bin_count

    def compute(self):
        avg = self.bin_sum / self.bin_count.clamp(min=1)
        return avg.where(self.bin_count > 0, torch.nan)
