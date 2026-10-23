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
from dataclasses import dataclass, field, asdict


@dataclass
class StrategyWeights:
    """
    The probability with which each frame masking strategy is sampled during training.
    Should sum to 1.0.
    """

    random: float = 0.3
    blockwise: float = 0.25
    interpolation: float = 0.1
    full_dropout: float = 0.35

    def __post_init__(self):
        """Validate that weights are non-negative and sum to 1.0."""
        total_weight = 0
        for name, weight in asdict(self).items():
            if weight < 0:
                raise ValueError(
                    f"Strategy weight '{name}' cannot be negative: {weight}"
                )
            total_weight += weight

        if abs(total_weight - 1.0) > 1e-5:
            raise ValueError(
                f"Strategy weights must sum to 1.0, but sum to {total_weight}"
            )


@dataclass
class MaskingConfig:
    """Configuration for frame masking in video models."""

    strategy_weights: StrategyWeights = field(default_factory=StrategyWeights)

    # Random strategy parameters
    random_mask_prob: float = 0.80  # Probability to mask each frame (>0.5)

    # Blockwise strategy parameters
    block_mask_fraction: float = 0.80  # Fraction of frames to mask
    block_predict_past: bool = True  # Whether to also predict past frames

    # Interpolation strategy parameters
    interpolation_mask_fraction: float = 0.80  # Fraction of interior frames to mask


def base_masking_config() -> MaskingConfig:
    return MaskingConfig(
        strategy_weights=StrategyWeights(
            random=0.3,
            blockwise=0.25,
            interpolation=0.1,
            full_dropout=0.35,
        ),
        random_mask_prob=0.8,
        block_mask_fraction=0.8,
        block_predict_past=True,
        interpolation_mask_fraction=0.8,
    )
