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
from enum import Enum, auto
from typing import Optional
from cbottle.diffusion_samplers import edm_sampler_steps
from typing import Callable

DenoiserT = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class DenoiserType(Enum):
    mask_filling = auto()  # When the input contains NaNs, uses mask filling denoiser
    infill = auto()  # Fill in the Nans given the not nans.
    standard = auto()
    guided = auto()  # Uses classifier guidance to steer the denoising process


def create_standard_denoiser(
    net,
    *,
    second_of_day,
    day_of_year,
    labels,
    condition,
):
    def D(x_hat, t_hat):
        return net(
            x_hat,
            t_hat,
            labels,
            condition=condition,
            second_of_day=second_of_day,
            day_of_year=day_of_year,
        ).out

    D.round_sigma = net.round_sigma
    D.sigma_max = net.sigma_max
    D.sigma_min = net.sigma_min
    return D


def create_mask_filling_denoiser(
    net,
    mask,
    *,
    second_of_day,
    day_of_year,
    labels,
    condition,
    labels_when_nan: torch.Tensor,
):
    def denoise(x_hat, t_hat):
        D_when_nan = net(
            x_hat,
            t_hat,
            labels_when_nan,
            condition=condition,
            second_of_day=second_of_day,
            day_of_year=day_of_year,
        ).out

        D = net(
            torch.where(mask, x_hat, 0),
            t_hat,
            labels,
            condition=condition,
            second_of_day=second_of_day,
            day_of_year=day_of_year,
        ).out

        # compute icon
        return torch.where(mask, D, D_when_nan)

    denoise.round_sigma = net.round_sigma
    denoise.sigma_max = net.sigma_max
    denoise.sigma_min = net.sigma_min
    return denoise


def _get_infilling_denoiser(
    net,
    mask,
    images,
    tsteps,
    *,
    second_of_day,
    day_of_year,
):
    tsteps = torch.unique_consecutive(torch.tensor(tsteps))
    tsteps = tsteps.flip(0)
    dt = torch.diff(tsteps).float()
    dW = torch.randn([len(dt), *images.shape], dtype=images.dtype) * dt.view(
        -1, 1, 1, 1, 1
    )
    W = torch.cumsum(dW, dim=0)
    y = images + W.to(mask.device)

    def denoise(x, t, labels, condition):
        ts = tsteps[1:]
        # TODO could use linear interp instead
        i = torch.searchsorted(ts.cpu(), t.cpu())
        x = torch.where(mask, y[i], x)
        D = net(
            x,
            t,
            labels,
            condition=condition,
            second_of_day=second_of_day,
            day_of_year=day_of_year,
        ).out
        return D

    return denoise


def create_infilling_denoiser(
    net,
    mask,
    images,
    sigma_max,
    labels,
    condition,
    second_of_day,
    day_of_year,
):
    tsteps = edm_sampler_steps(sigma_max=int(sigma_max))

    base_denoiser = _get_infilling_denoiser(
        net,
        mask,
        images,
        tsteps,
        second_of_day=second_of_day,
        day_of_year=day_of_year,
    )

    # Create a denoiser that matches the expected signature
    def D(x_hat, t_hat):
        return base_denoiser(x_hat, t_hat, labels, condition)

    D.round_sigma = net.round_sigma
    D.sigma_max = net.sigma_max
    D.sigma_min = net.sigma_min
    return D


def get_guidance(guidance_data, logits, x_hat, denoised, t_hat) -> float | torch.Tensor:
    guidance_mask = ~torch.isnan(guidance_data)
    valid_logits = logits[guidance_mask]
    valid_targets = guidance_data[guidance_mask]

    if valid_logits.numel() > 0:
        classifier_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            valid_logits, valid_targets
        )
        classifier_grad = torch.autograd.grad(
            classifier_loss, x_hat, retain_graph=False
        )[0]

        # Match norms for stable guidance
        score = (x_hat - denoised) / t_hat
        score_norm = torch.norm(score)
        classifier_norm = torch.norm(classifier_grad)

        # Avoid division by zero
        if classifier_norm > 0:
            scale = score_norm / classifier_norm
        else:
            scale = 0.0

        # Apply guided denoising
        return -t_hat * scale * classifier_grad
    else:
        return 0


def create_guided_denoiser(
    net,
    *,
    second_of_day,
    day_of_year,
    labels,
    condition,
    guidance_data,
    guidance_scale: float = 0.03,
):
    def D(x_hat, t_hat):
        x_hat.requires_grad_(True)
        out = net(
            x_hat,
            t_hat,
            labels,
            condition=condition,
            second_of_day=second_of_day,
            day_of_year=day_of_year,
        )

        denoised = out.out
        if guidance_scale != 0:
            d_guide = get_guidance(guidance_data, out.logits, x_hat, denoised, t_hat)
            denoised = denoised + guidance_scale * d_guide
        return denoised

    D.round_sigma = net.round_sigma
    D.sigma_max = net.sigma_max
    D.sigma_min = net.sigma_min
    return D


def get_denoiser(
    net,
    images,
    labels,
    condition,
    second_of_day,
    day_of_year,
    *,
    denoiser_type: DenoiserType = DenoiserType.standard,
    sigma_max: float = 80.0,
    labels_when_nan: Optional[torch.Tensor] = None,
    guidance_data: Optional[torch.Tensor] = None,
    guidance_scale: float = 0.0,
):
    mask = ~torch.isnan(images)

    # Regular single-model denoiser logic below
    if denoiser_type == DenoiserType.guided:
        if guidance_data is None:
            raise ValueError("guidance_data must be provided for guided denoiser")
        D = create_guided_denoiser(
            net=net,
            labels=labels,
            condition=condition,
            second_of_day=second_of_day,
            day_of_year=day_of_year,
            guidance_data=guidance_data,
            guidance_scale=guidance_scale,
        )
    elif denoiser_type == DenoiserType.infill:
        D = create_infilling_denoiser(
            net=net,
            mask=mask,
            images=images,
            sigma_max=sigma_max,
            labels=labels,
            condition=condition,
            second_of_day=second_of_day,
            day_of_year=day_of_year,
        )
    elif denoiser_type == DenoiserType.mask_filling and torch.any(~mask):
        if labels_when_nan is None:
            raise ValueError(
                "labels_when_nan must be provided for mask filling denoiser"
            )
        D = create_mask_filling_denoiser(
            net=net,
            mask=mask,
            labels=labels,
            labels_when_nan=labels_when_nan,
            condition=condition,
            second_of_day=second_of_day,
            day_of_year=day_of_year,
        )

    else:
        D = create_standard_denoiser(
            net=net,
            labels=labels,
            condition=condition,
            second_of_day=second_of_day,
            day_of_year=day_of_year,
        )

    return D
