"""DDPM math, written out explicitly.

This module implements the core math of the Denoising Diffusion Probabilistic
Models (DDPM) framework from Ho et al. 2020. Nothing here learns anything. The
point is to make the closed forms readable and to give the tests something
concrete to check against.

Notation follows the paper:

  beta_t              variance added at forward step t
  alpha_t   = 1 - beta_t
  alpha_bar_t = prod_{s=1..t} alpha_s    (cumulative product)

Forward process (adds noise):

  q(x_t | x_{t-1}) = N(sqrt(alpha_t) x_{t-1}, beta_t I)

Closed form forward marginal (jump straight from x_0 to x_t):

  q(x_t | x_0) = N(sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I)

Forward posterior used by the reverse process:

  q(x_{t-1} | x_t, x_0) = N(mu_tilde_t(x_t, x_0), beta_tilde_t I)

with

  beta_tilde_t = (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) * beta_t

  mu_tilde_t   = sqrt(alpha_bar_{t-1}) beta_t / (1 - alpha_bar_t) * x_0
               + sqrt(alpha_t) (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) * x_t

The reverse mean expressed through a predicted noise eps:

  mu_theta(x_t, t) = 1/sqrt(alpha_t) * ( x_t - beta_t / sqrt(1 - alpha_bar_t) * eps )

All tensors use float64 by default so the algebra checks are tight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def linear_beta_schedule(
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Linear beta schedule, betas evenly spaced from beta_start to beta_end.

    Returns a 1D tensor of length ``timesteps``. Index t in [0, timesteps) maps
    to forward step t+1 in the paper's 1-indexed math.
    """
    if timesteps < 1:
        raise ValueError("timesteps must be >= 1")
    if not (0.0 < beta_start < 1.0 and 0.0 < beta_end < 1.0):
        raise ValueError("betas must lie strictly inside (0, 1)")
    return torch.linspace(beta_start, beta_end, timesteps, dtype=dtype)


def cosine_beta_schedule(
    timesteps: int,
    s: float = 0.008,
    max_beta: float = 0.999,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Cosine schedule from Nichol and Dhariwal 2021.

    The schedule is defined through alpha_bar and then betas are recovered as
    ``beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}``, clipped to ``max_beta`` for
    numerical safety near the end.
    """
    if timesteps < 1:
        raise ValueError("timesteps must be >= 1")

    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=dtype)
    alphas_bar = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1.0 - (alphas_bar[1:] / alphas_bar[:-1])
    return torch.clamp(betas, max=max_beta)


@dataclass
class DiffusionSchedule:
    """Precomputed schedule quantities derived from a beta sequence.

    Build one with :meth:`from_betas`. All stored tensors are 1D and share the
    same length as ``betas``. Step t in the paper (1-indexed) corresponds to
    tensor index ``t - 1`` here.
    """

    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_bar: torch.Tensor
    alphas_bar_prev: torch.Tensor
    sqrt_alphas_bar: torch.Tensor
    sqrt_one_minus_alphas_bar: torch.Tensor
    posterior_variance: torch.Tensor
    posterior_mean_coef_x0: torch.Tensor
    posterior_mean_coef_xt: torch.Tensor

    @classmethod
    def from_betas(cls, betas: torch.Tensor) -> "DiffusionSchedule":
        if betas.dim() != 1:
            raise ValueError("betas must be a 1D tensor")
        if torch.any(betas <= 0) or torch.any(betas >= 1):
            raise ValueError("every beta must lie strictly inside (0, 1)")

        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # alpha_bar_{t-1}: for t = 1 there is no previous step, the convention
        # is alpha_bar_0 = 1.
        alphas_bar_prev = torch.cat(
            [torch.ones(1, dtype=alphas_bar.dtype, device=alphas_bar.device), alphas_bar[:-1]]
        )

        sqrt_alphas_bar = torch.sqrt(alphas_bar)
        sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - alphas_bar)

        # beta_tilde_t, the analytic posterior variance.
        posterior_variance = betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)

        posterior_mean_coef_x0 = (
            betas * torch.sqrt(alphas_bar_prev) / (1.0 - alphas_bar)
        )
        posterior_mean_coef_xt = (
            torch.sqrt(alphas) * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)
        )

        return cls(
            betas=betas,
            alphas=alphas,
            alphas_bar=alphas_bar,
            alphas_bar_prev=alphas_bar_prev,
            sqrt_alphas_bar=sqrt_alphas_bar,
            sqrt_one_minus_alphas_bar=sqrt_one_minus_alphas_bar,
            posterior_variance=posterior_variance,
            posterior_mean_coef_x0=posterior_mean_coef_x0,
            posterior_mean_coef_xt=posterior_mean_coef_xt,
        )

    def __len__(self) -> int:
        return self.betas.shape[0]


def _gather(values: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Pick ``values[t]`` per batch element and broadcast over a sample shape.

    ``t`` is a 1D long tensor of indices (one per batch element). The result has
    shape ``(batch, 1, 1, ...)`` matching the rank of ``shape`` so it multiplies
    cleanly against a batch of samples.
    """
    out = values.to(t.device).gather(0, t)
    while out.dim() < len(shape):
        out = out.unsqueeze(-1)
    return out


def forward_step(
    schedule: DiffusionSchedule,
    x_prev: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    """One forward step, sampling x_t from x_{t-1} given a noise draw.

    Implements x_t = sqrt(alpha_t) x_{t-1} + sqrt(beta_t) * noise, which is the
    reparameterized sample from q(x_t | x_{t-1}). ``t`` indexes the step.
    """
    alpha_t = _gather(schedule.alphas, t, x_prev.shape)
    beta_t = _gather(schedule.betas, t, x_prev.shape)
    return torch.sqrt(alpha_t) * x_prev + torch.sqrt(beta_t) * noise


def q_sample(
    schedule: DiffusionSchedule,
    x_0: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    """Closed form forward marginal sample of x_t directly from x_0.

    Implements x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) * noise.
    """
    sqrt_ab = _gather(schedule.sqrt_alphas_bar, t, x_0.shape)
    sqrt_one_minus_ab = _gather(schedule.sqrt_one_minus_alphas_bar, t, x_0.shape)
    return sqrt_ab * x_0 + sqrt_one_minus_ab * noise


def q_posterior_mean_variance(
    schedule: DiffusionSchedule,
    x_0: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and variance of the forward posterior q(x_{t-1} | x_t, x_0).

    Returns ``(mean, variance)``. The mean has the same shape as the inputs, the
    variance is broadcast from the per step scalar beta_tilde_t.
    """
    coef_x0 = _gather(schedule.posterior_mean_coef_x0, t, x_0.shape)
    coef_xt = _gather(schedule.posterior_mean_coef_xt, t, x_t.shape)
    mean = coef_x0 * x_0 + coef_xt * x_t
    variance = _gather(schedule.posterior_variance, t, x_t.shape)
    return mean, variance


def predict_x0_from_noise(
    schedule: DiffusionSchedule,
    x_t: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    """Invert the forward marginal to recover x_0 from x_t and the noise.

    From x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps we solve for x_0.
    """
    sqrt_ab = _gather(schedule.sqrt_alphas_bar, t, x_t.shape)
    sqrt_one_minus_ab = _gather(schedule.sqrt_one_minus_alphas_bar, t, x_t.shape)
    return (x_t - sqrt_one_minus_ab * noise) / sqrt_ab


def reverse_mean_from_noise(
    schedule: DiffusionSchedule,
    x_t: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    """The reverse model mean mu_theta written through a predicted noise.

    mu_theta = 1/sqrt(alpha_t) * ( x_t - beta_t / sqrt(1 - alpha_bar_t) * eps ).

    A short algebra identity links this to the posterior mean: plugging the x_0
    recovered by :func:`predict_x0_from_noise` into
    :func:`q_posterior_mean_variance` gives exactly this expression. The tests
    confirm the two agree.
    """
    alpha_t = _gather(schedule.alphas, t, x_t.shape)
    beta_t = _gather(schedule.betas, t, x_t.shape)
    sqrt_one_minus_ab = _gather(schedule.sqrt_one_minus_alphas_bar, t, x_t.shape)
    return (x_t - beta_t / sqrt_one_minus_ab * noise) / torch.sqrt(alpha_t)
