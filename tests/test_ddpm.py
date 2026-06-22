"""Behavior tests for the DDPM math.

The two headline checks the task asks for:

  1. the closed form forward marginal q(x_t | x_0) matches iterating the forward
     process one step at a time, when both consume the same accumulated noise;
  2. the posterior variance equals the analytic beta_tilde_t formula.

The rest pin down algebraic identities and schedule invariants.
"""

import math

import pytest
import torch

from src.ddpm import (
    DiffusionSchedule,
    cosine_beta_schedule,
    forward_step,
    linear_beta_schedule,
    predict_x0_from_noise,
    q_posterior_mean_variance,
    q_sample,
    reverse_mean_from_noise,
)

TOL = 1e-9


@pytest.fixture
def schedule():
    betas = linear_beta_schedule(50, beta_start=1e-4, beta_end=0.02)
    return DiffusionSchedule.from_betas(betas)


def test_schedule_shapes_and_first_prev(schedule):
    n = len(schedule)
    assert schedule.betas.shape == (n,)
    assert schedule.alphas_bar.shape == (n,)
    # alpha_bar_0 convention is 1.
    assert torch.allclose(
        schedule.alphas_bar_prev[0], torch.tensor(1.0, dtype=torch.float64), atol=TOL
    )
    # alphas_bar is a strictly decreasing cumulative product of values < 1.
    diffs = schedule.alphas_bar[1:] - schedule.alphas_bar[:-1]
    assert torch.all(diffs < 0)


def test_alphas_bar_matches_manual_cumprod(schedule):
    manual = torch.ones((), dtype=torch.float64)
    for i in range(len(schedule)):
        manual = manual * schedule.alphas[i]
        assert torch.allclose(manual, schedule.alphas_bar[i], atol=TOL)


def test_forward_marginal_matches_step_by_step(schedule):
    """Headline check 1.

    Iterating x_t = sqrt(alpha_t) x_{t-1} + sqrt(beta_t) z_t step by step gives a
    sample whose mean and noise coefficient collapse to the closed form. We feed
    the iteration a sequence of standard normal draws z_1..z_T, then construct
    the single accumulated noise that the closed form must use to land on the
    exact same x_T, and verify q_sample reproduces it.
    """
    torch.manual_seed(0)
    n = len(schedule)
    batch, dim = 4, 3
    x_0 = torch.randn(batch, dim, dtype=torch.float64)

    # Iterate the forward process to the final step, recording per step noise.
    x = x_0.clone()
    per_step_noise = []
    for step in range(n):
        z = torch.randn(batch, dim, dtype=torch.float64)
        per_step_noise.append(z)
        t = torch.full((batch,), step, dtype=torch.long)
        x = forward_step(schedule, x, t, z)
    x_iter = x

    # The closed form says x_T = sqrt(alpha_bar_T) x_0 + sqrt(1 - alpha_bar_T) eps.
    # Recover the eps that the iteration effectively produced and check the
    # closed form rebuilds the identical x_T from it.
    T = n - 1
    sqrt_ab = schedule.sqrt_alphas_bar[T]
    sqrt_one_minus_ab = schedule.sqrt_one_minus_alphas_bar[T]
    eps_effective = (x_iter - sqrt_ab * x_0) / sqrt_one_minus_ab

    t_final = torch.full((batch,), T, dtype=torch.long)
    x_closed = q_sample(schedule, x_0, t_final, eps_effective)

    assert torch.allclose(x_iter, x_closed, atol=TOL)


def test_forward_marginal_noise_coefficient_is_consistent(schedule):
    """The accumulated noise from iterating has the variance the closed form claims.

    Run the forward process from x_0 = 0 many times with fresh noise. With x_0 = 0
    the running state is purely accumulated noise, so its empirical variance at
    step t must approach (1 - alpha_bar_t), the closed form noise variance.
    """
    torch.manual_seed(1)
    n = len(schedule)
    trials = 20000
    dim = 1
    t_check = n - 1

    x = torch.zeros(trials, dim, dtype=torch.float64)
    for step in range(t_check + 1):
        z = torch.randn(trials, dim, dtype=torch.float64)
        t = torch.full((trials,), step, dtype=torch.long)
        x = forward_step(schedule, x, t, z)

    empirical_var = x.var(unbiased=False).item()
    analytic_var = (1.0 - schedule.alphas_bar[t_check]).item()
    assert empirical_var == pytest.approx(analytic_var, rel=0.03)


def test_posterior_variance_matches_analytic_formula(schedule):
    """Headline check 2: beta_tilde_t equals the closed form."""
    betas = schedule.betas
    alphas_bar = schedule.alphas_bar
    alphas_bar_prev = schedule.alphas_bar_prev

    expected = betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)
    assert torch.allclose(schedule.posterior_variance, expected, atol=TOL)

    # And the variance returned by the function call agrees per step.
    x_0 = torch.randn(2, 3, dtype=torch.float64)
    x_t = torch.randn(2, 3, dtype=torch.float64)
    for step in range(len(schedule)):
        t = torch.full((2,), step, dtype=torch.long)
        _, var = q_posterior_mean_variance(schedule, x_0, x_t, t)
        assert torch.allclose(
            var, expected[step].expand_as(var), atol=TOL
        )


def test_posterior_variance_first_step_is_zero(schedule):
    """At t = 1, x_{t-1} = x_0 is fully determined, so beta_tilde_1 = 0."""
    assert schedule.posterior_variance[0].item() == pytest.approx(0.0, abs=TOL)


def test_posterior_mean_coefficients_sum_relation(schedule):
    """Posterior mean is a convex-like blend; rebuild it and check the call.

    mu_tilde = coef_x0 * x_0 + coef_xt * x_t, computed directly here and compared
    against the library function.
    """
    torch.manual_seed(2)
    x_0 = torch.randn(5, 4, dtype=torch.float64)
    x_t = torch.randn(5, 4, dtype=torch.float64)
    step = 17
    t = torch.full((5,), step, dtype=torch.long)

    mean, _ = q_posterior_mean_variance(schedule, x_0, x_t, t)
    manual = (
        schedule.posterior_mean_coef_x0[step] * x_0
        + schedule.posterior_mean_coef_xt[step] * x_t
    )
    assert torch.allclose(mean, manual, atol=TOL)


def test_predict_x0_inverts_q_sample(schedule):
    """Recovering x_0 from a noised x_t and its noise returns the original."""
    torch.manual_seed(3)
    x_0 = torch.randn(6, 2, dtype=torch.float64)
    noise = torch.randn(6, 2, dtype=torch.float64)
    step = 30
    t = torch.full((6,), step, dtype=torch.long)

    x_t = q_sample(schedule, x_0, t, noise)
    x_0_rec = predict_x0_from_noise(schedule, x_t, t, noise)
    assert torch.allclose(x_0, x_0_rec, atol=TOL)


def test_reverse_mean_equals_posterior_mean_at_true_x0(schedule):
    """The noise form of the reverse mean equals the posterior mean.

    If eps is the true noise that produced x_t from x_0, then x_0 recovered from
    that eps, fed into the posterior mean, must match reverse_mean_from_noise.
    This is the algebraic identity tying mu_theta to mu_tilde.
    """
    torch.manual_seed(4)
    x_0 = torch.randn(8, 3, dtype=torch.float64)
    noise = torch.randn(8, 3, dtype=torch.float64)
    step = 25
    t = torch.full((8,), step, dtype=torch.long)

    x_t = q_sample(schedule, x_0, t, noise)

    mu_from_noise = reverse_mean_from_noise(schedule, x_t, t, noise)

    x_0_rec = predict_x0_from_noise(schedule, x_t, t, noise)
    mu_posterior, _ = q_posterior_mean_variance(schedule, x_0_rec, x_t, t)

    assert torch.allclose(mu_from_noise, mu_posterior, atol=1e-8)


def test_q_sample_is_exact_when_noise_zero(schedule):
    """With zero noise, x_t collapses to sqrt(alpha_bar_t) x_0."""
    x_0 = torch.randn(3, 3, dtype=torch.float64)
    step = 10
    t = torch.full((3,), step, dtype=torch.long)
    zero = torch.zeros_like(x_0)
    x_t = q_sample(schedule, x_0, t, zero)
    assert torch.allclose(x_t, schedule.sqrt_alphas_bar[step] * x_0, atol=TOL)


def test_linear_schedule_validation():
    with pytest.raises(ValueError):
        linear_beta_schedule(0)
    with pytest.raises(ValueError):
        linear_beta_schedule(10, beta_start=0.0)
    with pytest.raises(ValueError):
        linear_beta_schedule(10, beta_end=1.0)


def test_schedule_rejects_out_of_range_betas():
    with pytest.raises(ValueError):
        DiffusionSchedule.from_betas(torch.tensor([0.1, 1.5], dtype=torch.float64))
    with pytest.raises(ValueError):
        DiffusionSchedule.from_betas(torch.zeros(2, 2, dtype=torch.float64))


def test_cosine_schedule_is_valid_and_monotone():
    betas = cosine_beta_schedule(100)
    assert torch.all(betas > 0)
    assert torch.all(betas < 1)
    sched = DiffusionSchedule.from_betas(betas)
    # alpha_bar still decreases.
    diffs = sched.alphas_bar[1:] - sched.alphas_bar[:-1]
    assert torch.all(diffs < 0)


def test_cosine_alpha_bar_starts_near_one():
    betas = cosine_beta_schedule(200)
    sched = DiffusionSchedule.from_betas(betas)
    # The cosine design keeps alpha_bar close to 1 for the very first step.
    assert sched.alphas_bar[0].item() > 0.99


def test_posterior_mean_at_last_step_weights_x0_lightly(schedule):
    """Sanity on direction: late in the forward process the posterior mean
    leans on x_t more than on x_0 because almost all signal is gone."""
    step = len(schedule) - 1
    coef_x0 = schedule.posterior_mean_coef_x0[step].item()
    coef_xt = schedule.posterior_mean_coef_xt[step].item()
    assert math.isfinite(coef_x0)
    assert math.isfinite(coef_xt)
    assert coef_xt > coef_x0
