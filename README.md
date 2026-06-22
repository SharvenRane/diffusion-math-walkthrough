# diffusion-math-walkthrough

A small, tested walkthrough of the math behind DDPM (Ho et al. 2020). The goal
is not to train anything. It is to write the core closed forms in plain code,
keep them readable, and back every formula with a test that fails loudly if the
algebra is wrong.

Everything runs on CPU in about a second. The tensors are tiny and there are no
downloads.

## What is in here

`src/ddpm.py` holds the whole story:

* `linear_beta_schedule` and `cosine_beta_schedule` build the noise schedule.
* `DiffusionSchedule.from_betas` precomputes the derived quantities: the alphas,
  the cumulative product `alpha_bar`, the shifted `alpha_bar_prev`, and the
  posterior coefficients.
* `forward_step` applies one noising step, sampling `x_t` from `x_{t-1}`.
* `q_sample` is the closed form forward marginal that jumps straight from `x_0`
  to `x_t` in a single shot.
* `q_posterior_mean_variance` returns the mean and variance of
  `q(x_{t-1} | x_t, x_0)`.
* `predict_x0_from_noise` inverts the forward marginal.
* `reverse_mean_from_noise` writes the reverse model mean through a predicted
  noise term.

## The math, briefly

The forward process adds Gaussian noise one step at a time:

```
q(x_t | x_{t-1}) = N(sqrt(alpha_t) x_{t-1}, beta_t I)
```

Because each step is linear and Gaussian, you can compose all the steps and get
a closed form that skips straight to any `t`:

```
q(x_t | x_0) = N(sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I)
```

where `alpha_t = 1 - beta_t` and `alpha_bar_t` is the running product of the
alphas up to `t`. That single expression is the workhorse of training, since it
lets you sample any noise level without walking the chain.

Reversing the chain needs the forward posterior, which is also Gaussian:

```
q(x_{t-1} | x_t, x_0) = N(mu_tilde_t, beta_tilde_t I)
```

with an analytic variance

```
beta_tilde_t = (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) * beta_t
```

and a mean that blends `x_0` and `x_t`. The reverse model in DDPM predicts the
noise instead of `x_0`, which gives the equivalent mean

```
mu_theta = 1/sqrt(alpha_t) * ( x_t - beta_t / sqrt(1 - alpha_bar_t) * eps )
```

The convention `alpha_bar_0 = 1` makes the first step behave: at `t = 1` the
posterior variance is exactly zero because `x_0` is fully determined.

## What the tests actually check

These are behavior checks, not snapshots of numbers I typed in.

* **Closed form matches the step by step chain.** The test iterates the forward
  process one step at a time with a fresh standard normal at every step, then
  recovers the single accumulated noise that the closed form must consume and
  confirms `q_sample` rebuilds the exact same `x_t`. They agree to within `1e-9`.
* **Accumulated noise has the right variance.** Running the chain from `x_0 = 0`
  over twenty thousand trials, the empirical variance of the state at the final
  step lands on `1 - alpha_bar_t`, the variance the closed form claims, within
  three percent.
* **Posterior variance equals the analytic formula.** The stored
  `posterior_variance` matches `beta_tilde_t` per step, and `beta_tilde_1` is
  zero.
* **Reverse mean equals the posterior mean.** Feeding the true noise through the
  noise form of the reverse mean gives the same vector as recovering `x_0` and
  plugging it into the posterior mean. This is the identity that ties
  `mu_theta` to `mu_tilde`.
* **`predict_x0_from_noise` inverts `q_sample`** exactly.
* Schedule invariants: `alpha_bar` is a strictly decreasing cumulative product,
  the cosine schedule stays inside `(0, 1)` and keeps `alpha_bar` near one at the
  start, and bad inputs raise.

## Running it

```
pip install -r requirements.txt
python -m pytest tests/ -q
```

On my machine this reports `15 passed` in about a second on CPU.

## Layout

```
src/ddpm.py        the math
tests/test_ddpm.py the property checks
requirements.txt   torch and pytest
```

## A note on precision

Everything defaults to `float64`. The point of the repo is to verify algebra, so
the tolerances are tight and double precision keeps the closed form versus chain
comparison honest at the `1e-9` level. In a real training setup you would drop to
`float32`, which is fine because the model never relies on these checks holding
to nine decimals.
