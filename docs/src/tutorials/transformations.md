# Tutorial 2: Transformations

This tutorial covers lag/lead transformations and cumulative responses.

## Setup

```@example trans
using LocalProjections, DataFrames, Random
using StatsModels: @formula
Random.seed!(123)

n = 150
df = DataFrame(
    y = cumsum(randn(n)),
    x = randn(n),
    z = randn(n)
)
nothing # hide
```

## Lag Transformations

Use `lag(x, n)` to include lagged regressors:

```@example trans
# Include lags of control variable
lp_lags = lp(@formula(leads(y) ~ x + lag(z, 1) + lag(z, 2)), df; horizon=8)
println("Coefficient names: ", lp_lags.coef_names[1])
```

### Multiple Lags with `lags()`

Create multiple lag columns at once:

```@example trans
# lags(z, 4) creates z_lag1, z_lag2, z_lag3, z_lag4
lp_multi = lp(@formula(leads(y) ~ x + lags(z, 4)), df; horizon=8)
println("Coefficients: ", lp_multi.coef_names[1])
```

## Forward-Looking Response: `leads()`

The `leads(y)` term specifies ``y_{t+h}`` as the response:

```@example trans
# Standard LP: y at horizon h
lp_leads = lp(@formula(leads(y) ~ x), df; horizon=8)
irf = coefpath(lp_leads)
println("IRF: ", round.(irf, digits=3))
```

## Cumulative Response: `cumul()`

Use `cumul(y)` to estimate cumulative impulse responses:

```@example trans
# Cumulative: sum of y from t to t+h
lp_cumul = lp(@formula(cumul(y) ~ x), df; horizon=8)
irf_cumul = coefpath(lp_cumul)
println("Cumulative IRF: ", round.(irf_cumul, digits=3))
```

Cumulative IRFs measure the **total accumulated effect** over ``[t, t+h]``:
- ``h=0``: ``y_t``
- ``h=1``: ``y_t + y_{t+1}``
- ``h=2``: ``y_t + y_{t+1} + y_{t+2}``

### Use cases for cumulative IRFs

- Total GDP change over forecast horizon
- Cumulative policy effects
- Aggregate impacts on level variables

## Anchored Response: `anchor()` or Pipe `|`

Anchored responses compute ``y_{t+h} - z_t`` (deviation from anchor):

```@example trans
df.baseline = cumsum(randn(n)) ./ 10

# Function syntax
lp_anchor = lp(@formula(anchor(y, baseline) ~ x), df; horizon=8)

# Pipe syntax (equivalent)
lp_pipe = lp(@formula(leads(y)|baseline ~ x), df; horizon=8)

# Verify they're identical
println("Coefficients match: ", coefpath(lp_anchor) ≈ coefpath(lp_pipe))
```

The anchor stays **fixed at time ``t``** while ``y`` shifts forward:
- At ``h=0``: ``y_t - z_t``
- At ``h=1``: ``y_{t+1} - z_t``
- At ``h=2``: ``y_{t+2} - z_t``

## Nested Transformations

Combine transformations for complex responses:

```@example trans
df.y_pos = abs.(df.y) .+ 1  # Positive values for log

# Cumulative of log
lp_cumul_log = lp(@formula(cumul(log(y_pos)) ~ x), df; horizon=5)
println("Cumulative log IRF: ", round.(coefpath(lp_cumul_log), digits=4))

# Leads of log
lp_leads_log = lp(@formula(leads(log(y_pos)) ~ x), df; horizon=5)
println("Leads log IRF: ", round.(coefpath(lp_leads_log), digits=4))
```

### Anchored log differences

```@example trans
# Anchored log transformation
lp_anchor_log = lp(@formula(anchor(log(y_pos), baseline) ~ x), df; horizon=5)
println("Anchored log IRF: ", round.(coefpath(lp_anchor_log), digits=4))
```

## Summary

| Term | Formula | Computes |
|------|---------|----------|
| `leads(y)` | ``y_{t+h}`` | Forward-looking response |
| `cumul(y)` | ``\sum_{j=0}^h y_{t+j}`` | Cumulative sum |
| `anchor(y, z)` | ``y_{t+h} - z_t`` | Anchored deviation |
| `leads(y)\|z` | ``y_{t+h} - z_t`` | Pipe syntax for anchor |
| `lag(x, n)` | ``x_{t-n}`` | Lagged regressor |
| `lags(x, n)` | ``[x_{t-1}, \ldots, x_{t-n}]`` | Multiple lags |
