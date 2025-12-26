# Tutorial 1: Getting Started with Local Projections

This tutorial covers basic local projection estimation without transformations.

## What are Local Projections?

Local projections estimate impulse response functions by running separate
regressions at each forecast horizon ``h``:

```math
y_{t+h} = \alpha_h + \beta_h x_t + \gamma_h \text{controls}_t + \varepsilon_{t+h}
```

The sequence ``\{\beta_0, \beta_1, \ldots, \beta_H\}`` traces out the response of ``y`` to a shock in ``x``.

## Basic Example

```@example basic
using LocalProjections
using DataFrames
using StatsModels: @formula
using Random

Random.seed!(42)

# Simulate AR(1) process with shock
n = 200
shock = randn(n)
y = zeros(n)
for t in 2:n
    y[t] = 0.7 * y[t-1] + shock[t]
end

df = DataFrame(y = y, shock = shock)
nothing # hide
```

### Estimate Local Projections

```@example basic
lp_result = lp(@formula(leads(y) ~ shock), df; horizon=10)
```

### Extract the Impulse Response

```@example basic
irf = coefpath(lp_result)
println("IRF coefficients: ", round.(irf, digits=3))
```

## With Control Variables

You can include lagged controls to improve efficiency:

```@example basic
lp_controlled = lp(@formula(leads(y) ~ shock + lag(y, 1) + lag(y, 2)), df; horizon=10)
irf_controlled = coefpath(lp_controlled)
println("Controlled IRF: ", round.(irf_controlled, digits=3))
```

## Understanding the Output

The `LocalProjection` object contains:
- `models`: Vector of fitted OLS models (one per horizon)
- `horizon`: Maximum horizon
- `response`: Response variable name
- `shock`: Shock variable name
- `coef_names`: Coefficient names for each horizon

```@example basic
println("Response: ", lp_result.response)
println("Shock: ", lp_result.shock)
println("Horizons: 0 to ", lp_result.horizon)
println("Coefficients: ", lp_result.coef_names[1])
```

## Specifying the Shock Variable

By default, `lp()` uses the first RHS coefficient (after intercept) as the shock.
You can specify a different variable:

```@example basic
lp_explicit = lp(@formula(leads(y) ~ shock + lag(y, 1)), df;
                 horizon=5, shock=:shock)
nothing # hide
```

!!! note
    The `shock` parameter must match the coefficient name, not the variable name.
    For `lag(x)`, the coefficient name is `"x_lag1"`.
