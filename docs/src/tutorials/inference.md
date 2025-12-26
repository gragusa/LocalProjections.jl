# Tutorial 3: Inference and Plotting

This tutorial covers robust variance estimation and IRF visualization.

## Setup

```@example infer
using LocalProjections, DataFrames, Random
using StatsModels: @formula
using CovarianceMatrices: HC1, HC3, Bartlett, Parzen, NeweyWest
Random.seed!(456)

n = 200
df = DataFrame(
    y = cumsum(randn(n)),
    x = randn(n)
)

lp_result = lp(@formula(leads(y) ~ x), df; horizon=12)
nothing # hide
```

## Robust Standard Errors

LocalProjections.jl integrates with CovarianceMatrices.jl for robust inference.

### Heteroskedasticity-Robust (HC)

```@example infer
# HC1 (default)
cov_hc1 = vcov(HC1(), lp_result)
se_hc1 = stderror(cov_hc1; term=:x)
println("HC1 standard errors: ", round.(se_hc1, digits=4))

# HC3 (more conservative)
cov_hc3 = vcov(HC3(), lp_result)
se_hc3 = stderror(cov_hc3; term=:x)
println("HC3 standard errors: ", round.(se_hc3, digits=4))
```

### HAC (Newey-West style)

For time series with autocorrelated errors:

```@example infer
# Bartlett kernel with Newey-West bandwidth
cov_nw = vcov(Bartlett{NeweyWest}(), lp_result)
se_nw = stderror(cov_nw; term=:x)
println("Newey-West SEs: ", round.(se_nw, digits=4))

# Parzen kernel
cov_parzen = vcov(Parzen{NeweyWest}(), lp_result)
se_parzen = stderror(cov_parzen; term=:x)
println("Parzen SEs: ", round.(se_parzen, digits=4))
```

### Fixed Bandwidth HAC

```@example infer
# Fixed bandwidth of 5
cov_fixed = vcov(Bartlett(5), lp_result)
se_fixed = stderror(cov_fixed; term=:x)
println("Fixed bandwidth SEs: ", round.(se_fixed, digits=4))
```

## Summary Tables with `summarize()`

Get a DataFrame with coefficients, standard errors, and confidence intervals:

```@example infer
summary_df = summarize(lp_result, cov_hc1)
println(summary_df)
```

### Convenience syntax

Pass the estimator directly instead of computing vcov separately:

```@example infer
summary_df = summarize(lp_result, HC1())
println(summary_df)
```

### Scaling for Percentages

```@example infer
summary_pct = summarize(lp_result, HC1(); scale=100, level=0.90)
println(summary_pct)
```

### Different Confidence Levels

```@example infer
# 68% CI (approx. 1 SE)
summary_68 = summarize(lp_result, HC1(); level=0.68)
println("68% CI width: ", round.(summary_68.upper .- summary_68.lower, digits=3))

# 99% CI
summary_99 = summarize(lp_result, HC1(); level=0.99)
println("99% CI width: ", round.(summary_99.upper .- summary_99.lower, digits=3))
```

## Plotting IRFs

Using Plots.jl with our plot recipes:

```@example infer
using Plots

# Basic plot with 95% CI
plot(lp_result, HC1())
```

### Customizing Plots

```@example infer
# Multiple confidence levels, scaled
plot(lp_result, HC1();
     levels=[0.68, 0.95],
     irf_scale=1,
     title="Impulse Response Function",
     xlabel="Horizon",
     ylabel="Response",
     legend=:topright)
```

### Comparing Variance Estimators

```@example infer
p1 = plot(lp_result, HC1(); title="HC1", legend=false)
p2 = plot(lp_result, HC3(); title="HC3", legend=false)
p3 = plot(lp_result, Bartlett{NeweyWest}(); title="Newey-West", legend=false)

plot(p1, p2, p3, layout=(1,3), size=(900, 300))
```

## Available Variance Estimators

| Estimator | Description |
|-----------|-------------|
| `HC0()` - `HC5()` | Heteroskedasticity-consistent (White) |
| `Bartlett{NeweyWest}()` | HAC with Newey-West bandwidth |
| `Bartlett(bw)` | HAC with fixed bandwidth |
| `Parzen{NeweyWest}()` | HAC with Parzen kernel |
| `QuadraticSpectral{Andrews}()` | HAC with QS kernel |

See [CovarianceMatrices.jl documentation](https://gragusa.github.io/CovarianceMatrices.jl/stable/) for more options.
