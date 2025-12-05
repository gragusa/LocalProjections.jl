# LocalProjections.jl

[![Build Status](https://github.com/USERNAME/LocalProjections.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/USERNAME/LocalProjections.jl/actions/workflows/CI.yml?query=branch%3Amain)

Estimate local projection impulse response functions using horizon-specific linear regressions.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/USERNAME/LocalProjections.jl")
```

## Example

```julia
using LocalProjections, DataFrames, StatsModels, CovarianceMatrices, Plots

# Generate sample data
n = 200
shock = randn(n)
y = cumsum(shock) + 0.5 * randn(n)  # y responds to cumulative shocks
df = DataFrame(y = y, shock = shock)

# Estimate local projections: y_{t+h} = α + β * shock_t + ε
# The coefficient β at each horizon h gives the impulse response
lp_result = lp(@formula(leads(y) ~ shock), df; horizon=20)

# Compute HAC-robust standard errors (Newey-West)
cov_result = vcov(Bartlett{NeweyWest}(), lp_result)

# Extract the impulse response function
irf = coefpath(lp_result; term=:shock)
se = stderror(cov_result; term=:shock)

# Plot IRF with 95% confidence bands
plot(lp_result, cov_result; term=:shock, levels=[0.95])
```

## Features

### Standard Local Projections
Forward-looking response at each horizon h:
```julia
lp(@formula(leads(y) ~ x + lag(z, 4)), df; horizon=12)
```

### Cumulative Responses
Accumulated effect from t to t+h:
```julia
lp(@formula(cumul(y) ~ x + lag(x, 4)), df; horizon=12)
```

### Anchored Responses
Deviation from a baseline variable (z stays fixed at time t):
```julia
# Pipe syntax
lp(@formula(leads(y)|z ~ x), df; horizon=12)

# Function syntax (equivalent)
lp(@formula(anchor(y, z) ~ x), df; horizon=12)
```

### Nested Transformations
Combine transformations as needed:
```julia
lp(@formula(cumul(log(y)) ~ x), df; horizon=12)
lp(@formula(leads(log(y))|log(baseline) ~ shock), df; horizon=12)
```

### Robust Standard Errors
Use any estimator from CovarianceMatrices.jl:
```julia
vcov(HC1(), lp_result)                    # Heteroskedasticity-robust
vcov(Bartlett{NeweyWest}(), lp_result)    # HAC with Newey-West bandwidth
vcov(Parzen{Andrews}(), lp_result)        # HAC with Andrews bandwidth
```

## API Reference

| Function | Description |
|----------|-------------|
| `lp(formula, data; horizon, shock)` | Estimate local projections |
| `coefpath(lp; term)` | Extract coefficient path across horizons |
| `vcov(estimator, lp)` | Compute robust variance-covariance |
| `stderror(cov; term)` | Extract standard errors for a term |
| `plot(lp, cov; term, levels)` | Plot IRF with confidence bands |

## License

MIT License - see [LICENSE](LICENSE) for details.
