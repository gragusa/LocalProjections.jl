module LocalProjections

export LocalProjection, LocalProjectionCovariance, IRFSummary
export lp, coefpath, stderror, vcov, summarize
export lag, lead, cumul, CumulTerm, lags, leads, LeadTerm, anchor, AnchorTerm

using DataFrames
using PrettyTables: pretty_table, TextHighlighter, TextTableFormat,
                    text_table_borders__unicode_rounded, fmt__round, @crayon_str
using Tables
using StatsModels
using StatsModels: AbstractTerm, Term, FunctionTerm, ConstantTerm, FormulaTerm,
                   ContinuousTerm, coefnames
using Regress
using Regress: OLSMatrixEstimator, ols
using CovarianceMatrices
using Statistics
using Distributions
using RecipesBase
using ShiftedArrays: lag, lead
using StatsBase
using Missings
using TestItems

# ============================================================================
# Helper Functions for Common Patterns
# ============================================================================

"""
    _parse_unary_binary_args(t::FunctionTerm, func_name::String, default_value)

Parse arguments from a FunctionTerm that accepts 1 or 2 arguments.
Returns (term, param_value) where param_value is either the provided value or default_value.
"""
function _parse_unary_binary_args(t::FunctionTerm, func_name::String, default_value)
    if length(t.args) == 1
        return (first(t.args), default_value)
    elseif length(t.args) == 2
        term, param_arg = t.args
        (param_arg isa ConstantTerm) ||
            throw(ArgumentError("$func_name parameter must be a number (got $param_arg)"))
        return (term, param_arg.n)
    else
        throw(ArgumentError("$func_name() requires 1 or 2 arguments"))
    end
end

"""
    _extract_single_column(cols, term_name::String="term")

Extract a single column from a matrix or vector, throwing an error if multiple columns.
Returns a Vector (using vec() for matrices).
"""
function _extract_single_column(cols, term_name::String = "term")
    if cols isa AbstractMatrix
        size(cols, 2) == 1 ||
            throw(ArgumentError("$term_name must be a single variable, got $(size(cols, 2)) columns"))
        return vec(cols)
    end
    return cols
end

"""
    _check_horizon_provided(horizon::Union{Int,Nothing}, func_name::String)

Check that horizon is not nothing (for standalone formula usage).
Throws error if horizon is nothing.
"""
function _check_horizon_provided(horizon::Union{Int, Nothing}, func_name::String)
    horizon === nothing &&
        throw(ArgumentError("$func_name() without explicit horizon can only be used in lp() context"))
    return horizon
end

"""
    _termvars_unary(t::FunctionTerm)

Extract termvars from a unary function term (extracts from first argument).
"""
function _termvars_unary(t::FunctionTerm)
    length(t.args) >= 1 && return StatsModels.termvars(t.args[1])
    return Symbol[]
end

# Add termvars support for lag/lead from ShiftedArrays (used in RHS formulas)
StatsModels.termvars(t::FunctionTerm{typeof(lag)}) = _termvars_unary(t)
StatsModels.termvars(t::FunctionTerm{typeof(lead)}) = _termvars_unary(t)

# Add termvars support for StatsModels.LeadLagTerm (created after apply_schema)
StatsModels.termvars(t::StatsModels.LeadLagTerm) = StatsModels.termvars(t.term)

"""
    _unwrap_lhs(lhs_term)

Unwrap LHS term to determine if it's anchored, cumulative, or leads.
Returns (is_anchor, is_cumul, is_leads, anchor_term, cumul_term, leads_term).
"""
function _unwrap_lhs(lhs_term)
    if lhs_term isa AnchorTerm
        inner = lhs_term.response
        is_cumul = inner isa CumulTerm
        is_leads = inner isa LeadTerm || !is_cumul  # Default to leads
        return (true, is_cumul, is_leads, lhs_term,
            is_cumul ? inner : nothing,
            is_leads && inner isa LeadTerm ? inner : nothing)
    else
        return (false, lhs_term isa CumulTerm, lhs_term isa LeadTerm,
            nothing, lhs_term isa CumulTerm ? lhs_term : nothing,
            lhs_term isa LeadTerm ? lhs_term : nothing)
    end
end

"""
    _extract_single_response(term, context::String)

Extract a single response variable from a term, throwing an error if multiple variables.
Returns the Symbol of the single base variable.
"""
function _extract_single_response(term, context::String)::Symbol
    vars = _extract_base_variables(term)
    length(vars) == 1 ||
        throw(ArgumentError("$context must reference a single base variable"))
    return vars[1]
end

"""
    _build_lhs_for_horizon(h::Int, is_anchor, is_cumul, is_leads, anchor_term, cumul_term, leads_term)

Build the LHS term for a specific horizon h, handling anchor/cumul/leads combinations.
"""
function _build_lhs_for_horizon(h::Int, is_anchor, is_cumul, is_leads,
        anchor_term, cumul_term, leads_term)
    if is_anchor
        inner = if is_cumul && cumul_term !== nothing
            CumulTerm{typeof(cumul_term.term), typeof(cumul)}(cumul_term.term, h)
        elseif is_leads && leads_term !== nothing
            LeadTerm{typeof(leads_term.term), typeof(leads)}(leads_term.term, h)
        else
            LeadTerm{typeof(anchor_term.response), typeof(leads)}(anchor_term.response, h)
        end
        return AnchorTerm{typeof(inner), typeof(anchor_term.anchor), typeof(anchor)}(
            inner, anchor_term.anchor, 0)
    elseif is_cumul
        return CumulTerm{typeof(cumul_term.term), typeof(cumul)}(cumul_term.term, h)
    elseif is_leads
        return LeadTerm{typeof(leads_term.term), typeof(leads)}(leads_term.term, h)
    else
        throw(ArgumentError("Invalid LHS term type"))
    end
end

"""
    lags(term, n)

Create multiple lag columns from 1 to n. Used in formulas like @formula(y ~ lags(x, 5))
to create a matrix with lag(x,1), lag(x,2), ..., lag(x,5).
"""
lags(t::T, n::Int) where {T <: AbstractTerm} = LagTerm{T, typeof(lags)}(t, n)

# Struct for behavior
struct LagTerm{T <: AbstractTerm, F <: typeof(lags)} <: AbstractTerm
    term::T
    nsteps::Int
end

StatsModels.terms(t::LagTerm) = (t.term,)

function StatsModels.apply_schema(
        t::FunctionTerm{F}, sch::StatsModels.Schema, ctx::Type) where {F <: typeof(lags)}
    term, nsteps = _parse_unary_binary_args(t, "lags", 1)
    term = apply_schema(term, sch, ctx)
    return LagTerm{typeof(term), F}(term, nsteps)
end

function StatsModels.apply_schema(t::LagTerm{T, F}, sch::StatsModels.Schema, ctx::Type) where {
        T, F}
    term = apply_schema(t.term, sch, ctx)
    LagTerm{typeof(term), F}(term, t.nsteps)
end

# modelcols: Create matrix with nsteps columns, each being lag(x, i) for i=1:nsteps
function StatsModels.modelcols(ll::LagTerm{<:Any, F}, d::Tables.ColumnTable) where {F}
    original_cols = StatsModels.modelcols(ll.term, d)
    n = length(original_cols)
    nsteps = ll.nsteps

    # Create matrix with nsteps columns
    # Each column i contains lag(original_cols, i)
    result = Matrix{eltype(original_cols)}(undef, n, nsteps)

    for i in 1:nsteps
        result[:, i] = lag(original_cols, i, default = NaN)
    end

    return result
end

# width: Return number of columns (nsteps)
StatsModels.width(ll::LagTerm) = ll.nsteps

# show: Display the term
function Base.show(io::IO, ll::LagTerm{<:Any, F}) where {F}
    opname = string(nameof(F.instance))
    print(io, "$opname($(ll.term), $(ll.nsteps))")
end

# coefnames: Return nsteps coefficient names
function StatsModels.coefnames(ll::LagTerm{<:Any, F}) where {F}
    opname = string(nameof(F.instance))
    base_names = StatsModels.coefnames(ll.term)
    # Create names like "x_lag1", "x_lag2", ..., "x_lagn"
    return [base_names[1] * "_lag$i" for i in 1:ll.nsteps]
end

# ============================================================================
# Cumulative Sum Term (cumul)
# This implements cumul(y) for cumulative impulse responses
# Supports nested transforms like cumul(log(y))
# ============================================================================

"""
    cumul(term)
    cumul(term, horizon)

Create cumulative sum term. Used in formulas like:
- `@formula(cumul(y) ~ x)` - horizon determined by lp() context
- `@formula(cumul(y, 5) ~ x)` - explicit horizon for standalone use
- `@formula(cumul(log(y)) ~ x)` - supports nested transformations
"""
cumul(t::T) where {T <: AbstractTerm} = CumulTerm{T, typeof(cumul)}(t, nothing)
cumul(t::T, h::Int) where {T <: AbstractTerm} = CumulTerm{T, typeof(cumul)}(t, h)

# termvars: Extract variables from cumul() for schema creation
StatsModels.termvars(t::FunctionTerm{typeof(cumul)}) = _termvars_unary(t)

# Struct for cumulative sum term
struct CumulTerm{T <: AbstractTerm, F <: typeof(cumul)} <: AbstractTerm
    term::T                      # The term to cumulate (can be nested like log(y))
    horizon::Union{Int, Nothing}  # nothing in lp() context, Int for standalone
end

StatsModels.terms(t::CumulTerm) = (t.term,)

function StatsModels.apply_schema(
        t::FunctionTerm{F}, sch::StatsModels.Schema, ctx::Type) where {F <: typeof(cumul)}
    term, horizon = _parse_unary_binary_args(t, "cumul", nothing)
    term = StatsModels.apply_schema(term, sch, ctx)
    return CumulTerm{typeof(term), F}(term, horizon)
end

function StatsModels.apply_schema(t::CumulTerm{T, F}, sch::StatsModels.Schema, ctx::Type) where {
        T, F}
    term = StatsModels.apply_schema(t.term, sch, ctx)
    CumulTerm{typeof(term), F}(term, t.horizon)
end

# modelcols: Apply cumulative sum transformation
# Note: In lp() context, horizon is nothing and will be handled specially
function StatsModels.modelcols(ct::CumulTerm{<:Any, F}, d::Tables.ColumnTable) where {F}
    _check_horizon_provided(ct.horizon, "cumul")
    original_cols = StatsModels.modelcols(ct.term, d)
    original_cols = _extract_single_column(original_cols, "cumul() response")
    return _create_cumulative(original_cols, ct.horizon)
end

# width: Return number of columns (always 1 for cumul)
StatsModels.width(ct::CumulTerm) = 1

# show: Display the term
function Base.show(io::IO, ct::CumulTerm{<:Any, F}) where {F}
    if ct.horizon === nothing
        print(io, "cumul($(ct.term))")
    else
        print(io, "cumul($(ct.term), $(ct.horizon))")
    end
end

# coefnames: Return coefficient name
function StatsModels.coefnames(ct::CumulTerm{<:Any, F}) where {F}
    base_names = StatsModels.coefnames(ct.term)
    if ct.horizon === nothing
        return ["cumul(" * base_names[1] * ")"]
    else
        return ["cumul(" * base_names[1] * ", $(ct.horizon))"]
    end
end

# ============================================================================
# Lead Term (leads)
# This implements leads(y) for forward-looking regressions with NaN handling
# Named 'leads' to avoid conflict with ShiftedArrays.lead
# Uses our _lead_to_float64() for type stability (NaN instead of missing)
# ============================================================================

"""
    leads(term)
    leads(term, horizon)

Create lead term with NaN handling. Used in formulas like:
- `@formula(leads(y) ~ x)` - horizon determined by lp() context
- `@formula(leads(y, 3) ~ x)` - explicit horizon for standalone use
- `@formula(leads(log(y)) ~ x)` - supports nested transformations

Note: Named 'leads' to distinguish from ShiftedArrays.lead (which returns missing).
This version returns Float64 with NaN for type stability.
"""
leads(t::T) where {T <: AbstractTerm} = LeadTerm{T, typeof(leads)}(t, nothing)
leads(t::T, h::Int) where {T <: AbstractTerm} = LeadTerm{T, typeof(leads)}(t, h)

# termvars: Extract variables from leads() for schema creation
StatsModels.termvars(t::FunctionTerm{typeof(leads)}) = _termvars_unary(t)

# Struct for lead term
struct LeadTerm{T <: AbstractTerm, F <: typeof(leads)} <: AbstractTerm
    term::T                      # The term to lead (can be nested like log(y))
    horizon::Union{Int, Nothing}  # nothing in lp() context, Int for standalone
end

StatsModels.terms(t::LeadTerm) = (t.term,)

function StatsModels.apply_schema(
        t::FunctionTerm{F}, sch::StatsModels.Schema, ctx::Type) where {F <: typeof(leads)}
    term, horizon = _parse_unary_binary_args(t, "leads", nothing)
    term = StatsModels.apply_schema(term, sch, ctx)
    return LeadTerm{typeof(term), F}(term, horizon)
end

function StatsModels.apply_schema(t::LeadTerm{T, F}, sch::StatsModels.Schema, ctx::Type) where {
        T, F}
    term = StatsModels.apply_schema(t.term, sch, ctx)
    LeadTerm{typeof(term), F}(term, t.horizon)
end

# modelcols: Apply lead transformation with NaN handling
function StatsModels.modelcols(lt::LeadTerm{<:Any, F}, d::Tables.ColumnTable) where {F}
    _check_horizon_provided(lt.horizon, "leads")
    original_cols = StatsModels.modelcols(lt.term, d)
    original_cols = _extract_single_column(original_cols, "leads() response")
    return _lead_to_float64(original_cols, lt.horizon)
end

# width: Return number of columns (always 1 for leads)
StatsModels.width(lt::LeadTerm) = 1

# show: Display the term
function Base.show(io::IO, lt::LeadTerm{<:Any, F}) where {F}
    if lt.horizon === nothing
        print(io, "leads($(lt.term))")
    else
        print(io, "leads($(lt.term), $(lt.horizon))")
    end
end

# coefnames: Return coefficient name
function StatsModels.coefnames(lt::LeadTerm{<:Any, F}) where {F}
    base_names = StatsModels.coefnames(lt.term)
    if lt.horizon === nothing
        return ["leads(" * base_names[1] * ")"]
    else
        return ["leads(" * base_names[1] * ", $(lt.horizon))"]
    end
end

# ============================================================================
# Anchor Term (anchor)
# This implements anchor(y, z) for anchored responses: y_{t+h} - z_t
# The anchor variable z stays fixed at time t while y shifts forward to t+h
# ============================================================================

"""
    anchor(response_term, anchor_term)
    anchor(response_term, anchor_term, horizon)

Create anchored response term for local projections. Used in formulas like:
- `@formula(anchor(y, z) ~ x)` - horizon determined by lp() context
- `@formula(anchor(y, z, 5) ~ x)` - explicit horizon for standalone use
- `@formula(anchor(log(y), z) ~ x)` - supports nested transformations on response

Computes y_{t+h} - z_t where:
- y is the response variable (can be transformed)
- z is the anchor variable (stays at time t)
- h is the horizon (0, 1, 2, ...)

At h=0: returns y_t - z_t
At h=1: returns y_{t+1} - z_t
At h=2: returns y_{t+2} - z_t, etc.
"""
function anchor(response::T, anchor_var::S) where {T <: AbstractTerm, S <: AbstractTerm}
    AnchorTerm{T, S, typeof(anchor)}(response, anchor_var, nothing)
end
function anchor(response::T, anchor_var::S, h::Int) where {
        T <: AbstractTerm, S <: AbstractTerm}
    AnchorTerm{T, S, typeof(anchor)}(response, anchor_var, h)
end

# termvars: Extract variables from anchor() for schema creation
function StatsModels.termvars(t::FunctionTerm{typeof(anchor)})
    length(t.args) >= 2 || return Symbol[]
    return unique(vcat(StatsModels.termvars(t.args[1]), StatsModels.termvars(t.args[2])))
end

# Struct for anchored response term
struct AnchorTerm{T <: AbstractTerm, S <: AbstractTerm, F <: typeof(anchor)} <: AbstractTerm
    response::T                  # The response term (can be nested like log(y))
    anchor::S                    # The anchor term (stays at time t)
    horizon::Union{Int, Nothing}  # nothing in lp() context, Int for standalone
end

StatsModels.terms(t::AnchorTerm) = (t.response, t.anchor)

function StatsModels.apply_schema(
        t::FunctionTerm{F}, sch::StatsModels.Schema, ctx::Type) where {F <: typeof(anchor)}
    if length(t.args) == 2  # anchor(response, anchor_var) - horizon from context
        response, anchor_var = t.args
        horizon = nothing
    elseif length(t.args) == 3  # anchor(response, anchor_var, horizon)
        response, anchor_var, h_arg = t.args
        (h_arg isa ConstantTerm) ||
            throw(ArgumentError("anchor horizon must be a number (got $h_arg)"))
        horizon = h_arg.n
    else
        throw(ArgumentError("anchor() requires 2 or 3 arguments"))
    end

    response = StatsModels.apply_schema(response, sch, ctx)
    anchor_var = StatsModels.apply_schema(anchor_var, sch, ctx)
    return AnchorTerm{typeof(response), typeof(anchor_var), F}(response, anchor_var, horizon)
end

function StatsModels.apply_schema(t::AnchorTerm{T, S, F}, sch::StatsModels.Schema, ctx::Type) where {
        T, S, F}
    response = StatsModels.apply_schema(t.response, sch, ctx)
    anchor_var = StatsModels.apply_schema(t.anchor, sch, ctx)
    AnchorTerm{typeof(response), typeof(anchor_var), F}(response, anchor_var, t.horizon)
end

# modelcols: Apply anchored transformation (y_{t+h} - z_t)
function StatsModels.modelcols(at::AnchorTerm{<:Any, <:Any, F}, d::Tables.ColumnTable) where {F}
    _check_horizon_provided(at.horizon, "anchor")
    response_cols = StatsModels.modelcols(at.response, d)
    anchor_cols = StatsModels.modelcols(at.anchor, d)
    response_cols = _extract_single_column(response_cols, "anchor() response")
    anchor_cols = _extract_single_column(anchor_cols, "anchor() anchor variable")
    return _create_anchored(response_cols, anchor_cols, at.horizon)
end

# width: Return number of columns (always 1 for anchor)
StatsModels.width(at::AnchorTerm) = 1

# show: Display the term
function Base.show(io::IO, at::AnchorTerm{<:Any, <:Any, F}) where {F}
    if at.horizon === nothing
        print(io, "anchor($(at.response), $(at.anchor))")
    else
        print(io, "anchor($(at.response), $(at.anchor), $(at.horizon))")
    end
end

# coefnames: Return coefficient name
function StatsModels.coefnames(at::AnchorTerm{<:Any, <:Any, F}) where {F}
    response_names = StatsModels.coefnames(at.response)
    anchor_names = StatsModels.coefnames(at.anchor)
    if at.horizon === nothing
        return ["anchor(" * response_names[1] * ", " * anchor_names[1] * ")"]
    else
        return ["anchor(" * response_names[1] * ", " * anchor_names[1] * ", $(at.horizon))"]
    end
end

# termvars: Return variables used in anchor term
function StatsModels.termvars(at::AnchorTerm)
    unique(vcat(StatsModels.termvars(at.response), StatsModels.termvars(at.anchor)))
end

# ============================================================================
# Pipe Operator (|) for Anchored Response Syntax
# Intercept | during schema application to create AnchorTerm
# ============================================================================

"""
    termvars for FunctionTerm{typeof(|)}

Extract variable names from pipe operator for schema creation.
This tells StatsModels which variables are referenced so it can
properly determine their types (continuous vs categorical).
"""
function StatsModels.termvars(t::FunctionTerm{typeof(|)})
    if length(t.args) != 2
        return Symbol[]
    end
    lhs, rhs = t.args
    return unique(vcat(StatsModels.termvars(lhs), StatsModels.termvars(rhs)))
end

"""
    apply_schema for FunctionTerm{typeof(|)}

Intercepts pipe operator in formulas to create AnchorTerm for anchored responses.
Enables syntax like:
- `@formula(leads(y)|z ~ x)` - equivalent to `@formula(anchor(y, z) ~ x)`
- `@formula(cumul(y)|z ~ x)` - cumulative response anchored to z

The pipe creates an AnchorTerm where:
- lhs is the response term (can be transformed: leads(y), cumul(y), log(y), etc.)
- rhs is the anchor term (stays fixed at time t)

# Examples
```julia
# Standard: y_{t+h}
lp(@formula(leads(y) ~ x), df; horizon=12)

# Anchored: y_{t+h} - z_t (pipe syntax)
lp(@formula(leads(y)|z ~ x), df; horizon=12)

```
"""
function StatsModels.apply_schema(t::FunctionTerm{typeof(|)}, sch::StatsModels.Schema, ctx::Type)
    # The pipe operator in formulas: lhs | rhs
    # Convert to AnchorTerm(lhs, rhs, nothing)
    if length(t.args) != 2
        throw(ArgumentError("Pipe operator | requires exactly 2 arguments (got $(length(t.args)))"))
    end

    lhs, rhs = t.args

    # Apply schema to both sides
    lhs_term = StatsModels.apply_schema(lhs, sch, ctx)
    rhs_term = StatsModels.apply_schema(rhs, sch, ctx)

    # Return AnchorTerm
    return AnchorTerm{typeof(lhs_term), typeof(rhs_term), typeof(anchor)}(lhs_term, rhs_term, nothing)
end

"""
    LocalProjection

Stack of horizon-specific OLS models produced by [`lp`](@ref).
"""
struct LocalProjection{M <: OLSMatrixEstimator}
    models::Vector{M}
    horizon::Int
    response::Symbol
    shock::Symbol
    base_formula::FormulaTerm
    coef_names::Vector{Vector{String}}  # Coefficient names for each horizon
end

"""
    coefnames(lp::LocalProjection)

Return the coefficient names for the local projection models.
Since all horizons share the same RHS, returns the coefficient names from horizon 0.
"""
StatsModels.coefnames(lp::LocalProjection) = lp.coef_names[1]

"""
    coefnames(lp::LocalProjection, h::Int)

Return the coefficient names for horizon `h` (0-indexed).
"""
StatsModels.coefnames(lp::LocalProjection, h::Int) = lp.coef_names[h + 1]

"""
    model_summary(lp::LocalProjection, h::Int)

Print a summary of the model at horizon `h` (0-indexed) with proper coefficient names.
"""
function model_summary(lp::LocalProjection, h::Int)
    m = lp.models[h + 1]
    names = lp.coef_names[h + 1]
    coefs = coef(m)
    se = stderror(HC1(), m)

    println("Local Projection at horizon $h")
    println("=" ^ 60)
    println("Response: $(lp.response)")
    println("Observations: $(Int(nobs(m)))")
    println("R²: $(round(r2(m), digits=4))")
    println()
    println(rpad("Coefficient", 25), rpad("Estimate", 15), "Std. Error")
    println("-" ^ 60)
    for (name, c, s) in zip(names, coefs, se)
        println(rpad(name, 25), rpad(round(c, digits = 6), 15), round(s, digits = 6))
    end
end

function Base.show(io::IO, lp::LocalProjection)
    print(io, "LocalProjection(horizon=0:$(lp.horizon), response=$(lp.response), shock=$(lp.shock))")
end

function Base.show(io::IO, ::MIME"text/plain", lp::LocalProjection)
    println(io, "LocalProjection")
    println(io, "  Response:   $(lp.response)")
    println(io, "  Shock:      $(lp.shock)")
    println(io, "  Horizon:    0:$(lp.horizon)")
    println(io, "  Formula:    $(lp.base_formula)")
    println(io, "  Coef names: $(lp.coef_names[1])")
end

"""
    diagnose_vcov(lp::LocalProjection, h::Int=0)

Run diagnostics on the model at horizon `h` to help debug vcov issues.
Checks if different HAC estimators produce identical results (which would indicate a problem).
"""
function diagnose_vcov(lp::LocalProjection, h::Int = 0)
    m = lp.models[h + 1]
    println("Diagnostic for LocalProjection at horizon $h")
    println("=" ^ 60)
    println("Observations: $(Int(nobs(m)))")
    println("Parameters:   $(length(coef(m)))")
    println("Model matrix size: $(size(modelmatrix(m)))")
    println()

    # Compute different estimators
    se_hc1 = CovarianceMatrices.stderror(HC1(), m)
    se_bart_nw = CovarianceMatrices.stderror(Bartlett{NeweyWest}(), m)
    se_bart_an = CovarianceMatrices.stderror(Bartlett{Andrews}(), m)
    se_parzen_nw = CovarianceMatrices.stderror(Parzen{NeweyWest}(), m)

    println("Standard errors by estimator:")
    println("  HC1:           $(se_hc1)")
    println("  Bartlett NW:   $(se_bart_nw)")
    println("  Bartlett AN:   $(se_bart_an)")
    println("  Parzen NW:     $(se_parzen_nw)")
    println()

    # Check for identical results
    if se_hc1 ≈ se_bart_nw
        println("⚠️  WARNING: HC1 ≈ Bartlett NW (identical results)")
        println("    This may indicate bandwidth = 0 or numerical issues")
    end
    if se_bart_nw ≈ se_bart_an ≈ se_parzen_nw
        println("⚠️  WARNING: All HAC estimators produce identical results")
        println("    This suggests bandwidth selection is returning 0")
    end

    # Check residuals autocorrelation
    resid = residuals(m)
    if length(resid) > 1
        autocorr1 = cor(resid[1:(end - 1)], resid[2:end])
        println("Residual autocorrelation (lag 1): $(round(autocorr1, digits=4))")
    end
end

"""
    LocalProjectionCovariance

Diagonal covariance entries term-by-term across horizons.
"""
struct LocalProjectionCovariance{E}
    estimator::E
    variances::Dict{Symbol, Vector{Float64}}
    horizon::Int
end

"""
    _lead_to_float64(y::AbstractVector, h::Int) -> Vector{Float64}

Apply lead transformation and convert to Float64 with NaN for boundaries.
This maintains type stability by always returning Vector{Float64}.

# Arguments
- `y::AbstractVector`: Input vector
- `h::Int`: Lead horizon (h=0 returns y unchanged)

# Returns
- `Vector{Float64}`: Type-stable Float64 vector with NaN for missing/boundary values
"""
function _lead_to_float64(y::AbstractVector, h::Int)::Vector{Float64}
    n = length(y)
    lead(y, h, default = NaN)
end

"""
    _create_cumulative(y::AbstractVector, h::Int) -> Vector{Float64}

Helper function to compute cumulative sum from t to t+h for each observation.
Returns sum_{j=0}^{h} y_{t+j}.

At h=0, returns y itself (converted to Float64).
At h=1, returns y_t + y_{t+1}.
At h=2, returns y_t + y_{t+1} + y_{t+2}, etc.

NaN values are returned when:
- The sum cannot be computed (at the end of the series)
- Any component y_{t+j} is missing or NaN

# Returns
- `Vector{Float64}`: Type-stable Float64 vector with NaN at boundaries
"""
function _create_cumulative(y::AbstractVector, h::Int)::Vector{Float64}
    n = length(y)

    # Convert input to Float64, replacing missing with NaN
    # This handles both pure Float64 vectors and Union{Missing, Float64} vectors
    y_float = map(v -> ismissing(v) ? NaN : Float64(v), y)

    # Fast path for h=0
    h == 0 && return y_float

    result = Vector{Float64}(undef, n)
    @inbounds for t in 1:n
        if t + h > n
            result[t] = NaN
        else
            # Sum from t to t+h (inclusive, so h+1 values total)
            cumsum_val = 0.0
            all_valid = true
            for j in 0:h
                val = y_float[t + j]
                if isnan(val)
                    all_valid = false
                    break
                end
                cumsum_val += val
            end
            result[t] = all_valid ? cumsum_val : NaN
        end
    end

    return result
end

"""
    _create_anchored(y::AbstractVector, z::AbstractVector, h::Int) -> Vector{Float64}

Helper function to compute anchored response: y_{t+h} - z_t for each observation.

At h=0, returns y_t - z_t (both at time t).
At h=1, returns y_{t+1} - z_t (y shifted forward, z stays at t).
At h=2, returns y_{t+2} - z_t, etc.

The key difference from standard lead:
- Standard lead: y_{t+h} evolves freely
- Anchored: y_{t+h} - z_t measures deviation from anchor z_t

NaN values are returned when:
- Cannot compute lead (at the end of the series)
- Either y_{t+h} or z_t is missing or NaN

# Returns
- `Vector{Float64}`: Type-stable Float64 vector with NaN at boundaries
"""
function _create_anchored(y::AbstractVector, z::AbstractVector, h::Int)::Vector{Float64}
    n = length(y)
    length(z) == n || throw(ArgumentError("y and z must have same length"))
    h > n && throw(ArgumentError("horizon h=$h is too large for series of length $n"))
    return lead(y, h, default = NaN) .- z
end

"""

(term::AbstractTerm)

Recursively extract base variable names from a term, stripping away
function transformations like lag(), lead(), cumul().

Returns a vector of Symbol representing the raw variables referenced.

# Examples
```julia
_extract_base_variables(Term(:x))                    # [:x]
_extract_base_variables(FunctionTerm(lag, [:x, 4])) # [:x]  (strips lag)
```
"""
# function _extract_base_variables(term::AbstractTerm)::Vector{Symbol}
#     return StatsModels.termvars(term)  # Fallback
# end

# Specialized methods for specific term types
function _extract_base_variables(t::Union{
        Term, StatsModels.ContinuousTerm, StatsModels.CategoricalTerm})
    [t.sym]
end
_extract_base_variables(t::CumulTerm) = _extract_base_variables(t.term)
_extract_base_variables(t::LeadTerm) = _extract_base_variables(t.term)
function _extract_base_variables(t::AnchorTerm)
    unique(vcat(_extract_base_variables(t.response), _extract_base_variables(t.anchor)))
end

function _extract_base_variables(ft::FunctionTerm)
    # For lag/lead/cumul/leads, extract the base variable (first argument)
    if ft.f in (lag, lead, cumul, leads)
        return _extract_base_variables(ft.args[1])
    elseif ft.f === anchor
        # For anchor, extract both response and anchor variables
        length(ft.args) >= 2 || return StatsModels.termvars(ft)
        return unique(vcat(_extract_base_variables(ft.args[1]),
            _extract_base_variables(ft.args[2])))
    else
        # For other functions, use termvars
        return StatsModels.termvars(ft)
    end
end

function _extract_base_variables(terms::Tuple)
    # For RHS tuple of terms
    all_vars = Symbol[]
    for t in terms
        append!(all_vars, _extract_base_variables(t))
    end
    return unique(all_vars)
end

function _extract_base_variables(ft::FormulaTerm)
    unique(vcat(
        _extract_base_variables(ft.lhs),
        _extract_base_variables(ft.rhs)
    ))
end

# InteractionTerm and other composite terms
function _extract_base_variables(t::StatsModels.InteractionTerm)
    all_vars = Symbol[]
    for component in t.terms
        append!(all_vars, _extract_base_variables(component))
    end
    return unique(all_vars)
end

"""
    lp(formula, data; horizon, shock=nothing)

Estimate local projections implied by `formula` up to the supplied `horizon`.
`shock` selects the coefficient path of interest (defaults to the first RHS term).
"""
function lp(formula::FormulaTerm, data::AbstractDataFrame;
        horizon::Integer, shock::Union{Symbol, Nothing} = nothing)
    horizon < 0 && throw(ArgumentError("horizon must be non-negative"))
    df_base = DataFrame(data)  # avoid mutating caller's data

    # Apply schema to formula to convert FunctionTerms to proper terms (CumulTerm, LagTerm, etc.)
    # Collect all variable names from the formula
    all_vars = StatsModels.termvars(formula)

    # Create hints to treat all variables as continuous (not categorical)
    # This prevents StatsModels from treating numeric columns with many unique values as categorical
    hints = Dict{Symbol, Any}(var => StatsModels.ContinuousTerm for var in all_vars)

    sch = StatsModels.schema(formula, df_base, hints)
    lhs_term = StatsModels.apply_schema(formula.lhs, sch, StatisticalModel)
    rhs_term = StatsModels.apply_schema(formula.rhs, sch, StatisticalModel)

    # Check if LHS is a CumulTerm (cumulative impulse response), LeadTerm (forward-looking), or AnchorTerm (anchored)
    # Note: AnchorTerm can contain LeadTerm or CumulTerm inside (from pipe syntax like leads(y)|z or cumul(y)|z)
    # If AnchorTerm contains plain term (y|z), default to leads behavior
    is_anchor, is_cumulative, is_leads, anchor_term, cumul_term,
    leads_term = _unwrap_lhs(lhs_term)

    # Extract response variable/data
    # For cumulative/leads/anchor cases, we need to extract the base variable names for Stage 1 filtering
    # but we'll evaluate the transformed term for actual calculation
    response = if is_anchor
        _extract_single_response(anchor_term.response, "anchor() response term")
    elseif is_cumulative
        _extract_single_response(cumul_term, "cumul() term")
    elseif is_leads
        _extract_single_response(leads_term, "leads() term")
    else
        throw(ArgumentError("A local projection without leads and cumulated variables does not make much sense"))
    end

    # Extract all variable names from RHS (including from function terms)
    rhs_terms = StatsModels.termvars(rhs_term)
    isempty(rhs_terms) &&
        throw(ArgumentError("formula must contain at least one regressor"))

    # ========================================================================
    # Stage 1: Remove rows with missing values in base variables
    # Extract all base variables (without transformations) from formula
    # ========================================================================
    base_vars_lhs = _extract_base_variables(formula.lhs)
    base_vars_rhs = _extract_base_variables(formula.rhs)
    base_vars = unique(vcat(base_vars_lhs, base_vars_rhs))

    # Keep only complete cases (remove rows with missing base variables)
    df_base_complete = dropmissing(df_base, base_vars, disallowmissing = true)

    # ========================================================================
    # Stage 2: Compute X matrix ONCE (RHS is constant across all horizons)
    # ========================================================================

    # Create a dummy formula with the response on LHS to compute RHS ModelFrame
    # We use the original response variable since it exists in df_base
    dummy_formula = StatsModels.FormulaTerm(StatsModels.Term(response), formula.rhs)

    # Create ModelFrame for the RHS computation (only once!)
    mf_base = StatsModels.ModelFrame(dummy_formula, df_base_complete)

    # Extract X matrix (this handles all lag/lead transformations on RHS)
    X_raw = StatsModels.modelcols(mf_base.f.rhs, mf_base.data)

    # Convert missing to NaN for type stability (handles lag(w) which returns missing)
    # This ensures X is always Float64, matching our leads/cumul which use default=NaN
    X = map(v -> ismissing(v) ? NaN : Float64(v), X_raw)

    # Store coefficient names (constant across all horizons)
    coef_names_base = coefnames(mf_base)

    # Set shock variable: use provided shock or default to first RHS coefficient
    # Note: coef_names_base includes intercept, so we need the second element (first RHS term)
    if shock === nothing
        # Default to first RHS coefficient (skip intercept which is first)
        shock_symbol = length(coef_names_base) >= 2 ? Symbol(coef_names_base[2]) :
                       Symbol(coef_names_base[1])
    else
        shock_symbol = shock
    end

    # Identify rows where X is complete (no NaN values)
    X_missing_ind = vec(all(!isnan, X, dims = 2))

    # Pre-allocate storage for models and coefficient names
    # Use Any for initial allocation; concrete type will be inferred at struct construction
    models = Vector{Any}(undef, horizon + 1)
    coef_names_vec = Vector{Vector{String}}(undef, horizon + 1)

    # ========================================================================
    # Stage 3: Estimate models for each horizon (only y_h changes)
    # ========================================================================
    for (i, h) in enumerate(0:horizon)
        # Create horizon-specific LHS term by injecting horizon
        lhs_h = _build_lhs_for_horizon(h, is_anchor, is_cumulative, is_leads,
            anchor_term, cumul_term, leads_term)

        # Extract y_h by calling modelcols directly on the constructed term
        y_h = StatsModels.modelcols(lhs_h, df_base_complete)

        # Extract response using standard StatsModels machinery
        # This automatically evaluates cumul(log(y), h), leads(log(y), h), or any nested transform!

        y_complete_rows = .!isnan.(y_h)
        complete_rows = X_missing_ind .& y_complete_rows

        if sum(complete_rows) == 0
            throw(ArgumentError("No complete observations available for horizon $h"))
        end

        # Extract complete cases using VIEWS (reduces allocations)
        y = view(y_h, complete_rows)
        x = view(X, complete_rows, :)

        # Fit model using matrix form
        model = ols(x, y)
        models[i] = model

        # Use pre-computed coefficient names (constant across horizons)
        coef_names_vec[i] = coef_names_base

        # Verify shock variable is present
        available = Symbol.(coef_names_base)
        shock_symbol in available ||
            throw(ArgumentError("shock term $(shock_symbol) not present in model for horizon $h"))
    end

    # Convert to properly typed vector for struct construction
    typed_models = [m for m in models]
    return LocalProjection(
        typed_models, horizon, response, shock_symbol, formula, coef_names_vec)
end

"""
    coefpath(lp; term = lp.shock)

Collect the coefficient path across horizons for `term`.
"""
function coefpath(lp::LocalProjection; term::Symbol = lp.shock)
    n = lp.horizon + 1
    coefficients = Vector{Float64}(undef, n)
    for (i, model) in enumerate(lp.models)
        names = lp.coef_names[i]
        idx = findfirst(==(String(term)), names)
        idx === nothing &&
            throw(ArgumentError("term $term not present in model at horizon $(i - 1)"))
        coefficients[i] = coef(model)[idx]
    end
    return coefficients
end

"""
    vcov(estimator, lp)

Compute diagonal covariance entries horizon-by-horizon using `estimator`
from `CovarianceMatrices.jl`.
"""
function vcov(estimator, lp::LocalProjection)
    n = lp.horizon + 1
    variances = Dict{Symbol, Vector{Float64}}()

    for (i, model) in enumerate(lp.models)
        cov = CovarianceMatrices.vcov(estimator, model)
        names = Symbol.(lp.coef_names[i])
        for (j, name) in enumerate(names)
            vec = get!(variances, name) do
                fill(NaN, n)
            end
            vec[i] = cov[j, j]
        end
    end

    return LocalProjectionCovariance(estimator, variances, lp.horizon)
end

"""
    stderror(cov; term)

Standard errors across horizons for `term`.
"""
function stderror(cov::LocalProjectionCovariance; term::Symbol)
    variances = get(cov.variances, term) do
        throw(ArgumentError("term $term not found in covariance object"))
    end
    return sqrt.(variances)
end

"""
    IRFSummary

Summary of impulse response function with coefficients, standard errors,
and confidence intervals. Displays with PrettyTables, highlighting
statistically significant coefficients in bold.
"""
struct IRFSummary
    term::Symbol
    level::Float64
    scale::Float64
    horizon::Vector{Int}
    coef::Vector{Float64}
    se::Vector{Float64}
    lower::Vector{Float64}
    upper::Vector{Float64}
end

# Convert to DataFrame for data access
function DataFrames.DataFrame(s::IRFSummary)
    DataFrame(
        horizon = s.horizon,
        coef = s.coef,
        se = s.se,
        lower = s.lower,
        upper = s.upper
    )
end

function Base.show(io::IO, s::IRFSummary)
    println(io, "IRFSummary(term=$(s.term), level=$(s.level), scale=$(s.scale))")
end

function Base.show(io::IO, ::MIME"text/plain", s::IRFSummary)
    # Determine significance: CI doesn't include zero
    significant = (s.lower .> 0) .| (s.upper .< 0)

    # Create data matrix for PrettyTables
    data = hcat(s.horizon, s.coef, s.se, s.lower, s.upper)

    # Format the level as percentage
    level_pct = round(Int, s.level * 100)

    # Column labels (PrettyTables v3 API)
    labels = ["Horizon", "Coef", "Std.Err.", "Lower $level_pct%", "Upper $level_pct%"]

    # Highlighters for bold significant values (PrettyTables v3 API)
    # Make coef, lower, upper bold when significant
    hl_coef = TextHighlighter(
        (d, i, j) -> j == 2 && significant[i],
        crayon"bold"
    )
    hl_lower = TextHighlighter(
        (d, i, j) -> j == 4 && significant[i],
        crayon"bold"
    )
    hl_upper = TextHighlighter(
        (d, i, j) -> j == 5 && significant[i],
        crayon"bold"
    )

    # Title
    title = "Impulse Response: $(s.term)"
    if s.scale != 1.0
        title *= " (scale=$(s.scale))"
    end

    # Table format with rounded borders
    table_fmt = TextTableFormat(borders = text_table_borders__unicode_rounded)

    # Enable color output for ANSI bold codes
    ioc = IOContext(io, :color => true)

    pretty_table(ioc, data;
        column_labels = labels,
        title = title,
        highlighters = [hl_coef, hl_lower, hl_upper],
        formatters = [fmt__round(4)],
        alignment = [:r, :r, :r, :r, :r],
        table_format = table_fmt
    )
end

"""
    summarize(lp, cov; term=lp.shock, level=0.95, scale=1.0) -> IRFSummary

Create a summary table of impulse response coefficients with standard errors
and confidence intervals. Statistically significant coefficients (where the
confidence interval excludes zero) are displayed in bold.

# Arguments
- `lp::LocalProjection`: The estimated local projection
- `cov::LocalProjectionCovariance`: Covariance object from vcov()
- `term::Symbol`: Which coefficient to summarize (default: `lp.shock`)
- `level::Real`: Confidence level for intervals (default: 0.95)
- `scale::Real`: Multiplicative scale factor (default: 1.0, use 100 for %)

# Returns
`IRFSummary` object that displays as a formatted table. Convert to DataFrame
with `DataFrame(summary)`.

# Example
```julia
lp_result = lp(@formula(leads(y) ~ x), df; horizon=12)
cov = vcov(HC1(), lp_result)
summarize(lp_result, cov; level=0.90, scale=100)
```
"""
function summarize(lp::LocalProjection, cov::LocalProjectionCovariance;
        term::Symbol = lp.shock, level::Real = 0.95, scale::Real = 1.0)
    beta = coefpath(lp; term = term) .* scale
    se = stderror(cov; term = term) .* scale
    z = quantile(Normal(), 0.5 + level / 2)
    lower = beta .- z .* se
    upper = beta .+ z .* se

    IRFSummary(term, Float64(level), Float64(scale),
        collect(0:lp.horizon), beta, se, lower, upper)
end

"""
    summarize(lp, estimator; term=lp.shock, level=0.95, scale=1.0) -> IRFSummary

Convenience method that computes vcov internally before creating summary table.

# Example
```julia
summarize(lp_result, HC1(); scale=100, level=0.90)
```
"""
function summarize(lp::LocalProjection,
        estimator::CovarianceMatrices.AbstractAsymptoticVarianceEstimator;
        term::Symbol = lp.shock, level::Real = 0.95, scale::Real = 1.0)
    cov = vcov(estimator, lp)
    summarize(lp, cov; term = term, level = level, scale = scale)
end

# ============================================================================
# Plot Recipes using RecipesBase
# ============================================================================

"""
    IRFPlot

Internal wrapper type for dispatching plot recipes on LocalProjection with covariance.
Users should call `plot(lp, cov; ...)` or `plot(lp, estimator; ...)` directly.
"""
struct IRFPlot{M <: OLSMatrixEstimator, E}
    lp::LocalProjection{M}
    cov::LocalProjectionCovariance{E}
    term::Symbol
    levels::Vector{Float64}
    irf_scale::Float64
end

"""
    plot(lp, cov; term=lp.shock, levels=[0.95], irf_scale=1.0, kwargs...)
    plot(lp, estimator; term=lp.shock, levels=[0.95], irf_scale=1.0, kwargs...)

Plot impulse response function with confidence bands.

# Arguments
- `lp::LocalProjection`: The estimated local projection
- `cov::LocalProjectionCovariance` or `estimator`: Covariance object or estimator (e.g., `HR1()`, `Bartlett(bw)`)
- `term::Symbol`: Which coefficient to plot (default: `lp.shock`)
- `levels::Vector`: Confidence levels for bands (default: `[0.95]`)
- `irf_scale::Real`: Multiplicative scale factor for IRF values (default: `1.0`, use `100` for percentage)

All other keyword arguments (e.g., `title`, `titlefontsize`, `color`, `fillcolor`, `fillalpha`,
`linewidth`, `xlabel`, `ylabel`, `legend`) are passed through to Plots.

# Example
```julia
lp_result = lp(@formula(leads(y) ~ x), df; horizon=12)
plot(lp_result, HR1(); levels=[0.68, 0.90], irf_scale=100, title="IRF", titlefontsize=12)
```
"""
@recipe function f(wrapper::IRFPlot)
    lp = wrapper.lp
    cov = wrapper.cov
    term = wrapper.term
    levels = wrapper.levels
    irf_scale = wrapper.irf_scale

    beta = coefpath(lp; term = term) .* irf_scale
    se = stderror(cov; term = term) .* irf_scale
    horizons = collect(0:lp.horizon)

    # Validate levels
    sorted_levels = sort(levels; rev = true)  # Widest band first for proper layering
    for level in sorted_levels
        (level <= 0 || level >= 1) && throw(ArgumentError("levels must be in (0, 1)"))
    end

    # Compute ribbon for widest confidence band (will be drawn first/bottom)
    z_max = quantile(Normal(), 0.5 + sorted_levels[1] / 2)
    ribbon_max = z_max .* se

    # Set default plot attributes (user can override with -->)
    xlabel --> "Horizon"
    ylabel --> String(term)
    label --> "IRF"
    linewidth --> 2
    fillalpha --> 0.3
    legend --> :best

    # For multiple confidence levels, add inner bands as additional series
    if length(sorted_levels) > 1
        for (idx, level) in enumerate(sorted_levels[2:end])
            @series begin
                z = quantile(Normal(), 0.5 + level / 2)
                band = z .* se
                ribbon := band
                fillalpha := 0.3 + 0.15 * idx  # Darker for inner bands
                label := ""  # No legend for inner bands
                linewidth := 0  # No line for inner bands
                linecolor := :transparent
                horizons, beta
            end
        end
    end

    # Return main series with widest ribbon (drawn last, on top)
    ribbon --> ribbon_max
    return horizons, beta
end

# Recipe for LocalProjection + LocalProjectionCovariance
@recipe function f(lp::LocalProjection, cov::LocalProjectionCovariance;
        term = lp.shock, levels = [0.95], irf_scale = 1.0)
    IRFPlot(lp, cov, term, Float64.(levels), Float64(irf_scale))
end

# Recipe for LocalProjection + covariance estimator
@recipe function f(lp::LocalProjection,
        estimator::CovarianceMatrices.AbstractAsymptoticVarianceEstimator;
        term = lp.shock, levels = [0.95], irf_scale = 1.0)
    cov = vcov(estimator, lp)
    IRFPlot(lp, cov, term, Float64.(levels), Float64(irf_scale))
end

# ============================================================================
# Test Items
# ============================================================================

@testitem "cumul transformation" tags=[:cumul, :core] begin
    using LocalProjections
    using DataFrames, StatsModels, Regress, StatsBase, Test

    # Create simple synthetic data
    n = 100
    df = DataFrame(
        x = 1.0:n,
        y = sin.(1.0:n) .+ (1.0:n) ./ 10
    )

    # Test cumul transformation in lp() context
    horizon = 3
    lp_result = lp(@formula(cumul(y) ~ x), df; horizon = horizon)

    # Manually compute cumulative sums and compare
    # Note: Must replicate lp's filtering logic exactly
    df_filtered = dropmissing(df, [:y, :x], disallowmissing = true)

    for h in 0:horizon
        # Get coefficient from lp result
        lp_coef = coef(lp_result.models[h + 1])[2]  # x coefficient (index 2, after intercept)

        # Manually compute cumulative y at horizon h using StatsModels
        cumul_term_h = CumulTerm{typeof(Term(:y)), typeof(cumul)}(Term(:y), h)
        y_h = StatsModels.modelcols(cumul_term_h, df_filtered)
        y_h_manual = map(x->sum(x), map(t->df_filtered.y[t:(t + h)], 1:(nrow(df_filtered) - h)))

        # Find complete observations (no NaN in y_h)
        complete_obs = .!isnan.(y_h)
        y_manual = y_h[complete_obs]
        # Manually run regression on complete data
        X_manual = hcat(ones(sum(complete_obs)), df_filtered.x[complete_obs])

        @test y_manual == y_h_manual  # Verify manual cumulative matches StatsModels output
        manual_coef = (X_manual \ y_manual)[2]  # x coefficient

        # Compare coefficients
        @test lp_coef ≈ manual_coef atol=1e-10
    end
end

@testitem "leads transformation" tags=[:leads, :core] begin
    using LocalProjections
    using LocalProjections: _lead_to_float64
    using DataFrames, StatsModels, Regress, StatsBase, Test

    # Create simple synthetic data
    n = 100
    df = DataFrame(
        x = 1.0:n,
        y = cos.(1.0:n) .+ (1.0:n) ./ 20
    )

    # Test leads transformation in lp() context
    horizon = 3
    lp_result = lp(@formula(leads(y) ~ x), df; horizon = horizon)

    # Manually compute leads and compare
    df_filtered = dropmissing(df, [:y, :x], disallowmissing = true)

    for h in 0:horizon
        # Get coefficient from lp result
        lp_coef = coef(lp_result.models[h + 1])[2]  # x coefficient

        # Manually compute lead of y at horizon h using StatsModels
        leads_term_h = LeadTerm{typeof(Term(:y)), typeof(leads)}(Term(:y), h)
        y_h = StatsModels.modelcols(leads_term_h, df_filtered)

        # Find complete observations (no NaN)
        complete_obs = .!isnan.(y_h)

        # Manually run regression on complete data
        X_manual = hcat(ones(sum(complete_obs)), df_filtered.x[complete_obs])
        y_manual = y_h[complete_obs]
        manual_coef = (X_manual \ y_manual)[2]  # x coefficient

        # Compare coefficients
        @test lp_coef ≈ manual_coef atol=1e-10
    end

    # Verify NaN handling (not missing)
    y_lead_test = _lead_to_float64(df.y, 5)
    @test eltype(y_lead_test) == Float64
    @test any(isnan, y_lead_test)  # Should have NaN at boundaries
end

@testitem "anchor function syntax" tags=[:anchor, :core] begin
    using LocalProjections
    using DataFrames, StatsModels, Regress, StatsBase, Test

    # Create simple synthetic data with both y and z
    n = 100
    df = DataFrame(
        x = 1.0:n,
        y = sin.(1.0:n) .+ (1.0:n) ./ 10,
        z = cos.(1.0:n)
    )

    # Test anchor transformation with function syntax
    horizon = 3
    lp_result = lp(@formula(anchor(y, z) ~ x), df; horizon = horizon)

    # Manually compute anchored response and compare
    df_filtered = dropmissing(df, [:y, :z, :x], disallowmissing = true)

    for h in 0:horizon
        # Get coefficient from lp result
        lp_coef = coef(lp_result.models[h + 1])[2]  # x coefficient

        # Manually compute anchored response using StatsModels
        inner_leads = LeadTerm{typeof(Term(:y)), typeof(leads)}(Term(:y), h)
        anchor_h = AnchorTerm{typeof(inner_leads), typeof(Term(:z)), typeof(anchor)}(
            inner_leads, Term(:z), 0)  # horizon=0 because lead is in inner term
        y_h = StatsModels.modelcols(anchor_h, df_filtered)

        # Find complete observations (no NaN)
        complete_obs = .!isnan.(y_h)

        # Manually run regression on complete data
        X_manual = hcat(ones(sum(complete_obs)), df_filtered.x[complete_obs])
        y_manual = y_h[complete_obs]
        manual_coef = (X_manual \ y_manual)[2]  # x coefficient

        # Compare coefficients
        @test lp_coef ≈ manual_coef atol=1e-10
    end
end

@testitem "anchor pipe syntax" tags=[:anchor, :core] begin
    using LocalProjections
    using DataFrames, StatsModels, Regress, StatsBase, Test

    # Create simple synthetic data
    n = 100
    df = DataFrame(
        x = 1.0:n,
        y = sin.(1.0:n) .+ (1.0:n) ./ 10,
        z = cos.(1.0:n)
    )

    # Test anchor with pipe syntax
    horizon = 3
    lp_pipe = lp(@formula(leads(y)|z ~ x), df; horizon = horizon)

    # Test anchor with function syntax (should be identical)
    lp_func = lp(@formula(anchor(y, z) ~ x), df; horizon = horizon)

    # Compare results from both syntaxes
    for h in 0:horizon
        pipe_coef = coef(lp_pipe.models[h + 1])[2]
        func_coef = coef(lp_func.models[h + 1])[2]

        @test pipe_coef ≈ func_coef atol=1e-10
    end

    # Also verify against manual computation
    df_filtered = dropmissing(df, [:y, :z, :x], disallowmissing = true)

    for h in 0:horizon
        lp_coef = coef(lp_pipe.models[h + 1])[2]

        # Manual computation using StatsModels
        inner_leads = LeadTerm{typeof(Term(:y)), typeof(leads)}(Term(:y), h)
        anchor_h = AnchorTerm{typeof(inner_leads), typeof(Term(:z)), typeof(anchor)}(
            inner_leads, Term(:z), 0)  # horizon=0 because lead is in inner term
        y_h = StatsModels.modelcols(anchor_h, df_filtered)

        y_h_manual = lead(df_filtered.y, h, default = NaN) .- df_filtered.z
        complete_obs = .!isnan.(y_h)
        X_manual = hcat(ones(sum(complete_obs)), df_filtered.x[complete_obs])
        y_manual = y_h[complete_obs]
        y_manual_manual = y_h_manual[complete_obs]
        @test y_manual == y_manual_manual  # Verify manual matches StatsModels output
        manual_coef = (X_manual \ y_manual)[2]

        @test lp_coef ≈ manual_coef atol=1e-10
    end
end

@testitem "modelcols anchor matches manual lead computation" tags=[:anchor, :verification] begin
    using LocalProjections
    using DataFrames, StatsModels, Test

    # Create test data
    n = 100
    df = DataFrame(
        y = sin.(1.0:n) .+ (1.0:n) ./ 10,
        z = cos.(1.0:n)
    )

    # Test that modelcols(AnchorTerm) matches manual lead() - z computation
    for h in 0:5
        # Method 1: Using StatsModels.modelcols with AnchorTerm
        inner_leads = LeadTerm{typeof(Term(:y)), typeof(leads)}(Term(:y), h)
        anchor_h = AnchorTerm{typeof(inner_leads), typeof(Term(:z)), typeof(anchor)}(
            inner_leads, Term(:z), 0)
        y_modelcols = StatsModels.modelcols(anchor_h, df)

        # Method 2: Manual computation using lead() - z
        y_manual = lead(df.y, h, default = NaN) .- df.z

        # Compare (must handle NaN carefully)
        for i in 1:n
            if isnan(y_modelcols[i]) && isnan(y_manual[i])
                @test true  # Both NaN, OK
            elseif isnan(y_modelcols[i]) || isnan(y_manual[i])
                @test false  # One NaN, other not - FAIL
            else
                @test y_modelcols[i] ≈ y_manual[i] atol=1e-10
            end
        end
    end
end

@testitem "cumulative anchor (nested)" tags=[:nested, :anchor, :cumul] begin
    using LocalProjections
    using DataFrames, StatsModels, Regress, StatsBase, Test

    # Create simple synthetic data
    n = 100
    df = DataFrame(
        x = 1.0:n,
        y = sin.(1.0:n) .+ (1.0:n) ./ 10,
        z = cos.(1.0:n)
    )

    # Test cumulative anchor: cumul(y)|z
    horizon = 3
    lp_result = lp(@formula(cumul(y)|z ~ x), df; horizon = horizon)

    # Manually compute cumulative anchored response
    df_filtered = dropmissing(df, [:y, :z, :x], disallowmissing = true)

    for h in 0:horizon
        # Get coefficient from lp result
        lp_coef = coef(lp_result.models[h + 1])[2]

        # Manual computation using StatsModels: first cumul, then anchor
        inner_cumul = CumulTerm{typeof(Term(:y)), typeof(cumul)}(Term(:y), h)
        anchor_h = AnchorTerm{typeof(inner_cumul), typeof(Term(:z)), typeof(anchor)}(
            inner_cumul, Term(:z), 0)  # horizon=0 because cumul already has horizon
        y_h = StatsModels.modelcols(anchor_h, df_filtered)

        # Find complete observations (no NaN)
        complete_obs = .!isnan.(y_h)

        # Manually run regression
        X_manual = hcat(ones(sum(complete_obs)), df_filtered.x[complete_obs])
        y_manual = y_h[complete_obs]
        manual_coef = (X_manual \ y_manual)[2]

        # Compare coefficients
        @test lp_coef ≈ manual_coef atol=1e-10
    end
end

@testitem "nested log transformations" tags=[:nested, :core] begin
    using LocalProjections
    using DataFrames, StatsModels, Regress, StatsBase, Test

    # Create simple synthetic data (positive values for log)
    n = 100
    df = DataFrame(
        x = 1.0:n,
        y = exp.(1.0:n) ./ 100,  # Positive values
        z = exp.(2.0:101) ./ 150
    )

    # Test cumul(log(y))
    horizon = 2
    lp_cumul_log = lp(@formula(cumul(log(y)) ~ x), df; horizon = horizon)
    df_filtered1 = dropmissing(df, [:y, :x], disallowmissing = true)

    for h in 0:horizon
        lp_coef = coef(lp_cumul_log.models[h + 1])[2]

        # Manual computation: cumulative sum of log(y) from t to t+h
        log_y = log.(df_filtered1.y)
        y_h = [t + h <= length(log_y) ? sum(log_y[t:(t + h)]) : NaN
               for t in 1:length(log_y)]
        complete_obs = .!isnan.(y_h)

        X_manual = hcat(ones(sum(complete_obs)), df_filtered1.x[complete_obs])
        y_manual = y_h[complete_obs]
        manual_coef = (X_manual \ y_manual)[2]

        @test lp_coef ≈ manual_coef atol=1e-10
    end

    # Test leads(log(y))
    lp_lead_log = lp(@formula(leads(log(y)) ~ x), df; horizon = horizon)
    df_filtered2 = dropmissing(df, [:y, :x], disallowmissing = true)

    for h in 0:horizon
        lp_coef = coef(lp_lead_log.models[h + 1])[2]

        # Manual computation: lead of log(y) by h
        log_y = log.(df_filtered2.y)
        y_h = [t + h <= length(log_y) ? log_y[t + h] : NaN for t in 1:length(log_y)]
        complete_obs = .!isnan.(y_h)

        X_manual = hcat(ones(sum(complete_obs)), df_filtered2.x[complete_obs])
        y_manual = y_h[complete_obs]
        manual_coef = (X_manual \ y_manual)[2]

        @test lp_coef ≈ manual_coef atol=1e-10
    end

    # Test anchor(log(y), z)
    lp_anchor_log = lp(@formula(anchor(log(y), z) ~ x), df; horizon = horizon)
    df_filtered3 = dropmissing(df, [:y, :z, :x], disallowmissing = true)

    for h in 0:horizon
        lp_coef = coef(lp_anchor_log.models[h + 1])[2]

        # Manual computation: lead of log(y) by h, minus z at t
        log_y = log.(df_filtered3.y)
        y_h = [t + h <= length(log_y) ? log_y[t + h] - df_filtered3.z[t] : NaN
               for t in 1:length(log_y)]
        complete_obs = .!isnan.(y_h)

        X_manual = hcat(ones(sum(complete_obs)), df_filtered3.x[complete_obs])
        y_manual = y_h[complete_obs]
        manual_coef = (X_manual \ y_manual)[2]

        # Use relative tolerance for large values
        @test lp_coef ≈ manual_coef rtol=1e-10
    end
end

@testitem "summarize function" tags=[:summarize, :api] begin
    using LocalProjections
    using DataFrames, StatsModels, Test
    using CovarianceMatrices: HC1

    n = 100
    df = DataFrame(x = randn(n), y = randn(n))
    lp_result = lp(@formula(leads(y) ~ x), df; horizon = 5)
    cov = LocalProjections.vcov(HC1(), lp_result)

    # Test basic summarize returns IRFSummary
    summary_obj = summarize(lp_result, cov)
    @test summary_obj isa IRFSummary
    @test length(summary_obj.horizon) == 6  # horizons 0-5
    @test summary_obj.horizon == collect(0:5)
    @test summary_obj.term == :x
    @test summary_obj.level == 0.95

    # Test conversion to DataFrame
    summary_df = DataFrame(summary_obj)
    @test summary_df isa DataFrame
    @test nrow(summary_df) == 6
    @test names(summary_df) == ["horizon", "coef", "se", "lower", "upper"]

    # Test with scale
    summary_scaled = summarize(lp_result, cov; scale = 100)
    @test summary_scaled.coef ≈ summary_obj.coef .* 100
    @test summary_scaled.scale == 100.0

    # Test with estimator directly
    summary_direct = summarize(lp_result, HC1())
    @test summary_direct.coef ≈ summary_obj.coef atol=1e-10

    # Test confidence bounds are sensible (lower < coef < upper when se > 0)
    for i in 1:length(summary_obj.horizon)
        if summary_obj.se[i] > 0
            @test summary_obj.lower[i] < summary_obj.coef[i] < summary_obj.upper[i]
        end
    end

    # Test different confidence level
    summary_90 = summarize(lp_result, cov; level = 0.90)
    @test summary_90.level == 0.90
    # 90% CI should be narrower than 95% CI
    @test all(summary_90.upper .- summary_90.lower .<
              summary_obj.upper .- summary_obj.lower)
end

end # module LocalProjections
