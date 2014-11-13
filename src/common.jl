using Distributions

import Distributions: mean, var, cov, rand

## -------------- ##
#- Filtered State -#
## -------------- ##

type FilteredState{T, D<:ContinuousMultivariateDistribution}
    observations::Array{T, 2}
    state_dist::Array{D}
    pred_state::Array{D}
    loglik::T
end

function show(io::IO, fs::FilteredState)
    n = length(fs.state_dist)
    dobs = size(fs.observations, 1)
    dstate = length(fs.state_dist[1])
    msg = "FilteredState\n"
    msg *= "  - $n estimates of $dstate-D process from $dobs-D observations\n"
    msg *= "  - Log-likelihood: $(fs.loglik)"
    print(io, msg)
    nothing
end

for op in (:mean, :var, :cov, :cor, :rand)
    @eval begin
        function ($op){T}(s::FilteredState{T})
            result = Array(T, length(s.state_dist[1]), length(s.state_dist))
            for i in 1:length(s.state_dist)
                result[:, i] = ($op)(s.state_dist[i])
            end
            return result
        end
    end
end


## ---------------- ##
#- TimeVaryingParam -#
## ---------------- ##

# are two ranges disjoint?
isdisjoint(r1::UnitRange, r2::UnitRange) = isempty(intersect(r1, r2))

# are a collection of ranges all disjoint?
function all_disjoint(rs::UnitRange...)
    n = length(rs)
    for i=1:n, j=i+1:n
        if !(isdisjoint(rs[i], rs[j]))
            return false
        end
    end
    return true
end

# are two unit ranges adjacent, meaning does one start where other left off
adjacent(r1::UnitRange, r2::UnitRange) = r1.stop + 1 == r2.start

# are there any gaps in the ranges?
any_gaps(rs::UnitRange) = false  # can't have gaps in UnitRange (step === 1)
function any_gaps(r1::UnitRange, rs::UnitRange...)
    # Just need to check gaps at the edges
    n = length(rs)
    !(adjacent(r1, rs[1])) && return true
    for i=1:n-1
        !(adjacent(rs[i], rs[i+i])) && return true
    end
    return false
end

type TimeVaryingParam{T<:Real, S<:Integer}
    mats::Vector{Matrix{T}}
    ranges::Vector{UnitRange{S}}

    function TimeVaryingParam(mats, ranges)
        if !(all_disjoint(ranges...))
            throw(ArugumentError("Ranges are overlapping"))
        end
        if length(mats) != length(ranges)
            throw(ArugumentError("Must supply same number of mats and ranges"))
        end
        if any_gaps(ranges...)
            throw(ArugumentError("Ranges contain missing periods"))
        end
        new(mats, ranges)
    end
end

function TimeVaryingParam{T<:Real, S<:Integer}(mats::Vector{Matrix{T}},
                                               ranges::Vector{UnitRange{S}})
    TimeVaryingParam{T, S}(mats, ranges)
end

# constructor of the form (mat, period_range), (mat2, period_range2)
function TimeVaryingParam{T<:Real, S<:Integer}(input::(Matrix{T}, UnitRange{S})...)
    mats = Matrix{T}[]
    ranges = UnitRange{S}[]
    for t in input
        push!(mats, t[1])
        push!(ranges, t[2])
    end
    TimeVaryingParam(mats, ranges)
end

function getindex(tvp::TimeVaryingParam, t::Int)
    # if there is only one matrix, return that for all t.
    if length(tvp.mats) == 1
        return tvp.mats[1]
    end

    # b/c all ranges are disjoint, ind has exactly zero or one elements
    ind = find(x->t in x, tvp.ranges)
    isempty(ind) && error("Invalid index. t=$t not in any supplied ranges")
    return tvp.mats[ind[1]]
end

function convert(::Type{TimeVaryingParam}, x::Matrix)
    TimeVaryingParam((x, 1:typemax(1)))
end
convert(::Type{TimeVaryingParam}, x::TimeVaryingParam) = x

function show(io::IO, tvp::TimeVaryingParam)
    n = length(tvp.mats)
    msg = "TimeVaryingParam with $n different matrices"
    print(io, msg)
    nothing
end
