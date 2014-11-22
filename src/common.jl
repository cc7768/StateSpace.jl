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
        !(adjacent(rs[i], rs[i+1])) && return true
    end
    return false
end

type TimeVaryingParam{T, S<:Integer}
    vals::Vector{T}
    ranges::Vector{UnitRange{S}}

    function TimeVaryingParam(vals, ranges)
        if !(all_disjoint(ranges...))
            throw(ArgumentError("Ranges are overlapping"))
        end

        if length(vals) != length(ranges)
            throw(ArgumentError("Must supply same number of vals and ranges"))
        end

        if any_gaps(ranges...)
            throw(ArgumentError("Ranges contain missing periods"))
        end

        # at this point I think we are good!
        new(vals, ranges)
    end
end

function TimeVaryingParam{T, S<:Integer}(vals::Vector{T},
                                         ranges::Vector{UnitRange{S}})
    TimeVaryingParam{T, S}(vals, ranges)
end

# constructor of the form (mat, period_range), (mat2, period_range2)
function TimeVaryingParam{T, S<:Integer}(input::(T, UnitRange{S})...)
    vals = T[]
    ranges = UnitRange{S}[]
    for t in input
        push!(vals, t[1])
        push!(ranges, t[2])
    end
    TimeVaryingParam(vals, ranges)
end

# constructor for single item to be repeated on range 1:T
TimeVaryingParam(x, T::Int) = TimeVaryingParam((x, 1:T))

function getindex(tvp::TimeVaryingParam, t::Int)
    # if there is only one value, return that for all t.
    if length(tvp.vals) == 1
        return tvp.vals[1]
    end

    # b/c all ranges are disjoint, ind has exactly zero or one elements
    ind = find(x->t in x, tvp.ranges)
    isempty(ind) && error("Invalid index. t=$t not in any supplied ranges")
    return tvp.vals[ind[1]]
end

length(t::TimeVaryingParam) = sum([length(r) for r in t.ranges])

# catch all for making the single item always returned by getindex
convert(::Type{TimeVaryingParam}, x) = TimeVaryingParam((x, 1:typemax(1)))
convert(::Type{TimeVaryingParam}, x::TimeVaryingParam) = x

function show{T}(io::IO, tvp::TimeVaryingParam{T})
    n = length(tvp.vals)
    msg = "TimeVaryingParam with $n different params of type $T"
    print(io, msg)
    nothing
end


# helper functions to retrieve elements from TimeVaryingParam or matrix in
# consistent way
_get_yt(y::AbstractMatrix, t::Int) = y[:, t]
_get_yt(y::TimeVaryingParam, t) = y[t]

_get_T(y::AbstractMatrix) = size(y, 2)
_get_T(y::TimeVaryingParam) = length(y)


## -------------- ##
#- Filtered State -#
## -------------- ##

type FilteredState{T, D<:ContinuousMultivariateDistribution}
    observations::Union(Array{T, 2}, TimeVaryingParam{Array{T, 1}})
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
