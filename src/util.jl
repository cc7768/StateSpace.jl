function replacenan(x::Array, replacement::Array)
    nrow = size(x, 1)
    newx = copy(x)
    for i=1:nrow
        newx[i] = isnan(x[i]) ? replacement[i] : x[i]
    end

    return newx
end

issquare(x::AbstractMatrix) = size(x, 1) == size(x, 2)
issquare(t::TimeVaryingParam) = all(map(x->issquare(x), t.vals))
