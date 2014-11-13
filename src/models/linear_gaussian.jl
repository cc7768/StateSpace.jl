## ----------------------- ##
#- type: LinearGaussianSSM -#
## ----------------------- ##

type LinearGaussianSSM
    F::TimeVaryingParam
    V::TimeVaryingParam
    G::TimeVaryingParam
    W::TimeVaryingParam

    # Handle matrix or TimeVaryingParam inputs
    function LinearGaussianSSM(F::Union(TimeVaryingParam, Matrix),
                               V::Union(TimeVaryingParam, Matrix),
                               G::Union(TimeVaryingParam, Matrix),
                               W::Union(TimeVaryingParam, Matrix))
        @assert issquare(F)
        @assert issquare(V)
        @assert issquare(W)
        @assert size(F) == size(V)
        @assert size(G, 1) == size(W, 1)
        F, V, G, W = map(x->convert(TimeVaryingParam, x), Any[F, V, G, W])
        return new(F, V, G, W)
    end
end

# Univariate state and data
function LinearGaussianSSM(F::Real, V::Real, G::Real, W::Real)
    f = Array(Float64, 1, 1); f[1] = F
    v = Array(Float64, 1, 1); v[1] = V
    g = Array(Float64, 1, 1); g[1] = G
    w = Array(Float64, 1, 1); w[1] = W
    return LinearGaussianSSM(f, v, g, w)
end

# Univariate state, n-d data
function LinearGaussianSSM(F::Real, V::Real, G::Matrix, W::Matrix)
    f = Array(Float64, 1, 1); f[1] = F
    v = Array(Float64, 1, 1); v[1] = V
    return LinearGaussianSSM(f, v, G, W)
end

# m-d state, univariate data
function LinearGaussianSSM(F::Matrix, V::Matrix, G::Matrix, W::Real)
    w = Array(Float64, 1, 1); w[1] = W
    return LinearGaussianSSM(F, V, G, w)
end
