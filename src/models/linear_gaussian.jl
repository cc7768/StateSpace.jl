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
        F, V, G, W = map(x->convert(TimeVaryingParam, x), Any[F, V, G, W])
        @assert size(F[1]) == size(V[1])
        @assert size(G[1], 1) == size(W[1], 1)
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

function simulate(m::LinearGaussianSSM, n::Int64, x0::MvNormal)
    # NOTE: this code will only work when x_t and y_t have the same dimension
    #       for all time periods
    x = zeros(Float64, length(x0), n)
    y = zeros(Float64, size(m.G[1], 1), n)
    x[:, 1] = rand(MvNormal(m.F[1] * rand(x0), m.V[1]))
    y[:, 1] = rand(MvNormal(m.G[1] * x[:, 1], m.W[1]))
    for t=2:n
        x[:, t] = rand(MvNormal(m.F[t] * x[:, t-1], m.V[t]))
        y[:, t] = rand(MvNormal(m.G[t] * x[:, t], m.W[t]))
    end
    return (x, y)
end
