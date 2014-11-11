
issquare(x::Matrix) = size(x, 1) == size(x, 2)

type LinearGaussianSSM{T}
    F::Matrix{T}
    V::Matrix{T}
    G::Matrix{T}
    W::Matrix{T}

    function LinearGaussianSSM(F::Matrix{T}, V::Matrix{T}, G::Matrix{T}, W::Matrix{T})
        @assert issquare(F)
        @assert issquare(V)
        @assert issquare(W)
        @assert size(F) == size(V)
        @assert size(G, 1) == size(W, 1)
        return new(F, V, G, W)
    end
end

function LinearGaussianSSM{T <: Real}(F::Matrix{T}, V::Matrix{T}, G::Matrix{T},
        W::Matrix{T})
    return LinearGaussianSSM{T}(F, V, G, W)
end

# Univariate state and data
function LinearGaussianSSM{T<:Real}(F::T, V::T, G::T, W::T)
    f = Array(T, 1, 1); f[1] = F
    v = Array(T, 1, 1); v[1] = V
    g = Array(T, 1, 1); g[1] = G
    w = Array(T, 1, 1); w[1] = W
    return LinearGaussianSSM(f, v, g, w)
end

# Univariate state, n-d data
function LinearGaussianSSM{T<:Real}(F::T, V::T, G::Matrix{T}, W::Matrix{T})
    f = Array(T, 1, 1); f[1] = F
    v = Array(T, 1, 1); v[1] = V
    return LinearGaussianSSM(f, v, G, W)
end

# m-d state, univariate data
function LinearGaussianSSM{T<:Real}(F::Matrix{T}, V::Matrix{T}, G::Matrix{T}, W::T)
    w = Array(T, 1, 1); w[1] = W
    return LinearGaussianSSM(F, V, G, w)
end

## Core methods
function predict(m::LinearGaussianSSM, x::GenericMvNormal)
    return MvNormal(m.F * mean(x), m.F * cov(x) * m.F' + m.V)
end

function observe(m::LinearGaussianSSM, x::GenericMvNormal)
    return MvNormal(m.G * mean(x), m.G * cov(x) * m.G' + m.W)
end

function update(m::LinearGaussianSSM, pred::GenericMvNormal, y)
    innovation = y - m.G * mean(pred)
    innovation_cov = m.G * cov(pred) * m.G' + m.W
    K = cov(pred) * m.G' * inv(innovation_cov)
    mean_update = mean(pred) + K * innovation
    cov_update = (eye(cov(pred)) - K * m.G) * cov(pred)
    return MvNormal(mean_update, cov_update)
end

function update(m::LinearGaussianSSM, xpred::GenericMvNormal,
                ypred::GenericMvNormal, y)
    innovation = y - mean(ypred)
    innovation_cov = cov(ypred)
    K = cov(xpred) * m.G' * inv(innovation_cov)
    mean_update = mean(xpred) + K * innovation
    cov_update = (eye(cov(xpred)) - K * m.G) * cov(xpred)
    return MvNormal(mean_update, cov_update)
end

function update!(m::LinearGaussianSSM, fs::FilteredState, y)
    x_pred = predict(m, fs.state_dist[end])
    x_filt = update(m, x_pred, y)
    push!(fs.state_dist, x_filt)
    fs.observations = [fs.observations y]
    return fs
end

function filter{T}(y::Array{T}, m::LinearGaussianSSM{T}, x0::GenericMvNormal)
    # Initial Parameters and Allocate Space
    ysize = size(y, 2)
    loglik = 0.
    x_filtered = Array(GenericMvNormal, ysize)

    # Kalman Filter
    x_pred = predict(m, x0)
    y_pred = observe(m, x_pred)
    loglik += logpdf(y_pred, y[:, 1] - mean(y_pred))
    x_filtered[1] = update(m, x_pred, y_pred, y[:, 1])

    for i in 2:ysize
        x_pred = predict(m, x_filtered[i-1])
        y_pred = observe(m, x_pred)
        # Check for missing values in observation
        if any(isnan, y[:, i])
            # The replacenan function is a quick fix.  Should decide on
            # future convention to make this better.
            ytwiddly = replacenan(y[:, i], mean(y_pred))
            loglik += logpdf(y_pred, ytwiddly)
            x_filtered[i] = x_pred
        else
            loglik += logpdf(y_pred, y[:, i])
            x_filtered[i] = update(m, x_pred, y[:, i])
        end
    end
    return FilteredState(y, x_filtered, loglik)
end

function replacenan(x::Array, replacement::Array)
    nrow = size(x, 1)
    newx = copy(x)
    for i=1:nrow
        newx[i] = isnan(x[i])?replacement[i]:x[i]
    end

    return newx
end

function smooth{T}(m::LinearGaussianSSM{T}, fs::FilteredState{T})
	# Withdraw and Use Parameters
	y = fs.y
	ysize = size(y, 2)
	x_filtered = fs.x_filtered
	x_smoothed = Array(GenericMvNormal, ysize)

	# Smooth this baby out
	for t=ysize:1
		nothing
	end
    error("Not implemented yet")
end

function smooth{T}(y::Array{T}, m::LinearGaussianSSM, x0::GenericMvNormal)
	fs = filter(y, m, x0)
	return smooth(m, fs)
end

function simulate{T}(m::LinearGaussianSSM{T}, n::Int64, x0::GenericMvNormal)
    x = zeros(T, length(x0), n)
    y = zeros(T, size(m.G, 1), n)
    x[:, 1] = rand(MvNormal(m.F * rand(x0), m.V))
    y[:, 1] = rand(MvNormal(m.G * x[:, 1], m.W))
    for i in 2:n
        x[:, i] = rand(MvNormal(m.F * x[:, i-1], m.V))
        y[:, i] = rand(MvNormal(m.G * x[:, i], m.W))
    end
    return (x, y)
end

