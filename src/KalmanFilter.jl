## ------------ ##
#- Core Methods -#
## ------------ ##

function predict(m::LinearGaussianSSM, x::MvNormal, t::Int)
    return MvNormal(m.F[t] * mean(x), m.F[t] * cov(x) * m.F[t]' + m.V[t])
end

function observe(m::LinearGaussianSSM, x::MvNormal, t::Int)
    return MvNormal(m.G[t] * mean(x), m.G[t] * cov(x) * m.G[t]' + m.W[t])
end

function update(m::LinearGaussianSSM, pred::MvNormal, y, t::Int)
    innovation = y - m.G[t] * mean(pred)
    innovation_cov = m.G[t] * cov(pred) * m.G[t]' + m.W[t]
    K = cov(pred) * m.G[t]' * inv(innovation_cov)
    mean_update = mean(pred) + K * innovation
    cov_update = (eye(cov(pred)) - K * m.G[t]) * cov(pred)
    return MvNormal(mean_update, cov_update)
end

function update(m::LinearGaussianSSM, xpred::MvNormal,
                ypred::MvNormal, y, t::Int)
    innovation = y - mean(ypred)
    innovation_cov = cov(ypred)
    K = cov(xpred) * m.G[t]' * inv(innovation_cov)
    mean_update = mean(xpred) + K * innovation
    cov_update = (eye(cov(xpred)) - K * m.G[t]) * cov(xpred)
    return MvNormal(mean_update, cov_update)
end

function update!(m::LinearGaussianSSM, fs::FilteredState, y, t::Int)
    x_pred = predict(m, fs.state_dist[end], t::Int)
    x_filt = update(m, x_pred, y, t::Int)
    push!(fs.state_dist, x_filt)
    push!(fs.pred_state, x_pred)

    # TODO: This will be horribly inefficient. Is there a better solution?
    fs.observations = [fs.observations y]
    return fs
end

function filter(y::Array, m::LinearGaussianSSM, x0::MvNormal)
    # Initial Parameters and Allocate Space
    T = size(y, 2)
    loglik = 0.
    x_filtered = Array(MvNormal, T)
    x_pred = Array(MvNormal, T)

    # Kalman Filter
    x_pred[1] = predict(m, x0, 1)
    y_pred = observe(m, x_pred[1], 1)
    loglik += logpdf(y_pred, y[:, 1] - mean(y_pred))
    x_filtered[1] = update(m, x_pred[1], y_pred, y[:, 1], 1)

    for i in 2:T
        x_pred[i] = predict(m, x_filtered[i-1], i)
        y_pred = observe(m, x_pred[i], i)
        # Check for missing values in observation
        if any(isnan, y[:, i])
            # The replacenan function is a quick fix.  Should decide on
            # future convention to make this better.
            ytwiddly = replacenan(y[:, i], mean(y_pred))
            loglik += logpdf(y_pred, ytwiddly)
            x_filtered[i] = x_pred[i]
        else
            loglik += logpdf(y_pred, y[:, i])
            x_filtered[i] = update(m, x_pred[i], y[:, i], i)
        end
    end
    return FilteredState(y, x_filtered, x_pred, loglik)
end

# TODO: should this function return the samples, or the distributions?
#       I kinda think it should return distributions. Need to think about it
function bw_sampler(m::LinearGaussianSSM, fs::FilteredState)
    # pull out necessary objects
    F = m.F
    filt, pred = fs.state_dist, fs.pred_state
    T = length(filt)      # number of samples
    ns = length(filt[1])  # number of states

    # allocate space for samples
    x_sample = Array(Float64, ns, T)

    # first (well, actually last) sample
    x_sample[:, end] = rand(filt[end])

    # iterate backward for rest of samples
    for t=T-1:-1:1
        # pull out sufficient stats
        xt_t, pt_t = mean(filt[t]), cov(filt[t])
        ptp1_t_inv = inv(cov(pred[t+1]))  # CHASE: Check this index please

        # P_{t|t}A_t'P_{t+1|t}^{-1}  - A useful constant
        PM = pt_t*F[t]'*inv(ptp1_t_inv)

        # x_{t|t+1}=x_{t|t} + P_{t|t}A_t'P_{t+1|t}^{-1}(x_{t+1} - A_t x_{t|t})
        xt_tp1 = xt_t + PM*(x_sample[:, t+1] - F[t]*xt_t)

        # P_{t|t+1} = (I - P_{t|t}A_t'P_{t+1|t}^{-1}A_t) P_{t|t}
        pt_tp1 = (I - PM*F[t])*pt_t

        # construct distribution and sample
        new_dist = MvNormal(xt_tp1, pt_tp1)
        x_sample[:, t] = rand(new_dist)
    end
    return x_sample
end


function fwfilter_bwsampler(y::Union(Array, TimeVaryingParam), m::LinearGaussianSSM,
                            x0::MvNormal)
    return bw_sampler(filter(y, m, x0))
end


function smooth(m::LinearGaussianSSM, fs::FilteredState)
    # Uses RauchTung-Striebel Algorithm to smooth: See wiki
    # Withdraw and Use Parameters
    y = fs.y
    T = size(y, 2)
    x_filtered = fs.state_dist
    x_pred = fs.pred_state
    x_smoothed = Array(MvNormal, T)
    x_smoothed[end] = x_filtered[end]

    error("Not implemented yet")

    # Smooth States
    for t=T:-1:1
        C = cov(x_filtered[t])m.F[t]'inv(mean(x_pred[t+1]))
        newmean = mean(x_filtered[t]) +
       x_smoothed[t]
    end
end

function smooth(y::Array, m::LinearGaussianSSM, x0::MvNormal)
	fs = filter(y, m, x0)
	return smooth(m, fs)
end
