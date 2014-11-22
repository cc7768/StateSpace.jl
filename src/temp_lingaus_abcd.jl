using Distributions

## -------------------------- ##
#- type: LinearGaussianSSabcd -#
## -------------------------- ##

type LinearGaussianSSabcd
    A::TimeVaryingParam
    B::TimeVaryingParam
    C::TimeVaryingParam
    D::TimeVaryingParam

    # Handle AbstractMatrix or TimeVaryingParam inputs
    function LinearGaussianSSabcd(A::Union(TimeVaryingParam, AbstractMatrix),
                                  B::Union(TimeVaryingParam, AbstractMatrix),
                                  C::Union(TimeVaryingParam, AbstractMatrix),
                                  D::Union(TimeVaryingParam, AbstractMatrix))
        @assert issquare(A)
        @assert issquare(D)

        # convert to TimeVaryingParam
        A, B, C, D = map(x->convert(TimeVaryingParam, x), Any[A, B, C, D])

        # at least make sure things line up on initial value of param
        @assert size(A[1], 1) == size(B[1], 1)
        @assert size(C[1], 1) == size(D[1], 1)
        @assert size(C[1], 2) == size(A[1], 1)
        return new(A, B, C, D)
    end
end

# Univariate state and data
function LinearGaussianSSabcd(A::Real, B::Real, C::Real, D::Real)
    a = Array(Float64, 1, 1); a[1] = A
    b = Array(Float64, 1, 1); b[1] = B
    c = Array(Float64, 1, 1); c[1] = C
    d = Array(Float64, 1, 1); d[1] = D
    return LinearGaussianSSabcd(a, b, c, d)
end

# Univariate state, n-d data
function LinearGaussianSSabcd(A::Real, B::Real, C::AbstractMatrix,
                              D::AbstractMatrix)
    a = Array(Float64, 1, 1); a[1] = A
    b = Array(Float64, 1, 1); b[1] = B
    return LinearGaussianSSabcd(a, b, C, D)
end

# m-d state, univariate data
function LinearGaussianSSabcd(A::AbstractMatrix, B::AbstractMatrix,
                              C::AbstractMatrix, D::Real)
    d = Array(Float64, 1, 1); d[1] = D
    return LinearGaussianSSabcd(A, B, C, d)
end

function simulate(m::LinearGaussianSSabcd, n::Int64, x0::MvNormal)
    # NOTE: this code will only work when x_t and y_t have constant shapes
    #       for all time periods
    x = zeros(Float64, length(x0), n)
    y = zeros(Float64, size(m.C[1], 1), n)
    x[:, 1] = m.A[1]*rand(x0) + m.B[1]*randn(size(m.B[1], 2))
    y[:, 1] = m.C[1]*x[:, 1]  + m.D[1]*randn(size(m.D[1], 2))
    for t=2:n
        x[:, t] = m.A[t]*x[:, t-1] + m.B[t]*randn(size(m.B[t], 2))
        y[:, t] = m.C[t]*x[:, t]   + m.D[t]*randn(size(m.D[t], 2))
    end
    return (x, y)
end

## ------------ ##
#- type: KFstep -#
## ------------ ##

immutable KFstep
    x_p::AbstractVector{Float64}  # predicted state
    P_p::AbstractMatrix{Float64}  # predicted cov
    x_f::AbstractVector{Float64}  # updated (filtered) state
    P_f::AbstractMatrix{Float64}  # updated (filtered) cov
    y_t::AbstractVector{Float64}  # data for this step
    ll::Float64                   # log-likelihood of this step
end

## ------------------ ##
#- Kalman filter code -#
## ------------------ ##

# helper functions
_get_x0_P0(x::MvNormal) = (mean(x), cov(x))
_get_x0_P0(x::AbstractVector) = (x, eye(length(x)))

function _get_T_ny_fixy(m::LinearGaussianSSabcd, y::AbstractMatrix)
    # y should have observations as columns
    T = size(y, 2)
    ny = size(y, 1)
    ny_model = size(m.C[1], 1)

    if ny_model != ny
        T, ny = ny, T
        y = y'
    end

    return y, T, ny
end

function _get_T_ny_fixy(m::LinearGaussianSSabcd, y::AbstractMatrix)
    # y should have observations as columns
    T = size(y, 2)
    ny = size(y, 1)
    ny_model = size(m.C[1], 1)

    if ny_model != ny
        T, ny = ny, T
        y = y'
    end

    return y, T, ny
end

function _get_T_ny_fixy(m::LinearGaussianSSabcd, y::TimeVaryingParam)
    #=
    NOTES

    * The third argument probably isn't accurate. If user passed in a
      TimeVaryingParam, y probably isn't constant dimensionality (otherwise
      we should just use a Matrix)
    * length(TimeVaryingParam) is a dangerous way to compute the number
      of observations because if we `convert` a constant to a
      TimeVaryingParam the range is 1:typemax(1). This would be huge! We
      should not have that problem here because the observed data should
      be non-constant. I guard against this and make sure we don't
      return an obscenely large T
    =#

    if length(y.vals) == 1
        msg = "kfilter: Cannot pass a 'constant' TimeVaryingParam as "
        msg *= "\nobserved data"
        throw(ArgumentError(msg))
    end

    return y, length(y), length(y[1])
end

## ------------- ##
#- Kalman Filter -#
## ------------- ##

function kfilter_step(m::LinearGaussianSSabcd,
                      x::AbstractVector,
                      P::AbstractMatrix,
                      t::Int,
                      y_t)
    # unpack model parameters
    A, B, C, D = m.A, m.B, m.C, m.D

    # prediction
    x_p = A[t]*x                           # predict state
    P_p = A[t]*P*A[t]' + B[t]*B[t]'        # predicted covariance

    # observation
    yhat = C[t]*x_p                        # predicted obs
    y_tilde = y_t - yhat                   # innovation
    V = C[t]*P_p*C[t]' + D[t]*D[t]'        # innovation cov

    # update
    V_inv = inv(V)                         # don't repeat computing V^{-1}
    x_f = x_p + P_p*C[t]'*V_inv*y_tilde    # updated state
    P_f = P_p - P_p'*C[t]'*V_inv*C[t]*P_p  # Update covariance

    # compute log-likelihood for this step
    ll = logpdf(MvNormal(yhat, V), y_t)

    # _p variables are for prediction, _f for filtered
    return KFstep(x_p, P_p, x_f, P_f, y_t, ll)
end


function kfilter(m::LinearGaussianSSabcd,
                 y::Union(AbstractMatrix, TimeVaryingParam),
                 x0::Union(AbstractVector, MvNormal))
    # Make sure y has obs in columns. Compute number obs and dim(y)
    y, T, _ = _get_T_ny_fixy(m, y)

    # compute the initial state and covariance
    x, P = _get_x0_P0(x0)

    # make sure dimensionality of initial state is correct
    nx = length(x0)
    @assert nx == size(m.A[1], 1) "Initial distribution incompatible with system"

    # allocate space for filter output
    filtered = Array(KFstep, T)

    # fill first observation
    filtered[1] = kfilter_step(m, x, P, 1, _get_yt(y, 1))

    # run the filter for the rest of the periods
    for t=2:T
        y_t = _get_yt(y, t)  # this period's observation
        x, P = filtered[t-1].x_f, filtered[t-1].P_f  # Prev updated state/cov
        filtered[t] = kfilter_step(m, x, P, t, y_t)
    end
    filtered
end

loglik(s::AbstractVector{KFstep}) = sum([x.ll for x in s])

## --------------- ##
#- Particle filter -#
## --------------- ##


function pfilter(m::LinearGaussianSSabcd, y, x0::MvNormal;
                 cloudsize=5000)
    # unpack model parameters
    A, B, C, D = m.A, m.B, m.C, m.D

    # Make sure y has obs in columns. Compute number obs and dim(y)
    y, T, ny = _get_T_ny_fixy(m, y)

    # Useful Parameters
    nx = length(x0)
    randY = MvNormal(zeros(ny), eye(ny))

    # Allocate Space
    filtered_x = Array(Float64, nx, cloudsize, T+1)
    xtwiddle = Array(Float64, nx, cloudsize)
    proby_x = Array(Float64, cloudsize)
    weights = Array(Float64, cloudsize)
    inds = Array(Int, cloudsize)
    loglik = 0.0

    filtered_x[:, :, 1] = rand(x0, cloudsize)

    for t=1:T
        # pull out observation
        y_t = _get_yt(y, t)

        # compute H^{-1} ( in this case)
        Dt_inv = inv(D[t])
        det_J = det(Dt_inv)
        nx_eps = size(B[t], 2)

        # Move cloud forward and compute probability of each event
        xtwiddle[:, :] = A[t]*filtered_x[:, :, t] + B[t]*randn(nx_eps, cloudsize)
        proby_x[:] = det_J*pdf(randY, Dt_inv*(y_t .- C[t]*xtwiddle))

        # Normalize Weights
        probx_y_sum = sum(proby_x)
        weights = proby_x ./ probx_y_sum

        # Resample
        wsample!(1:cloudsize, weights, inds)
        filtered_x[:, :, t+1] = xtwiddle[:, inds]

        # Loglikelihood
        loglik = loglik + log(probx_y_sum / cloudsize)

    end

    return filtered_x, loglik
end
