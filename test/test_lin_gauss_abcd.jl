using StateSpace
using Distributions

# read in python data
# const y = readdlm("/Users/grantlyon/Spencer/temp/y.csv", ',')
# const ll_py = -223.02877677509935

const N = 2

# generate state space matrices
const F = eye(N)
const V = .25 .* eye(N)
const G = eye(N)
const W = 0.25 .* eye(N)
const A, B, C, D = F, sqrtm(V), G, sqrtm(W)

# initial distribution
const x0_dist = MvNormal(ones(N), 0.1 .* eye(N))

# construct state space
const lgs = LinearGaussianSSabcd(A, B, C, D)

x, y = simulate(lgs, 100, x0_dist)

ks = kfilter(lgs, y, x0_dist);

function test_kfilter()
    # run once to warm up
    ll_k = loglik(kfilter(lgs, y, x0_dist))

    # now time it
    tic()
    ll_k = loglik(kfilter(lgs, y, x0_dist))
    println("Execution time for kfilter: $(toq())")

    err = abs(ll_k - ll_py)
    println("Absolute difference between py and jl kf log_lik: $err")
end


function test_pfilter()
    # run once to warm up
    ll_p = pfilter(lgs, y, x0_dist, cloudsize=100)[2]
    ll_k = loglik(kfilter(lgs, y, x0_dist))

    cloudsize = 40000

    # now time it
    tic()
    ll_p = pfilter(lgs, y, x0_dist, cloudsize=cloudsize)[2]
    @show ll_p
    println("Execution time for pfilter(n=$cloudsize): $(toq())")

    err = abs(ll_p - ll_k)
    pct_err = (err / abs(ll_k)) * 100
    @printf "Absolute difference between kfilter, pfilter: %1.3e (%.2e %%)" err pct_err
end

test_kfilter()
test_pfilter()

