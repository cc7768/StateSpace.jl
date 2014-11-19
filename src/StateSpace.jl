module StateSpace

import Base: mean, filter, show, convert, getindex


export
    # types
    FilteredState,
    LinearGaussianSSM,
    LinearGaussianSSabcd,
    NonlinearGaussianSSM,
    TimeVaryingParam,

    # functions
    bw_sampler,
    filter,
    fwfilter_bwsampler,
    kfilter,
    loglik,
    observe,
    pfilter,
    predict,
    show,
    simulate,
    update!,
    update

include("common.jl")
include("util.jl")
include("models/linear_gaussian.jl")
include("ExtendedKalmanFilter.jl")
include("KalmanFilter.jl")
include("temp_lingaus_abcd.jl")

end # module
