module StateSpace

import Base: mean, filter, show, convert


export
    # types
    FilteredState,
    LinearGaussianSSM,
    NonlinearGaussianSSM,
    TimeVaryingParam,

    # functions
    bw_sampler,
    filter,
    fwfilter_bwsampler,
    observe,
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

end # module
