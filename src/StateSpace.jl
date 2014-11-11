module StateSpace

import Base: mean, filter, show


export
    FilteredState,
    LinearGaussianSSM,
    NonlinearGaussianSSM,
    bw_sampler,
    filter,
    fwbw_sampler,
    observe,
    predict,
    show,
    simulate,
    update!,
    update

include("util.jl")
include("common.jl")
include("ExtendedKalmanFilter.jl")
include("KalmanFilter.jl")

end # module
