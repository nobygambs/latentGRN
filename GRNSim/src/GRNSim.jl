module GRNSim

import Pkg
import Distributed
import Dates
import Distributions
import LinearAlgebra
import Plots
import ProgressBars
import Random
import SharedArrays
import StatsBase

# Define greet function
greet() = print("Welcome to GRNSim!\n")

# Define parallelTest function to initialize project
function parallelTest(procs)
    project_path = splitdir(Pkg.project().path)[1]
    Distributed.@everywhere procs begin
        Main.eval(quote
            import Pkg
            Pkg.activate($$project_path)
			import GRNSim
        end)
    end

    Distributed.pmap(_->println("Hello from " * string(Distributed.myid())), Distributed.WorkerPool(procs), 1:6)
end

# Define parallelTest function to initialize project
function parallelTest(num_procs::Integer)
    procs = Distributed.addprocs(num_procs)
    try
        parallelTest(procs)
    finally
        Distributed.rmprocs(procs)
    end
end

function RunGRN(; n_tf = 100, n_iter = 1000, sparsity = 0, bias = -1, ncores = 1, t_max = 500, dt = 0.1, beta_0 = 1, delta_0 = 0.125, noise_start = 20, noise_end = 40, plots_on = false)

    ## Define interaction matrix, parameters, functions
    #### Interaction matrix
    M = randn(n_tf, n_tf) .+ bias
    zero_idx = StatsBase.sample(1:length(M), round(Int, sparsity*length(M)); replace = false, ordered = true)
    M[zero_idx] .= 0
    M[LinearAlgebra.diagind(M)] .= 1

    #### Model parameters
    ###### Affinities
    beta = beta_0*ones(n_tf, 1)
    ###### Decay rates
    delta = delta_0*ones(n_tf, 1)

    #### Model functions
    ###### Activity
    f(x, beta) = (x ./ (beta .+ x))
    ###### Random noise
    noise(temp, n_tf)  = (temp .* exp.(randn(n_tf, 1)))

    ## Set up integration
    ###### Time vector
    t = collect(0:dt:t_max)
    ###### Time domain of random noise
    temp = zeros(1, length(t))
    temp[t .>= noise_start .&& t .<= noise_end] .= 0.1
    ###### Steady-state storage list
    #y_fin = zeros(n_tf, n_iter);
    y_fin = SharedArrays.SharedArray{Float64}(n_tf, n_iter)

    ## Iterate n_iter times
    Distributed.@sync Distributed.@distributed for iter = 1:n_iter

        ## Reset simulation
        y0 = zeros(n_tf, 1)
        y = zeros(n_tf, length(t))
        y[:, 1] = y0

        ## Integrate
        for ii in 2:length(t)

            ## Euler-Maruyama
            dy = dt .* (M * f(y[:, ii-1], beta) - delta .* y[:, ii-1]) .+ sqrt(dt) .* noise(temp[ii], n_tf)
            y_new = y[:, ii-1] .+ dy
            y_new[y_new .< 0] .= 0
            y[:, ii] = y_new

        end

        ## Store final value / steady-state
        y_fin[:, iter] = y[:, end]

    end

    return y_fin

end

export greet
export parallelTest
export RunGRN

end
