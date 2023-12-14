## Set up workspace
using Pkg
Pkg.activate("GRNSim/")

using Distributed
@everywhere using Dates
@everywhere using Distributions
@everywhere using LinearAlgebra
@everywhere using Plots
@everywhere using ProgressBars
@everywhere using Random
@everywhere using SharedArrays
@everywhere using StatsBase

## Define GRNSim function
@everywhere function RunGRN(;n_tf = 100, n_iter = 1000, sparsity = 0, bias = -1, rng_seed = 2024, ncores = 1, t_max = 500, dt = 0.1, beta_0 = 1, delta_0 = 0.125, noise_start = 20, noise_end = 40, plots_on = false)

    Random.seed!(rng_seed)
    addprocs(ncores)

    ## Define interaction matrix, parameters, functions
    #### Interaction matrix
    M = randn(n_tf, n_tf) .+ bias
    zero_idx = sample(1:length(M), round(Int, sparsity*length(M)); replace = false, ordered = true)
    M[zero_idx] .= 0
    M[diagind(M)] .= 1

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
    y_fin = SharedArray{Float64}(n_tf, n_iter)

    ## Iterate n_iter times
    @sync @distributed for iter = 1:n_iter

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

t_start = now()
result = RunGRN(; ncores = 8, n_tf = 100, n_iter = 1000)
t_end = now()
elapsed = canonicalize(t_end - t_start)
print("\n"); print(elapsed); print("\n")

## PCA of equlibria (final states)
fin_svd = svd(result)
p_pca = plot(fin_svd.V[:, 1], fin_svd.V[:, 2], seriestype=:scatter, legend = false)
display(p_pca)





