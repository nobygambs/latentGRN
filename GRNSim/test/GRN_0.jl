## Set up workspace
using Pkg
Pkg.activate("GRNSim/")
import GRNSim

using Distributed
using Distributions
using LinearAlgebra
using Plots
using ProgressBars
using Random
using SharedArrays
using StatsBase

Random.seed!(2024)

## Set simulation arguments
#### Number of iterations
n_iter = 1000
#### Plot each trajectory?
plots_on = false

#### Interaction matrix
n_tf = 100
sparsity = 0
bias = -1

#### Integrator
t_max = 500
dt = 0.1

## Define interaction matrix, parameters, functions
#### Interaction matrix
M = randn(n_tf, n_tf) .+ bias
zero_idx = sample(1:length(M), round(Int, sparsity*length(M)); replace = false, ordered = true)
M[zero_idx] .= 0
M[diagind(M)] .= 1

#### Model parameters
###### Affinities
beta = ones(n_tf, 1)
###### Decay rates
delta = 0.125*ones(n_tf, 1)

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
temp[t .>= 20 .&& t .<= 40] .= 0.1
###### Steady-state storage list
y_fin = zeros(n_tf, n_iter);

## Iterate n_iter times
for iter in ProgressBar(1:n_iter)
    print(iter)

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

    if plots_on
        p_iter = plot(t, y', legend = false)
        display(p_iter)
        sleep(1)
    end

end

## PCA of equlibria (final states)
fin_svd = svd(y_fin)
p_pca = plot(fin_svd.V[:, 1], fin_svd.V[:, 2], seriestype=:scatter, legend = false)
display(p_pca)
sleep(10000)






