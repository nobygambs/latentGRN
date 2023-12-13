#using Distributed
#@everywhere using SharedArrays
#@everywhere using Dates
#ncores = 2
#addprocs(ncores)

#print(Dates.format(now(), "HH:MM:SS"))
#print("\n")
#nheads = @distributed (+) for i = 1:2000000000
#    Int(rand(Bool))
#end
#print(Dates.format(now(), "HH:MM:SS"))
#print("\n")



#@everywhere a = SharedArray{Float64}(10)
#@distributed for i = 1:10
#    a[i] = i
#end
#print(a)

using Dates
using Distributed
using LinearAlgebra
ncores = 1
addprocs(ncores)

M = Matrix{Float64}[rand(3000,3000) for i = 1:10];

t_start = now()
result = pmap(svdvals, M)
t_end = now()
elapsed = canonicalize(t_end - t_start)

print("\n")
print(elapsed)
print("\n")

#print(result)

#@everywhere using SharedArrays

#v = SharedArray{Float64}(10)
#@everywhere function par_asgn_test(v)
#@distributed for i = 1:length(v)
#    v[i] = i
#end
#end
#print(fetch(v))

#@everywhere n = 10
#@everywhere arr = SharedArray{Float64}(n)

#t_start = now()
#result_address = par_asgn_test(arr)
#t_end = now()
#elapsed = canonicalize(t_end - t_start)

#fetch(result_address)

#print(arr)
#print("\n")
#print(elapsed)
#print("\n")
#print(result_address)
#print("\n\n\n\n")