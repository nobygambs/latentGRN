## Set up workspace
import Pkg
Pkg.activate("GRNSim/")

## Run GRN simulator
t_start = now()
result = RunGRN(; ncores = 8)
t_end = now()
elapsed = canonicalize(t_end - t_start)
print("\n"); print(elapsed); print("\n\n")