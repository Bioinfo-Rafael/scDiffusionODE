# Official ddHodge workflow only; no local Hodge or potential solver.
using ddHodge, HDF5, Graphs, Random, LinearAlgebra, SparseArrays, Pkg
length(ARGS) == 2 || error("Usage: run_ddhodge.jl input.h5 output.h5")
isfile(ARGS[2]) && error("Refusing to overwrite output")
expected_rev = "d5d6b09431c0ccbb16e0b63414a5038511c19b85"
dep = Pkg.dependencies()[Base.PkgId(ddHodge).uuid]
dep.git_revision == expected_rev || error("Unexpected ddHodge revision: $(dep.git_revision)")
X, V, k, rdim, seed, maxiter, ssa = h5open(ARGS[1], "r") do f
    n, d = Int.(read(f["shape_ND"]))
    # Flat C-order Python (N,D) equals Julia column-major (D,N); no HDF5 axis ambiguity.
    x = reshape(read(f["X_C_flat"]), d, n)
    v = reshape(read(f["V_C_flat"]), d, n)
    x, v, Int(read(f["k"])), Int(read(f["rdim"])), Int(read(f["seed"])),
        Int(read(f["maxiter"])), Bool(read(f["ssa"]))
end
Random.seed!(seed)
g, kdtree, = kNNGraph(X, k)
is_connected(g) || error("Disconnected graph: potentials have incomparable additive constants")
ddh = ddHodgeWorkflow(g, X, V; rdim=rdim, maxiter=maxiter, ssa=ssa)
all(isfinite, ddh.u) || error("Nonfinite official potential")
# Check the official least-squares stationarity; curl may make the edge residual nonzero.
r = ddh.d0 * ddh.u + ddh.w
stationarity = norm(ddh.d0' * r) / max(norm(ddh.d0' * ddh.w), eps(Float64))
stationarity <= 1e-5 || error("Official potential solve stationarity failed: $stationarity")
h5open(ARGS[2], "w") do f
    f["u"] = ddh.u
    f["w"] = ddh.w
    f["edge_src_zero"] = src.(edges(g)) .- 1
    f["edge_dst_zero"] = dst.(edges(g)) .- 1
    f["div"] = ddh.div
    f["rot"] = ddh.rot
    f["stationarity_relative"] = stationarity
    f["edge_residual_norm"] = norm(r)
    f["julia_version"] = string(VERSION)
    f["ddhodge_version"] = string(dep.version)
    f["ddhodge_commit"] = dep.git_revision
    f["shape_DN"] = collect(size(X))
end
