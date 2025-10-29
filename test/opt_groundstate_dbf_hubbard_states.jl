using DBF
using PauliOperators
using Printf
using Random
using LinearAlgebra
using Test
using Plots

"""
    particle_ket(N::Int, Nparticles::Int; mode=:first)

Return a `Ket` of length `N` with exactly `Nparticles` ones.

`mode` options:
  • `:first`   → first `Nparticles` sites occupied (|111000...>)
  • `:random`  → random positions for the 1s
  • `:alternate` → alternating pattern |101010...> (if Nparticles ≈ N/2)

If the alternating pattern has more or fewer 1s than `Nparticles`,
the result is truncated or padded with zeros.
"""
function particle_ket(N::Int, Nparticles::Int; mode=:first)
    @assert 0 ≤ Nparticles ≤ N "Number of particles must be between 0 and N"
    occ = zeros(Int, N)

    if mode == :random
        occ[randperm(N)[1:Nparticles]] .= 1

    elseif mode == :alternate
        occ[1:2:N] .= 1  # pattern 1,0,1,0,...
        # adjust to requested number of particles
        n1 = count(==(1), occ)
        if n1 > Nparticles
            # trim some 1s from the end
            idxs = findall(==(1), occ)
            occ[idxs[(Nparticles+1):end]] .= 0
        elseif n1 < Nparticles
            # add 1s where there are zeros
            idxs = findall(==(0), occ)
            occ[idxs[1:(Nparticles - n1)]] .= 1
        end

    else  # :first
        occ[1:Nparticles] .= 1
    end

    return Ket(occ)
end

function run(; Nparticles, U=U, threshold=1e-3, wmax=nothing, wtype=0)
    # Parameters for Hubbard model
    Lx = 8
    Ly = 1
    Nsites = Lx * Ly
    N = 2 * Nsites   # 2 spin states per site
    t = 1.0
    U = 1.0
    H = DBF.fermi_hubbard_2D(Lx, Ly, t, U)
    #H = DBF.fermi_hubbard_2D_snake(Lx, Ly, t, U; snake_ordering=true)
    #H = DBF.hubbard_model_1D(Nsites, t, U)

    println(" Original H:")
    #display(H)

    # Initial state: Defined by number of particles
    ψ = particle_ket(N, Nparticles, mode=:random)
    #ψ = particle_ket(N, Nparticles, mode=:first)
    #ψ = particle_ket(N, Nparticles, mode=:alternate)
    #ψ = Ket{N}(153)
    display(ψ)
    e0 = expectation_value(H,ψ)
   
    @printf(" E0 = %12.8f\n", e0)

    display(norm(H))
    display(norm(diag(H)))
    res = DBF.dbf_groundstate_test(H, ψ, 
                                verbose=1, 
                                max_iter=120, conv_thresh=1e-6, 
                                evolve_coeff_thresh=threshold,
                                grad_coeff_thresh=1e-6,
                                energy_lowering_thresh=1e-6,
                                clifford_check=true,
                                compute_pt2_error=false,
                                max_rots_per_grad=100)


    println("res", keys(res))
    
    #println(" New H:")
    #display(norm(H))
    #display(norm(diag(H)))
    #println("Exact Ground State Energy: ", groundE)
    @show DBF.variance(H,ψ)    
end


#= 
   Test set performing comparisons of the quality of the DBF-OPT ground state solver for the Hubbard model.
   Here we forculs solely on the 2D Hubbard model on a 2x2 lattice. which can be solved exactly.
   
   - Error comparison and performance with comparison to exact diagonalization
   - Comparison of different weight and coeff thresholding pruning strategies
   - How accurate is this approach with differen choices of the parameters?

   Comparisons in different coupling regimes:
      Weak coupling regime: t=0.1, U=0.001
      Middle coupling regime: t=0.1, U=0.09
      Strong coupling regime: t=0.1, U=0.5   
=#
Nparticles = [8, 6, 4]
threshs = [1e-2]#, 1e-3]#, 1e-4, 1e-5, 1e-6, 1e-8]
Pweights = [2, 3, 4, 5, 6, 7, 8]
Mweights = [2, 3, 4, 5, 6, 7, 8]

Pauli_errors = Dict{String,Float64}()
Majorana_errors = Dict{String,Float64}()

variance_list = Float64[]
absolute_errors = Float64[]
absolute_errors_Majorana = Float64[]
dbfEs_list = Vector{Float64}[]
nterms_list = Vector{Int}[]
loss_list = Vector{Float64}[]

for Np in Nparticles
    println("========================================")
    println(" Particle Number = ", Np)
    println("========================================")
    println("---- Coefficient Thresholding Only ----")
    for thresh in threshs
        println("  Coefficient Threshold = ", thresh)
        res = run(Nparticles=Np, U=1.0, threshold=thresh, wmax=nothing, wtype=0)
        
    end
end