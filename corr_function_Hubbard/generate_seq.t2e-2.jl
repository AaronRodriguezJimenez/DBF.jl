using DBF
using PauliOperators
using Printf
using Random
using LinearAlgebra
using JLD2

# Assuming we already have the sequence as an occ vector
function get_state_sequence(N,occ)
    g = Vector{PauliBasis{N}}([])
    a = Vector{Float64}([])
    for i in 1:N
        if occ[i] == 1
            push!(g, PauliBasis(Pauli(N, X=[i])))
            push!(a, π)
        end
    end
    return g, a 
end

function run()

     # Parameters for Hubbard model
    Lx = 10
    Ly = 1
    Nsites = Lx * Ly
    N = 2 * Nsites   # 2 spin states per site
    t = 1.0
    U = 1.0
    H = DBF.fermi_hubbard_2D(Lx, Ly, t, U)
    #H = DBF.fermi_hubbard_2D_zigzag(Lx, Ly, t, U)
    
    Nparticles = Nsites  # Half-filling
    ψ, occ = DBF.particle_ket(N, Nparticles, 0.0; mode=:Neel, flavor=:Aup)
    display(ψ)
    println(" Occupation: ", occ)

    # Transform H to make |000> the most stable bitstring
    for i in 1:N
        if occ[i] == 1
            H = Pauli(N, X=[i]) * H * Pauli(N, X=[i])
        end
    end 
    

#    g,a = get_state_sequence(N, occ)  
#    for (gi, ai) in zip(g,a)
#        H = evolve(H, gi, ai)
#    end
   
    ψ = Ket{N}(0)
    display(ψ)
    e0 = expectation_value(H,ψ)
  
    @printf(" Reference = %12.8f\n", e0)
    flush(stdout) 
    println("\n ########################")
    @time res = DBF.dbf_groundstate(H, ψ,
                                    verbose=1,
                                    max_iter=100, 
                                    conv_thresh=1e-3,
                                    evolve_coeff_thresh=1e-4,
                                    grad_coeff_thresh=1e-6,
                                    energy_lowering_thresh=1e-6,
                                    max_rots_per_grad=100, 
                                    checkfile = "t1e-2_hub2x2")
    
    
end


run()